"""
Time series forecasting for BTC/USD. 
Sourced 1h timeseries df from Gemini.
Sourced daily fear and greed index from alternative.me.

Steps
1. Download and prepare data
2. Select target variable (close)
3. Feature engineering (lagged close, moving averages, rsi, volatility, fear and greed index)
4. Train-test split
5. Model training
6. Evaluate and plot

Minor Improvements
- Work on getting the RMSE and MAE down
- Tune ARIMA order: consider experimenting with different values of p, d and q to see if more 
apropriate model can be fitted. You can use GridSearch or a similar approach to find the 
best combination of these parameters.
- Aim for 10-20 features.
- I can't imagine the way I calculate the RSI is very performant. 
- Drop or transform less significant features like `parkinson_volatility`
- Explore adding more leading indicators (e.g. MACD, Bollinger Bands)
- Address heteroskedasticity with GARCH to better model volatility clustering
- Use LSTM or GRU models for long-sequence time series if computational resources permit

Major Improvements
- You could combine this model with machine learning models like Random Forests or XGBoost, 
which might capture non-inear patterns better.
- Use other advanced models such as GARCH for volatility modeling or LSTM (Long Short-Term Memory) 
Networks for capturing long-term dependencies in time-series forecasting.
- For a more sophisticated model I should be using 10-20 features.

Current version: 1.1.0
Last updated: 29/11/2024 19:54
Author: Ian Summerlin
"""
import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings("ignore")

BTC_PRICEDATA_1H_URL = "https://www.cryptodatadownload.com/cdd/Gemini_BTCUSD_1h.csv"
BTC_PRICEDATA_PATH = "BTCUSD_1H.csv"
BTC_SENTIMENTDATA_URL = "https://api.alternative.me/fng/?limit=250000&format=csv&date_format=kr"
BTC_SENTIMENTDATA_PATH = "BTC_sentiment.csv"

BTC_CLOSING_PRICE_CHART_PATH = "bitcoin-closing-price.png"
BTC_ACTUAL_VS_PREDICTED_CHART_PATH = "bitcoin-actual-vs-predicted-price.png"

def print_divider():
    """
    print_divider
    Print a divider to improve organisation of print statements
    """
    print("------------------------")

def download_btcusd_file():
    """
    download_btcusd_file
    This file will check if the BTC/USD pricedata exists
    - if it does: it will skip
    - if it doesn't: it will download
    
    Date range
    2015-10-08 13:00:00 - 2024-11-24 23:00:00
    
    NOTE: don't forget to delete the first line of this CSV as it contains a URL.
    """
    
    # Check if the file already exists
    print_divider()
    print("Downloading BTCUSD price df...")
    if os.path.exists(BTC_PRICEDATA_PATH):
        print(f"The file '{BTC_PRICEDATA_PATH}' already exists.")
        return
    
    # Send a GET request to the URL
    print(f"Downloading the file from {BTC_PRICEDATA_1H_URL}...")
    response = requests.get(BTC_PRICEDATA_1H_URL)

    # Check if the request was successful (status code 200)
    if response.status_code != 200:
        print(f"Failed to retrieve the file. Status code: {response.status_code}")
        
    # Save the content to a file
    with open(BTC_PRICEDATA_PATH, 'wb') as file:
        file.write(response.content)
        
    print(f"File downloaded and saved as {BTC_PRICEDATA_PATH}")

def download_btc_sentiment_data():
    """
    download_btc_sentiment_data
    This file will check if the BTC/USD sentiment data exists
    - if it does: it will skip
    - if it doesn't: it will download
    
    Date range
    2018-01-02 - 2024-11-24
    
    NOTE: Don't forget to trim the results manually to the date range above and 
    sort out the headers.
    """
    
    # Check if the file already exists
    print_divider()
    print("Downloading BTCUSD sentiment df...")
    if os.path.exists(BTC_SENTIMENTDATA_PATH):
        print(f"The file '{BTC_SENTIMENTDATA_PATH}' already exists.")
        return
    
    # Send a GET request to the URL
    print(f"Downloading the file from {BTC_SENTIMENTDATA_URL}...")
    response = requests.get(BTC_SENTIMENTDATA_URL)
    
    # Check if the request was successful (status code 200)
    if response.status_code != 200:
        print(f"Failed to retrieve the file. Status code: {response.status_code}")
        
    # Save the content to a file   
    with open(BTC_SENTIMENTDATA_PATH, 'wb') as file:
        file.write(response.content)
        
    print(f"File downloaded and saved as {BTC_SENTIMENTDATA_PATH}")

def merge_sentiment_data(df):
    """
    merge_sentiment_data
    This function will merge the sentiment data with the BTCUSD price data.
    
    NOTE: The price data is hourly and the sentiment data is daily, so we will forward fill
    """
    print_divider()
    print("Merging sentiment data with BTCUSD price data...")
    if not os.path.exists(BTC_SENTIMENTDATA_PATH):
        print(f"The file '{BTC_SENTIMENTDATA_PATH}' does not exist.")
        return df
    
    # Read the sentiment data
    sentiment_df = pd.read_csv(BTC_SENTIMENTDATA_PATH, index_col=0)
    
    # Convert the date to datetime object
    sentiment_df.index = pd.to_datetime(sentiment_df.index)
    
    # Merge the sentiment data with the BTCUSD price data
    df = df.merge(sentiment_df, how='left', left_index=True, right_index=True)
    
    # Forward fill sentiment data for hourly entries within the same day
    df[['fng_value', 'fng_classification']] = df[['fng_value', 'fng_classification']].fillna(method='ffill')
    df = df.dropna(subset=['fng_value', 'fng_classification'])
    
    print("Sentiment data merged with BTCUSD price data")
    return df

def generate_btcusd_closing_price_graph(df):
    """
    generate_btcusd_closing_price_graph
    This will generate the BTCUSD closing price graph for the data downloaded
    by the function `download_btcusd_file`. This is used for visualising the price
    data.
    """
    print_divider()
    print("Handle generate Bitcoin Closing Price graph...")
    if os.path.exists(BTC_CLOSING_PRICE_CHART_PATH):
        print(f"The file '{BTC_CLOSING_PRICE_CHART_PATH}' already exists.")
        print_divider()
        return
    
    plt.figure(figsize=(10, 6))
    df['close'].plot(title='Bitcoin Closing Price')
    plt.savefig('bitcoin-closing-price.png')
    
    print (f"Bitcoin Closing Price graph generated and saved as {BTC_CLOSING_PRICE_CHART_PATH}")
    print_divider()
    
def main():
    """
    main
    This will handle all of the steps to download, format, add features, train and evaluate
    the price analysis model of BTCUSD
    """
    
    print("Preparing model...")
    
    """
    Download and prepare initial data
    """

    # Download and import df
    download_btcusd_file()
    df = pd.read_csv(BTC_PRICEDATA_PATH, index_col=0)

    # Convert date to datetime object, drop column `symbol` and sort
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.drop(columns=['symbol'], inplace=True)
    df = df.sort_index()
    
    # Merge sentiment data
    download_btc_sentiment_data()
    df = merge_sentiment_data(df)
    
    # Select target variable
    target = df['close']

    # Handle missing df
    df = df.ffill()

    # If hourly df is too noisy you can resamplet to a higher timeframe (e.g. daily)
    # df = df.resample('D').mean() 

    generate_btcusd_closing_price_graph(df)

    """
    Feature engineering
    Basic time-series models like ARIMA work on raw `close` df, you can improve the accuracy
    by adding features
    """

    # Generate lagged version of `close`
    for lag in [90, 120]:
        df[f"close_lag_{lag}"] = df['close'].shift(lag)
        
    # Moving averages (7 hour and 24 hour moving average)
    df['MA_7'] = df['close'].rolling(window=7).mean()
    df['MA_24'] = df['close'].rolling(window=24).mean()
    
    # RSI
    window_length = 14 # Typical RSI window
    df['change'] = df['close'].diff()
    df['gain'] = df['change'].apply(lambda x: x if x > 0 else 0)
    df['loss'] = df['change'].apply(lambda x: -x if x < 0 else 0)
    df['avg_gain'] = df['gain'].rolling(window=window_length, min_periods=1).mean()
    df['avg_loss'] = df['loss'].rolling(window=window_length, min_periods=1).mean()
    df['rsi'] = 100 - (100 / (1 + (df['avg_gain'] / df['avg_loss'])))

    # Volatility measured using rolling standard deviation
    df['volatility_24'] = df['close'].rolling(window=24).std()
    
    # Volatility measured with ewma
    df['volatility_ewma_24'] = df['close'].ewm(span=24).std()
    
    # Parkinson volatility
    df['parkinson_volatility'] = (1 / (4 * np.log(2))) * ((np.log(df['high'] / df['low'])) ** 2).rolling(window=24).mean()
    
    # Average true range
    # Shift the 'close' column once before using it in the lambda function
    df['shifted_close'] = df['close'].shift(1)

    # Now, use the shifted column inside the apply function
    df['true_range'] = df[['high', 'low', 'close', 'shifted_close']].apply(
        lambda row: max(
            row['high'] - row['low'], 
            abs(row['high'] - row['shifted_close']), 
            abs(row['low'] - row['shifted_close'])
        ), axis=1
    )    
    df['atr_24'] = df['true_range'].rolling(window=24).mean()
    
    # Handle fear and greed index
    df['fng_classification_encoded'] = df['fng_classification'].astype('category').cat.codes
    
    # Ensure to drop any rows with missing values due to lagging or encoding
    df = df.dropna()
    
    """
    Train-test split (80% training data, 20% testing data)
    """
    train_size = int(len(df) * 0.8)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    exog_columns = [
        'close_lag_90', 
        'close_lag_120', 
        'MA_7', 
        'MA_24', 
        'rsi',
        'volatility_24', 
        'volatility_ewma_24',
        'parkinson_volatility',
        'atr_24',
        'fng_value',
        'fng_classification_encoded'
    ]

    X_train = train[exog_columns]
    y_train = train['close']
    X_test = test[exog_columns]
    y_test = test['close']
    
    """
    Model training
    """

    # Fit ARIMA (adjust p, d, q values) using exogenous variables
    p = 5 # AR lag
    d = 1 # Differencing
    q = 0 # MA lag
    model = ARIMA(y_train, order=(p, d, q), exog=X_train) 
    arima_model = model.fit()
    print(arima_model.summary())

    # Forecast for the test period
    forecast = arima_model.forecast(steps=len(test), exog=X_test)

    # Evaluate (compare predictions with actual `close` prices using RMSE)
    rmse = np.sqrt(mean_squared_error(y_test, forecast))
    mae = mean_absolute_error(y_test, forecast)
    print('RMSE:', rmse)
    print ('MAE:', mae)
    print_divider()

    # Plot predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.index, y_test, label='Actual Prices', color='blue')
    plt.plot(y_test.index, forecast, label='Forecasted Prices', color='orange')
    plt.title('Bitcoin Price Prediction using ARIMA')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.savefig(BTC_ACTUAL_VS_PREDICTED_CHART_PATH)
    print(f"Chart saved as {BTC_ACTUAL_VS_PREDICTED_CHART_PATH}")

# Execute script
if __name__ == "__main__":
    main()