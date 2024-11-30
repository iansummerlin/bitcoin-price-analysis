"""
Time series forecasting for BTC/USD. 

Sourced 1h timeseries df from Gemini.
Sourced daily fear and greed index from alternative.me.

Steps:
1. Download and prepare data
2. Select target variable (close)
3. Feature engineering (lagged close, moving averages, rsi, volatility, volume, fear and greed index)
4. Train-test split
5. Model training
6. Evaluate and plot

Minor Improvements
- Work on getting the RMSE and MAE down
- Tune ARIMA order: consider experimenting with different values of p, d and q to see if more 
apropriate model can be fitted. You can use GridSearch or a similar approach to find the 
best combination of these parameters.
- Aim for 10-20 features.
- Drop or transform less significant features like `parkinson_volatility`
- Explore adding more leading indicators (e.g. MACD, Bollinger Bands)
- Address heteroskedasticity with GARCH to better model volatility clustering
- Use LSTM or GRU models for long-sequence time series if computational resources permit

Major Improvements
- You could combine this model with machine learning models like Random Forests or XGBoost, 
which might capture non-linear patterns better.
- Use other advanced models such as GARCH for volatility modeling or LSTM (Long Short-Term Memory) 
Networks for capturing long-term dependencies in time-series forecasting.
- For a more sophisticated model I should be using 10-20 features.

Misc Improvements
- Download merged csv so it doesn't need to be generated on each run of the model
- Split out the feature calculations into different functions > create a folder called
features > include each function as a file inside the folder and import here.

Current version: 1.1.1
Last updated: 29/11/2024 19:54
Author: Ian Summerlin
"""
import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time  # Add this import at the top

import warnings
warnings.filterwarnings("ignore")

# Constants
BTC_PRICEDATA_1H_URL = "https://www.cryptodatadownload.com/cdd/Gemini_BTCUSD_1h.csv"
BTC_PRICEDATA_PATH = "BTCUSD_1H.csv"
BTC_SENTIMENTDATA_URL = "https://api.alternative.me/fng/?limit=250000&format=csv&date_format=kr"
BTC_SENTIMENTDATA_PATH = "BTC_sentiment.csv"
BTC_CLOSING_PRICE_CHART_PATH = "bitcoin-closing-price.png"
BTC_ACTUAL_VS_PREDICTED_CHART_PATH = "bitcoin-actual-vs-predicted-price.png"

def print_divider():
    """Print a divider to improve organisation of print statements."""
    print("------------------------")

def download_file(url, path):
    """
    Download a file from a URL if it doesn't already exist.

    Parameters:
    url (str): The URL to download the file from.
    path (str): The local path where the file will be saved.
    """
    print_divider()
    print(f"Downloading file from {url}...")
    if os.path.exists(path):
        print(f"The file '{path}' already exists.")
        return
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve the file. Status code: {response.status_code}")
        return
    
    with open(path, 'wb') as file:
        file.write(response.content)
    print(f"File downloaded and saved as {path}")

def download_data():
    """Download both BTC price and sentiment data."""
    download_file(BTC_PRICEDATA_1H_URL, BTC_PRICEDATA_PATH)
    download_file(BTC_SENTIMENTDATA_URL, BTC_SENTIMENTDATA_PATH)

def merge_sentiment_data(df):
    """
    Merge sentiment data with BTCUSD price data.

    Parameters:
    df (DataFrame): The DataFrame containing BTCUSD price data.

    Returns:
    DataFrame: The merged DataFrame with sentiment data.
    """
    print_divider()
    print("Merging sentiment data with BTCUSD price data...")
    if not os.path.exists(BTC_SENTIMENTDATA_PATH):
        print(f"The file '{BTC_SENTIMENTDATA_PATH}' does not exist.")
        return df
    
    sentiment_df = pd.read_csv(BTC_SENTIMENTDATA_PATH, index_col=0)
    sentiment_df.index = pd.to_datetime(sentiment_df.index)
    df = df.merge(sentiment_df, how='left', left_index=True, right_index=True)
    df[['fng_value', 'fng_classification']] = df[['fng_value', 'fng_classification']].fillna(method='ffill')
    df = df.dropna(subset=['fng_value', 'fng_classification'])
    df = df.ffill() # Fill the daily sentiment data across the hourly price data
    print("Sentiment data merged with BTCUSD price data")
    return df

def generate_btcusd_closing_price_graph(df):
    """
    Generate the BTCUSD closing price graph.

    Parameters:
    df (DataFrame): The DataFrame containing BTCUSD price data.
    """
    print_divider()
    print("Generating Bitcoin Closing Price graph...")
    if os.path.exists(BTC_CLOSING_PRICE_CHART_PATH):
        print(f"The file '{BTC_CLOSING_PRICE_CHART_PATH}' already exists.")
        return
    
    plt.figure(figsize=(10, 6))
    df['close'].plot(title='Bitcoin Closing Price')
    plt.savefig(BTC_CLOSING_PRICE_CHART_PATH)
    print(f"Bitcoin Closing Price graph generated and saved as {BTC_CLOSING_PRICE_CHART_PATH}")
    print_divider()

def calculate_rsi(df, window_length=14):
    """
    Calculate the Relative Strength Index (RSI).

    Parameters:
    df (DataFrame): The DataFrame containing BTCUSD price data.
    window_length (int): The number of periods to use for the RSI calculation.

    Returns:
    Series: A Series containing the RSI values.
    """
    # Calculate the difference in closing prices
    delta = df['close'].diff()
    
    # Calculate gains and losses
    gain = (delta.where(delta > 0, 0)).rolling(window=window_length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window_length).mean()
    
    # Calculate the Relative Strength (RS)
    rs = gain / loss
    
    # Calculate the RSI
    df['rsi'] = 100 - (100 / (1 + rs))
    return df['rsi']

def calculate_moving_averages(df):
    """
    Calculate moving averages for the closing prices.

    Parameters:
    df (DataFrame): The DataFrame containing BTCUSD price data.

    Returns:
    None: The function modifies the DataFrame in place.
    """
    # Calculate the 7-period moving average
    df['MA_7'] = df['close'].rolling(window=7).mean()
    
    # Calculate the 24-period moving average
    df['MA_24'] = df['close'].rolling(window=24).mean()

def calculate_volatility(df):
    """
    Calculate volatility measures for the closing prices.

    Parameters:
    df (DataFrame): The DataFrame containing BTCUSD price data.

    Returns:
    None: The function modifies the DataFrame in place.
    """
    # Calculate the 24-period rolling standard deviation (volatility)
    df['volatility_24'] = df['close'].rolling(window=24).std()
    
    # Calculate the exponentially weighted moving average of volatility
    df['volatility_ewma_24'] = df['close'].ewm(span=24).std()

def calculate_average_true_range(df):
    """
    Calculate the Average True Range (ATR) for the given DataFrame.

    Parameters:
    df (DataFrame): The DataFrame containing BTCUSD price data.

    Returns:
    None: The function modifies the DataFrame in place.
    """
    # Shift the closing prices to calculate true range
    df['shifted_close'] = df['close'].shift(1)
    
    # Calculate the true range
    df['true_range'] = df[['high', 'low', 'close', 'shifted_close']].apply(
        lambda row: max(
            row['high'] - row['low'], 
            abs(row['high'] - row['shifted_close']), 
            abs(row['low'] - row['shifted_close'])
        ), axis=1
    )    
    
    # Calculate the 24-period rolling mean of the true range to get ATR
    df['atr_24'] = df['true_range'].rolling(window=24).mean()

def calculate_parkinson_volatility(df):
    """
    Calculate the Parkinson volatility for the given DataFrame.

    Parameters:
    df (DataFrame): The DataFrame containing BTCUSD price data.

    Returns:
    Series: A Series containing the Parkinson volatility values.
    """
    # Calculate Parkinson volatility using high and low prices
    df['parkinson_volatility'] = np.sqrt((1 / (4 * np.log(2))) * (df['high'] - df['low'])**2 / df['close'].shift(1)**2)
    return df['parkinson_volatility']

def feature_engineering(df):
    """
    Perform feature engineering on the DataFrame.

    Parameters:
    df (DataFrame): The DataFrame containing BTCUSD price data.

    Returns:
    DataFrame: The DataFrame with engineered features.
    """
    # Generate lagged version of `close`
    for lag in [90, 120]:
        df[f"close_lag_{lag}"] = df['close'].shift(lag)
        
    # Moving averages
    calculate_moving_averages(df)
    
    # RSI
    calculate_rsi(df)
    
    # Volatility
    calculate_volatility(df)
    
    # Average true range
    calculate_average_true_range(df)
    
    # Calculate Parkinson volatility
    calculate_parkinson_volatility(df)

    # Drop rows with missing values
    df = df.dropna()
    
    return df

def train_test_split(df):
    """
    Split the DataFrame into training and testing sets.

    Parameters:
    df (DataFrame): The DataFrame to split.

    Returns:
    tuple: A tuple containing the training DataFrame and the testing DataFrame.
    """
    train_size = int(len(df) * 0.8)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    return train, test

def train_arima_model(train, exog_columns):
    """
    Fit ARIMA model using training data.

    Parameters:
    train (DataFrame): The training DataFrame containing BTCUSD price data.
    exog_columns (list): List of exogenous variable column names to include in the model.

    Returns:
    ARIMA: The fitted ARIMA model.
    """
    p = 5  # AR lag
    d = 1  # Differencing
    q = 0  # MA lag
    
    start_time = time.time()  # Start timer
    model = ARIMA(train['close'], order=(p, d, q), exog=train[exog_columns]) 
    arima_model = model.fit()
    end_time = time.time()  # End timer
    
    print(f"Model training time: {end_time - start_time:.2f} seconds")  # Print training time
    print(arima_model.summary())
    return arima_model

def evaluate_model(arima_model, test, exog_columns):
    """
    Evaluate the ARIMA model and plot predictions vs actual values.

    Parameters:
    arima_model (ARIMA): The fitted ARIMA model.
    test (DataFrame): The testing DataFrame containing BTCUSD price data.
    exog_columns (list): List of exogenous variable column names used for forecasting.
    """
    forecast = arima_model.forecast(steps=len(test), exog=test[exog_columns])
    rmse = np.sqrt(mean_squared_error(test['close'], forecast))
    mae = mean_absolute_error(test['close'], forecast)
    print('RMSE:', rmse)
    print('MAE:', mae)
    print_divider()

    plt.figure(figsize=(10, 6))
    plt.plot(test.index, test['close'], label='Actual Prices', color='blue')
    plt.plot(test.index, forecast, label='Forecasted Prices', color='orange')
    plt.title('Bitcoin Price Prediction using ARIMA')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.savefig(BTC_ACTUAL_VS_PREDICTED_CHART_PATH)
    print(f"Chart saved as {BTC_ACTUAL_VS_PREDICTED_CHART_PATH}")

def main():
    """Main function to handle all steps for BTCUSD price analysis."""
    print("Preparing model...")
    
    # Download and import df
    download_data()
    df = pd.read_csv(BTC_PRICEDATA_PATH, index_col=0)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.drop(columns=['symbol'], inplace=True)
    df = df.sort_index()
    
    # Merge sentiment data
    df = merge_sentiment_data(df)
    
    # Generate closing price graph
    generate_btcusd_closing_price_graph(df)

    # Feature engineering
    df = feature_engineering(df)

    # Define exogenous variables
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
    ]

    # Train-test split
    train, test = train_test_split(df)

    # Train ARIMA model
    arima_model = train_arima_model(train, exog_columns)

    # Evaluate model
    evaluate_model(arima_model, test, exog_columns)

# Execute script
if __name__ == "__main__":
    main()