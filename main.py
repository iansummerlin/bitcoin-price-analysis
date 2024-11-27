"""
Time series forecasting for BTC/USD. Sourced 1h timeseries df from Gemini.

Steps
1. Download and prepare data
2. Select target variable (close)
3. Feature engineering (lagged close, moving averages and volatility)
4. Train-test split
5. Model training
6. Evaluate and plot

Minor Improvements
- More sophisticated volatility measurements
- More laggged close variables (beyond 90 or 120) to capture longer-term dependencies
- Additional exogenous variables such as market sentiment, new data or technical indicators
(e.g. RSI or MACD) 
- RMSE (3966) and MAE (3044), these are still pretty high
- Tune ARIMA order: consider experimenting with different values of p, d and q to see if more 
apropriate model can be fitted. You can use GridSearch or a similar approach to find the 
best combination of these parameters

Major Improvements
- You could combine this model with machine learning models like Random Forests or XGBoost, 
which might capture non-inear patterns better.
- Use other advanced models such as GARCH for volatility modeling or LSTM (Long Short-Term Memory) 
Networks for capturing long-term dependencies in time-series forecasting

Current version: 1.0.7
Last updated: 27/11/2024 00:26
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
for lag in range(1, 121):
    df[f"close_lag_{lag}"] = df['close'].shift(lag)
    
# Moving averages (7 hour and 24 hour moving average)
df['MA_7'] = df['close'].rolling(window=7).mean()
df['MA_24'] = df['close'].rolling(window=24).mean()

# Volatility measured using rolling standard deviation
df['volatility_24'] = df['close'].rolling(window=24).std()

# Drop rows with missing values due to lagging
df = df.dropna()

"""
Train-test split
"""
train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
test = df.iloc[train_size:]
exog_columns = ['close_lag_90', 'close_lag_120', 'MA_7', 'MA_24', 'volatility_24']

X_train = train[exog_columns]
y_train = df['close'][:train_size]
X_test = test[exog_columns]
y_test = df['close'][train_size:]

"""
Model training
"""

# Fit ARIMA (adjust p, d, q values) using exogenous variables
p = 5
d = 1
q = 0
model = ARIMA(train['close'], order=(p, d, q), exog=X_train) 
arima_model = model.fit()
print(arima_model.summary())

# Forecast for the test period
exog_test = X_test[exog_columns]
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

