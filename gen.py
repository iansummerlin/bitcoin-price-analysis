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
6. Evaluate, plot and download

Minor Improvements:
- Work on getting the values for AIC and BIC down whilst maintaining marginal errors
- Address heteroskedasticity with GARCH to better model volatility clustering
- Use LSTM or GRU models for long-sequence time series if computational resources permit

Major Improvements:
- You could combine this model with machine learning models like Random Forests or XGBoost, 
which might capture non-linear patterns better.
- Use other advanced models such as GARCH for volatility modeling or LSTM (Long Short-Term Memory) 
Networks for capturing long-term dependencies in time-series forecasting.
- For a more sophisticated model I should be using 10-20 features.

Misc Improvements:
- Download merged csv so it doesn't need to be generated on each run of the model

Current version: 1.2.0
Last updated: 29/11/2024 19:54
Author: Ian Summerlin
"""
import os
import time 
import numpy as np
import pandas as pd
import joblib

from itertools import product
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils import print_divider, download_file
from features.averages import calculate_moving_averages, calculate_average_true_range
from features.price import calculate_lagged_close
from features.technical import calculate_rsi
from features.volatility import calculate_rolling_volatility, calculate_ewma_volatility, calculate_parkinson_volatility

import warnings
warnings.filterwarnings("ignore")

# Constants
BTC_PRICEDATA_1H_URL = "https://www.cryptodatadownload.com/cdd/Gemini_BTCUSD_1h.csv"
BTC_PRICEDATA_PATH = "BTCUSD_1H.csv"
BTC_SENTIMENTDATA_URL = "https://api.alternative.me/fng/?limit=250000&format=csv&date_format=kr"
BTC_SENTIMENTDATA_PATH = "BTC_sentiment.csv"
BTC_CLOSING_PRICE_CHART_PATH = "bitcoin-closing-price.png"
BTC_ACTUAL_VS_PREDICTED_CHART_PATH = "bitcoin-actual-vs-predicted-price.png"
ARIMA_CLOSING_PRICE_MODEL = "arima-btc-closing-price.pkl"

def download_data():
    """
    Download both BTC price and sentiment data.

    Parameters:
    replace (bool): Whether to replace existing files.
    """
    print("Step 1/8: Downloading data...")
    print("  - Downloading BTC price data...")
    download_file(BTC_PRICEDATA_1H_URL, BTC_PRICEDATA_PATH)
    print("  - Downloading sentiment data...")
    download_file(BTC_SENTIMENTDATA_URL, BTC_SENTIMENTDATA_PATH)
    print("✓ Data download complete")

def merge_sentiment_data(df):
    """
    Merge sentiment data with BTCUSD price data.

    Parameters:
    df (DataFrame): The DataFrame containing BTCUSD price data.

    Returns:
    DataFrame: The merged DataFrame with sentiment data.
    """
    print("Step 2/8: Merging sentiment data...")
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
    print("✓ Sentiment data merged with BTCUSD price data")
    return df

def generate_btcusd_closing_price_graph(df):
    """
    Generate the BTCUSD closing price graph.

    Parameters:
    df (DataFrame): The DataFrame containing BTCUSD price data.
    """
    print("Step 3/8: Generating closing price graph...")
    print_divider()
    print("Generating Bitcoin Closing Price graph...")
    if os.path.exists(BTC_CLOSING_PRICE_CHART_PATH):
        print(f"The file '{BTC_CLOSING_PRICE_CHART_PATH}' already exists.")
        return
    
    plt.figure(figsize=(10, 6))
    df['close'].plot(title='Bitcoin Closing Price')
    plt.savefig(BTC_CLOSING_PRICE_CHART_PATH)
    print(f"✓ Bitcoin Closing Price graph generated and saved as {BTC_CLOSING_PRICE_CHART_PATH}")
    print_divider()

def feature_engineering(df):
    """
    Perform feature engineering on the DataFrame.

    Parameters:
    df (DataFrame): The DataFrame containing BTCUSD price data.

    Returns:
    DataFrame: The DataFrame with engineered features.
    """
    print("Step 4/8: Performing feature engineering...")
    print("  - Calculating lagged close prices...")

    # Price
    for lag in [90, 120]:
        calculate_lagged_close(df, lag)
        
    print("  - Calculating moving averages...")
    # Averages
    for ma in [7, 24]:
        calculate_moving_averages(df, ma)
    calculate_average_true_range(df)
    
    print("  - Calculating technical indicators...")
    # Technical
    calculate_rsi(df)
    
    print("  - Calculating volatility measures...")
    # Volatility
    calculate_rolling_volatility(df)
    calculate_ewma_volatility(df)
    calculate_parkinson_volatility(df)

    print("  - Cleaning data (removing NaN values)...")
    # Drop rows with missing values
    df = df.dropna()
    
    print("✓ Feature engineering complete")
    return df

def train_test_split(df):
    """
    Split the DataFrame into training and testing sets.

    Parameters:
    df (DataFrame): The DataFrame to split.

    Returns:
    tuple: A tuple containing the training DataFrame and the testing DataFrame.
    """
    print("Step 5/8: Splitting data into train/test sets...")
    train_size = int(len(df) * 0.8)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    print(f"✓ Data split complete - Training: {len(train)} rows, Testing: {len(test)} rows")
    return train, test

def grid_search_arima(df, p_values, d_values, q_values):
    """
    Perform GridSearch for the ARIMA model to find the best (p, d, q) combination.
    
    Parameters:
    df (DataFrame): The DataFrame containing BTCUSD price data.
    p_values (list): List of potential p values (AR order).
    d_values (list): List of potential d values (Differencing order).
    q_values (list): List of potential q values (MA order).
    
    Returns:
    tuple: Best (p, d, q) combination and the corresponding AIC score.
    """
    print("  - Performing grid search for optimal ARIMA parameters...")
    best_aic = np.inf
    best_order = None
    total_combinations = len(p_values) * len(d_values) * len(q_values)
    current_combination = 0
    
    # Grid search over the specified parameter combinations
    for p, d, q in product(p_values, d_values, q_values):
        current_combination += 1
        if current_combination % 10 == 0:  # Print progress every 10 combinations
            print(f"    Testing combination {current_combination}/{total_combinations}: ({p},{d},{q})")
        
        try:
            model = ARIMA(df['close'], order=(p, d, q))
            model_fit = model.fit()
            
            # Use AIC to determine the best model (you can also use BIC or RMSE)
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_order = (p, d, q)
        except Exception as e:
            print(f"    Error testing combination ({p},{d},{q}): {e}")
            continue
    
    print(f"  ✓ Grid search complete - Best order: {best_order}, AIC: {best_aic:.2f}")
    # Return the best order (p, d, q) and the best AIC score
    return best_order, best_aic

def train_arima_model(train, exog_columns):
    """
    Fit ARIMA model using training data.

    Parameters:
    train (DataFrame): The training DataFrame containing BTCUSD price data.
    exog_columns (list): List of exogenous variable column names to include in the model.

    Returns:
    ARIMA: The fitted ARIMA model.
    """
    print("Step 6/8: Training ARIMA model...")
    p_values = range(0, 6)
    d_values = range(0, 3)
    q_values = range(0, 6)

    start_time = time.time()  # Start timer
    best_order, best_aic = grid_search_arima(train, p_values, d_values, q_values)
    print("  - Fitting final ARIMA model with best parameters...")
    model = ARIMA(train['close'], order=best_order, exog=train[exog_columns]) 
    arima_model = model.fit()
    end_time = time.time()  # End timer
    
    print(f"✓ Model training complete - Time: {end_time - start_time:.2f} seconds")
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
    print("Step 7/8: Evaluating model performance...")
    print("  - Generating forecasts...")
    forecast = arima_model.forecast(steps=len(test), exog=test[exog_columns])
    rmse = np.sqrt(mean_squared_error(test['close'], forecast))
    mae = mean_absolute_error(test['close'], forecast)
    print('  ✓ Model evaluation complete:')
    print('    RMSE:', rmse)
    print('    MAE:', mae)
    print_divider()

    print("  - Generating prediction vs actual chart...")
    plt.figure(figsize=(10, 6))
    plt.plot(test.index, test['close'], label='Actual Prices', color='blue')
    plt.plot(test.index, forecast, label='Forecasted Prices', color='orange')
    plt.title('Bitcoin Price Prediction using ARIMA')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.savefig(BTC_ACTUAL_VS_PREDICTED_CHART_PATH)
    print(f"✓ Chart saved as {BTC_ACTUAL_VS_PREDICTED_CHART_PATH}")

def main():
    """Main function to handle all steps for BTCUSD price analysis."""
    print("=== Bitcoin Price Analysis Model Generation ===")
    print("Preparing model...")
    
    # Download and import df
    download_data()
    print("  - Loading price data...")
    df = pd.read_csv(BTC_PRICEDATA_PATH, index_col=0)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.drop(columns=['symbol'], inplace=True)
    df = df.sort_index()
    print("✓ Price data loaded and processed")
    
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

    # Download model
    print("Step 8/8: Saving trained model...")
    joblib.dump(arima_model, ARIMA_CLOSING_PRICE_MODEL)
    print(f"✓ Model saved as {ARIMA_CLOSING_PRICE_MODEL}")
    print("\n=== Model Generation Complete ===")

# Execute script
if __name__ == "__main__":
    main()