
# Project Documentation: Bitcoin Price Analysis and Trading Bot

## Overview

This project is a Python-based framework for analyzing Bitcoin prices and making real-time predictions. It consists of three main components: a model generation script, a data collection script, and a trading bot script.

### 1. Model Generation (`gen.py`)

This script is responsible for training a time-series forecasting model to predict the closing price of Bitcoin.

**Functionality:**

*   **Data Acquisition**: Downloads 1-hour BTC/USD price data and daily fear and greed index data.
*   **Feature Engineering**: Creates a rich set of features for the model, including:
    *   **Price-based**: Lagged closing prices.
    *   **Averages**: Simple Moving Averages (SMA) and Average True Range (ATR).
    *   **Technical Indicators**: Relative Strength Index (RSI).
    *   **Volatility Measures**: Rolling, EWMA, and Parkinson volatility.
*   **Model Training**: 
    *   Splits the data into training and testing sets.
    *   Performs a grid search to find the optimal parameters (p, d, q) for an ARIMA (Autoregressive Integrated Moving Average) model.
    *   Trains the ARIMA model on the historical data, using the engineered features as exogenous variables (additional inputs).
*   **Evaluation and Saving**:
    *   Evaluates the model's performance using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).
    *   Saves the trained model to a file named `arima-btc-closing-price.pkl` for later use.
    *   Generates plots to visualize the model's predictions against actual prices.

### 2. Data Collection (`collect.py`)

This script is responsible for collecting real-time Bitcoin price data and calculating features.

**Functionality:**

*   **Real-time Data Stream**: Connects to the Binance WebSocket stream to receive live `btcusdt@ticker` data.
*   **Feature Calculation**: On an hourly basis, it processes the incoming real-time data and calculates the same set of features that are used for model training.
*   **Data Logging**: Appends the incoming data and calculated features to a CSV file (`BTCUSD_trading.csv`).
*   **Error Handling**: Includes basic error logging to a text file (`BTCUSD_trading_errors.txt`).

### 3. Trading Bot (`trade.py`)

This script uses the trained model to make predictions on the collected data and execute trading decisions.

**Functionality:**

*   **Model Loading**: Loads the pre-trained ARIMA model from the `.pkl` file.
*   **Data Reading**: Reads the latest data and features from `BTCUSD_trading.csv`.
*   **Prediction**: Uses the trained model and the newly calculated features to predict the next hour's closing price.
*   **Trading Logic**: Implements a simple trading strategy based on the predicted price (e.g., buy if predicted price is significantly higher, sell if significantly lower).
*   **Trade Logging**: Logs the current price, predicted price, and the executed trade action (BUY, SELL, HOLD) to a separate CSV file (`BTCUSD_predictions_trades.csv`).

## How It Works: A Step-by-Step Flow

1.  **Run `gen.py`**:
    *   This script is executed first to download historical data, perform feature engineering, and train the ARIMA model.
    *   The output is a trained model file (`arima-btc-closing-price.pkl`) and some analytical charts.

2.  **Run `collect.py`**:
    *   This script is executed to start the data collection process.
    *   It connects to Binance and starts listening for live data.
    *   As new data arrives, it continuously calculates features and logs the results to `BTCUSD_trading.csv`.

3.  **Run `trade.py`**:
    *   This script is executed to start the trading bot.
    *   It loads the model created by `gen.py`.
    *   It continuously reads the latest data from `BTCUSD_trading.csv`.
    *   It makes predictions and logs trading decisions to `BTCUSD_predictions_trades.csv`.

## Project Structure

```
/home/ixn/Documents/code/crypto/bitcoin-price-analysis/
├───.gitignore
├───CHANGELOG.txt
├───gen.py                  # Main script for model generation and training
├───collect.py                 # Script for real-time data collection and feature calculation
├───trade.py                # Script for trading bot logic, prediction, and trade execution
├───README.md
├───requirements.txt
├───utils.py                # Utility functions (e.g., file download)
├───features/
│   ├───averages.py         # Functions for calculating moving averages and ATR
│   ├───price.py            # Functions for calculating price-based features
│   ├───technical.py        # Functions for calculating technical indicators (RSI)
│   └───volatility.py       # Functions for calculating volatility measures
└───__pycache__/
```

## Potential Areas for Improvement

The project itself, in its comments, suggests several areas for future development:

*   **Advanced Models**: Exploring more complex models like Gradient Boosting (XGBoost), LSTMs, or GRUs to capture non-linear patterns.
*   **More Features**: Expanding the feature set to potentially improve model accuracy.
*   **Automated Data Formatting**: The `README.md` mentions a manual step for formatting the downloaded data, which could be automated.
*   **Robustness**: Enhancing the error handling and overall robustness of the trading bot.
