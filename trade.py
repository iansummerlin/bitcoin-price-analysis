import os
import joblib
import pandas as pd
import time
from datetime import datetime

MODEL_PATH = "./arima-btc-closing-price.pkl"

def load_model():
    """
    Load the generated model from `gen.py` script

    Returns:
    Model: The generated model from `gen.py`
    """
    if not os.path.exists(MODEL_PATH):
        print(f"The file '{MODEL_PATH}' does not exist")
        return
    return joblib.load(MODEL_PATH)

def predict_next_price(model, exog_data):
    """
    Use the ARIMA model to predict the next closing price.

    Parameters:
    model (ARIMA): The fitted ARIMA model.
    exog_data (DataFrame): The exogenous variables for the prediction.

    Returns:
    float: The predicted next closing price.
    """
    # Forecast the next time step
    forecast = model.get_forecast(steps=1, exog=exog_data)
    print(forecast)
    
    # Extract the predicted mean (point forecast)
    predicted_price = forecast.predicted_mean.iloc[0]
    return predicted_price

def execute_buy(amount_usd):
    """
    Placeholder for executing a buy order.
    In a real scenario, this would interact with an exchange API.
    """
    print(f"Executing BUY order for {amount_usd} USD worth of BTC.")
    # Add your exchange API integration here
    pass

def execute_sell(amount_btc):
    """
    Placeholder for executing a sell order.
    In a real scenario, this would interact with an exchange API.
    """
    print(f"Executing SELL order for {amount_btc} BTC.")
    # Add your exchange API integration here
    pass

def main():
    model = load_model()
    if model is None:
        print("Could not load the model. Exiting.")
        return

    filename = "BTCUSD_trading.csv"
    trade_log_filename = "BTCUSD_predictions_trades.csv"

    # Initialize the trade log CSV with headers if it does not exist
    if not os.path.exists(trade_log_filename):
        with open(trade_log_filename, 'w') as f:
            f.write("date,current_close_price,predicted_closing_price,trade_action,buy_threshold_used,sell_threshold_used,amount_traded,trade_currency\n")

    while True:
        if not os.path.exists(filename):
            print(f"Waiting for {filename} to be created by collect.py...")
            time.sleep(60) # Wait for a minute before checking again
            continue

        try:
            df = pd.read_csv(filename)
            if df.empty:
                print(f"Waiting for data in {filename}...")
                time.sleep(60)
                continue

            # Get the latest row of data
            latest_data = df.iloc[[-1]].copy()

            # Select only the columns that were used for training the model
            required_columns = ['close_lag_90', 'close_lag_120', 'MA_7', 'MA_24', 'rsi', 
                                'volatility_24', 'volatility_ewma_24', 'parkinson_volatility', 
                                'atr_24', 'fng_value']  # Example of required columns
            
            # Ensure all columns are numeric and handle potential NaNs for prediction
            df_for_prediction = latest_data[required_columns].apply(pd.to_numeric, errors='coerce').dropna(axis=1)

            if df_for_prediction.empty:
                print("Not enough data or invalid data for prediction. Waiting for more data...")
                time.sleep(60)
                continue

            predicted_price = predict_next_price(model, df_for_prediction)
            current_close_price = latest_data['close'].iloc[0]
            current_date = latest_data['date'].iloc[0]

            print(f"Current Close Price: {current_close_price}")
            print(f"Predicted Next Close Price: {predicted_price}")

            trade_action = "HOLD"
            amount_traded = "N/A"
            trade_currency = "N/A"

            # Simple trading strategy:
            # If predicted price is significantly higher, consider buying
            # If predicted price is significantly lower, consider selling
            
            # You'll need to define your own thresholds and risk management here
            BUY_THRESHOLD = 1.001 # e.g., 0.1% increase
            SELL_THRESHOLD = 0.999 # e.g., 0.1% decrease

            if predicted_price > current_close_price * BUY_THRESHOLD:
                print("Prediction: Price will increase. Considering BUY.")
                trade_action = "BUY"
                amount_traded = 100 # Example: Buy $100 worth of BTC
                trade_currency = "USD"
                # execute_buy(amount_usd=100) 
            elif predicted_price < current_close_price * SELL_THRESHOLD:
                print("Prediction: Price will decrease. Considering SELL.")
                trade_action = "SELL"
                amount_traded = 0.001 # Example: Sell 0.001 BTC
                trade_currency = "BTC"
                # execute_sell(amount_btc=0.001) 
            else:
                print("Prediction: Price stable. No trade.")

            # Log prediction and trade action to the new CSV
            with open(trade_log_filename, 'a') as f:
                f.write(f"{current_date},{current_close_price},{predicted_price},{trade_action},{BUY_THRESHOLD},{SELL_THRESHOLD},{amount_traded},{trade_currency}\n")
            print(f"Logged prediction and trade action to {trade_log_filename}")

        except Exception as e:
            print(f"An error occurred: {e}")

        time.sleep(3600) # Check every hour (3600 seconds)

if __name__ == "__main__":
    main()
