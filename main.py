"""
Trading bot
"""

import os
import requests
import websocket
import json
import joblib
import pandas as pd
import time
import rel
from datetime import datetime

from features.averages import calculate_moving_averages, calculate_average_true_range
from features.price import calculate_lagged_close
from features.technical import calculate_rsi
from features.volatility import calculate_rolling_volatility, calculate_ewma_volatility, calculate_parkinson_volatility

MODEL_PATH = "./arima-btc-closing-price.pkl"
BTC_SENTIMENTDATA_URL = "https://api.alternative.me/fng/?limit=1&date_format=kr"
BTCUSD_STREAM_URL = "wss://fstream.binance.com/ws/btcusdt"

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

def get_daily_sentiment_data():
    """
    Get daily sentiment data

    Returns:
    Tuple: value, current_timestamp and update_timestamp
    """
    response = requests.get(BTC_SENTIMENTDATA_URL)
    if response.status_code != 200:
        print(f"Failed to retrieve sentiment data. Status code: {response.status_code}")
        return None, None, None
    
    data = response.json()
    value = data['data'][0]['value']
    
    # Use time library for timestamps
    current_timestamp = time.time()  # Current time in seconds since the epoch
    time_until_update = int(data['data'][0]['time_until_update'])
    update_timestamp = current_timestamp + time_until_update  # Update timestamp in seconds

    # Convert timestamps to datetime objects
    current_datetime = datetime.fromtimestamp(current_timestamp)
    update_datetime = datetime.fromtimestamp(update_timestamp)

    return value, current_datetime, update_datetime

def handle_is_new_hour(timestamp_ms):
    """
    Determines if the given timestamp (in milliseconds) is at the start of a new hour.
    
    Args:
    timestamp_ms (int): Timestamp in milliseconds.
        
    Returns:
    bool: True if the timestamp is at the start of a new hour, False otherwise.
    """
    # Convert milliseconds to seconds and then to a datetime object
    dt = datetime.fromtimestamp(timestamp_ms / 1000)
    
    # Check if it's the start of a new hour (minute and second are 0)
    return dt.minute == 0 and dt.second == 0

def handle_is_enough_data(df, minimum_rows=120):
    """
    Is there enough data to calculate the lagging features?

    Args:
    df (DataFrame): The DataFrame

    Returns:
    bool: True if enough data, false if not
    """
    return len(df) >= minimum_rows

def prepare_exog_data(model, df, stream_data):
    """
    Prepare exogenous variables for the next time step

    Args:
    model (ARIMA): The ARIMA BTCUSD closing price prediction model
    df (DataFrame): The DataFrame loaded from the CSV file
    stream_data (StreamData): Binance stream data

    Returns:
    DataFrame: The updated DataFrame with new data.
    """
    fng_value, _, _ = get_daily_sentiment_data()  # TODO: only fetch when required
    new_row = pd.DataFrame({
        'date': [datetime.fromtimestamp(int(stream_data['E'])/1000)],
        'open': [stream_data['o']],
        'high': [stream_data['h']],
        'low': [stream_data['l']],
        'close': [stream_data['c']],
        'Volume BTC': [stream_data['v']],
        'Volume USD': [stream_data['q']],
        'fng_value': [fng_value],
    })

    df = pd.concat([df, new_row], ignore_index=True)

    # Is there enough data to calculate features?
    is_enough_data = handle_is_enough_data(df)
    if is_enough_data:
        print("Calculating features...")
        # Price
        for lag in [90, 120]:
            calculate_lagged_close(df, lag)
            
        # Averages
        for ma in [7, 24]:
            calculate_moving_averages(df, ma)
        calculate_average_true_range(df)
        
        # Technical
        calculate_rsi(df)
        
        # Volatility
        calculate_rolling_volatility(df)
        calculate_ewma_volatility(df)
        calculate_parkinson_volatility(df)

        # Prepare the last row for prediction
        df_last_row = df.iloc[[-1]].copy()  # Create a copy of the last row
        
        # Select only the columns that were used for training the model
        required_columns = ['close_lag_90', 'close_lag_120', 'MA_7', 'MA_24', 'rsi', 
                            'volatility_24', 'volatility_ewma_24', 'parkinson_volatility', 
                            'atr_24', 'fng_value']  # Example of required columns
        
        # Remove columns with NaNs for prediction only
        df_for_prediction = df_last_row[required_columns].dropna(axis=1) 
         # Ensure all columns are numeric
        df_for_prediction = df_for_prediction.apply(pd.to_numeric, errors='coerce')

        predicted_price = predict_next_price(model, df_for_prediction)
        df.loc[df.index[-1], 'predicted_closing_price'] = predicted_price
    return df

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

def on_message_with_model(model, filename, df):
    """
    Creates a WebSocket `on_message` callback to handle incoming messages, 
    process them with a given model, and log results to a CSV file.

    Example stream data
    stream_data = {
        "e": "24hrTicker",  # Event type
        "E": 123456789,     # Event time
        "s": "BTCUSDT",     # Symbol
        "p": "0.0015",      # Price change
        "P": "250.00",      # Price change percent
        "w": "0.0018",      # Weighted average price
        "c": "0.0025",      # Last price (Close price)
        "Q": "10",          # Last quantity
        "o": "0.0010",      # Open price
        "h": "0.0025",      # High price
        "l": "0.0010",      # Low price
        "v": "10000",       # Total traded base asset volume
        "q": "18",          # Total traded quote asset volume
        "O": 0,             # Statistics open time
        "C": 86400000,      # Statistics close time
        "F": 0,             # First trade ID
        "L": 18150,         # Last trade ID
        "n": 18151          # Total number of trades
    }

    Args:
    model (ARIMA): The predictive model used to analyze the data.
    filename (str): The path to the CSV file where processed data will be appended.
    df (DataFrame): The initial DataFrame used for data processing.

    Returns:
    function: A callback function to handle WebSocket messages.
    """
    last_processed_hour = None  # Store the last processed hour

    def on_message(ws, message):
        nonlocal last_processed_hour  # Allow access to the outer variable
        data = json.loads(message)
        current_timestamp = int(data['E'])  # Current message timestamp in milliseconds
        # Convert current timestamp to datetime object
        current_hour = datetime.fromtimestamp(current_timestamp / 1000).replace(minute=0, second=0, microsecond=0)

        # Check if the current hour is different from the last processed hour
        if last_processed_hour is not None and current_hour <= last_processed_hour:
            return  # Skip processing if the current hour has not changed
        
        print("Preparing data for insert...")

        # Update the last processed hour to the current hour
        last_processed_hour = current_hour

        # Prepare exogenous data and get the updated DataFrame
        ndf = prepare_exog_data(model, df, data)
        
        # Overwrite the CSV with the updated DataFrame
        ndf.to_csv(filename, index=False) 
        print(f"Data appended to {filename}")
    return on_message


def on_error_with_timestamp(filename):
    """
    Creates an `on_error` function that logs errors with timestamps to a .txt file.
    
    Args:
    filename (str): The filename (without extension) for the log file.
        
    Returns:
    function: A callback function to handle errors.
    """
    def on_error(ws, error):
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format the error message
        log_message = f"{timestamp}: {error}\n"
        
        # Append the log message to the .txt file
        with open(f"{filename}.txt", "a") as log_file:
            log_file.write(log_message)
        
        print(f"Error logged: {log_message.strip()}")
        
    return on_error

def on_close(ws, close_status_code, close_msg):
    print(f"WebSocket connection closed: {close_status_code} - {close_msg}")

def on_open(ws):
    print("WebSocket connection opened")
    subscribe_message = {
        "method": "SUBSCRIBE",
        "params": ["btcusdt@ticker"],
        "id": 1
    }
    ws.send(json.dumps(subscribe_message))

def on_ping(ws, message):
    ws.send(message, websocket.ABNF.OPCODE_PONG)

def main():
    model = load_model()
    filename = "BTCUSD_trading.csv"  
    error_log_filename = "BTCUSD_trading_errors.txt"

    # Initialize the CSV with headers only if the file does not exist
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write("date,open,high,low,close,Volume BTC,Volume USD,close_lag_90,close_lag_120,MA_7,MA_24,rsi,volatility_24,volatility_ewma_24,parkinson_volatility,atr_24,fng_value,predicted_closing_price\n")

    # Initialize the error log with headers only if the file does not exist
    if not os.path.exists(error_log_filename):
        with open(error_log_filename, 'w') as f:
            f.write("Timestamp: Error Message\n")  # Optional header for clarity

    # Load existing data from the CSV file into the DataFrame if it exists
    if os.path.exists(filename):
        df = pd.read_csv(filename)  # Load existing data
    else:
        df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "Volume BTC", "Volume USD", 
                                   "close_lag_90", "close_lag_120", "MA_7", "MA_24", "rsi", 
                                   "volatility_24", "volatility_ewma_24", "parkinson_volatility", 
                                   "atr_24", "fng_value", "predicted_closing_price"])

    ws = websocket.WebSocketApp(BTCUSD_STREAM_URL,
                            on_message=on_message_with_model(model, filename, df),
                            on_error=on_error_with_timestamp(error_log_filename),  # Use the error log filename
                            on_close=on_close,
                            on_open=on_open,
                            on_ping=on_ping)
    ws.run_forever(dispatcher=rel, reconnect=5)
    rel.signal(2, rel.abort)
    rel.dispatch()

if __name__ == "__main__":
    main()  
