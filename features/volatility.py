import numpy as np

def calculate_rolling_volatility(df, window_length=24):
    """
    Calculate the rolling standard deviation (volatility) for the closing prices.

    Parameters:
    df (DataFrame): The DataFrame containing BTCUSD price data.
    window_length (int): The rolling window size. Default is 24.

    Returns:
    None: The function modifies the DataFrame in place by adding the 'volatility_24' column.
    """
    df[f'volatility_{window_length}'] = df['close'].rolling(window=window_length).std()

def calculate_ewma_volatility(df, span=24):
    """
    Calculate the exponentially weighted moving average (EWMA) volatility for the closing prices.

    Parameters:
    df (DataFrame): The DataFrame containing BTCUSD price data.
    span (int): The span for the EWMA calculation. Default is 24.

    Returns:
    None: The function modifies the DataFrame in place by adding the 'volatility_ewma_24' column.
    """
    df[f'volatility_ewma_{span}'] = df['close'].ewm(span=span).std()

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