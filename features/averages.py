def calculate_moving_averages(df, window_length):
    """
    Calculate moving averages for the closing prices.

    Parameters:
    df (DataFrame): The DataFrame containing BTCUSD price data.
    window_length (int): The number of periods to use for the moving averages calculation.

    Returns:
    None: The function modifies the DataFrame in place.
    """
    # Calculate the 7-period moving average
    df[f'MA_{window_length}'] = df['close'].rolling(window=window_length).mean()

def calculate_average_true_range(df, window_length=24):
    """
    Calculate the Average True Range (ATR) for the given DataFrame.

    Parameters:
    df (DataFrame): The DataFrame containing BTCUSD price data.
    window_length (int): The number of periods to use for the atr calculation.

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
    
    # Calculate the period rolling mean of the true range to get ATR for the given window_length
    df[f'atr_{window_length}'] = df['true_range'].rolling(window=window_length).mean()