def calculate_rsi(df, window_length=14):
    """
    Calculate the Relative Strength Index (RSI).

    Parameters:
    df (DataFrame): The DataFrame containing BTCUSD price data.
    window_length (int): The number of periods to use for the RSI calculation.

    Returns:
    None: The function modifies the DataFrame in place.
    """
    # Calculate the difference in closing prices
    delta = df['close'].diff()

    # Calculate gains and losses
    gain = (delta.where(delta > 0, 0)).rolling(window=window_length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window_length).mean()

    # Calculate the Relative Strength (RS)
    # Guard against division by zero: when loss is 0, RSI is 100
    rs = gain / loss.replace(0, float('nan'))

    # Calculate the RSI — where loss was 0 (rs is NaN), RSI should be 100
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(100)
