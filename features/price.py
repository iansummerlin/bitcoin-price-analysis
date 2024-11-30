def calculate_lagged_close(df, lag):
    """
    Calculate lagged close on the DataFrame.

    Parameters:
    df (DataFrame): The DataFrame containing the BTCUSD price data.
    lag (int): The number of periods to shift the closing price.

    Returns:
    None: The function modifies the DataFrame in place.
    """
    df[f"close_lag_{lag}"] = df['close'].shift(lag)
