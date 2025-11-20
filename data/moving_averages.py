def calculate_moving_averages(df, window=20):
    """Calculate moving averages for a given DataFrame."""
    return df.rolling(window=window).mean()