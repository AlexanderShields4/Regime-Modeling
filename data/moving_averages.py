def calculate_moving_averages(df, window=20):
    return df.rolling(window=window).mean()