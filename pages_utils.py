import streamlit as st
import pandas as pd
import numpy as np

# ========== Data Fetching ==========
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_all_data():
    """Fetch all data sources and return a dictionary of DataFrames."""
    from data.ind_stocks import load_individual_stocks
    from data.indices import load_indices
    from data.natural_resources import load_resources

    stocks_df = load_individual_stocks(period="1y", interval="1d")
    indices_df = load_indices(period="1y", interval="1d")
    resources_df = load_resources(period="1y", interval="1d")
    
    return {
        "stocks": stocks_df,
        "indices": indices_df,
        "resources": resources_df
    }

# ========== Moving Averages Calculation ==========
def calculate_moving_averages(df, window=20):
    """Calculate moving averages for a given DataFrame."""
    return df.rolling(window=window).mean()


def calculate_moving_average_mse(df, window=20):
    """Calculate a rolling Mean Squared Error (MSE) for a simple moving-average forecast.

    Implementation details:
    - The moving-average forecast for time t uses the mean of the previous `window` values
      (i.e. rolling mean shifted by 1). This ensures the prediction doesn't include the
      current observation.
    - Squared errors are computed as (actual - forecast)**2.
    - The returned value is the rolling mean of the squared errors over the same `window`.

    Args:
        df (pd.DataFrame): time-indexed DataFrame with one or more series (columns).
        window (int): window length for the moving average and for the rolling MSE.

    Returns:
        pd.DataFrame: DataFrame of the same shape as `df` containing the rolling MSE for
        each column. Leading values will be NaN until enough data for a full window exists.
    """
    if df is None:
        return pd.DataFrame()

    # Work on a numeric copy of the data
    df_numeric = df.copy()
    df_numeric = df_numeric.apply(pd.to_numeric, errors="coerce")

    # Forecast at time t is the mean of the previous `window` observations
    forecast = df_numeric.rolling(window=window, min_periods=window).mean().shift(1)

    # Squared error
    se = (df_numeric - forecast) ** 2

    # Rolling mean of squared error (MSE) over the same window
    mse = se.rolling(window=window, min_periods=window).mean()

    return mse
