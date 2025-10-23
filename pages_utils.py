import streamlit as st

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
    """Wrapper that delegates to data.data_utils.calculate_moving_average_mse.

    This keeps the default window consistent across pages.
    """
    from data.data_utils import calculate_moving_average_mse as _mse
    return _mse(df, window=window)