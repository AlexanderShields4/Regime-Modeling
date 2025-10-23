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