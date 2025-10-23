import streamlit as st
import pandas as pd

# Regime-Modeling/__main__.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import data loaders
from data.ind_stocks import load_individual_stocks
from data.indices import load_indices
from data.natural_resources import load_resources



# ========== Streamlit App ==========
st.set_page_config(page_title="Market Dashboard", layout="wide")
st.title("📊 Market Data Dashboard")

# ========== Data Fetching ==========
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_all_data():
    """Fetch all data sources and return a dictionary of DataFrames."""
    stocks_df = load_individual_stocks(period="1y", interval="1d")
    indices_df = load_indices(period="1y", interval="1d")
    resources_df = load_resources(period="1y", interval="1d")
    
    return {
        "stocks": stocks_df,
        "indices": indices_df,
        "resources": resources_df
    }

data = fetch_all_data()

# ========== Sidebar Controls ==========
st.sidebar.header("Select Data View")
view_option = st.sidebar.selectbox("Choose dataset:", ["Stocks", "Indices", "Resources"])

# Optional: let user pick tickers dynamically
selected_tickers = st.sidebar.multiselect(
    "Select tickers:",
    options=list(data[view_option.lower()].columns),
    default=list(data[view_option.lower()].columns)[:5]
)

# ========== Main Dashboard ==========
st.subheader(f"{view_option} Prices")
df_to_display = data[view_option.lower()][selected_tickers]
st.line_chart(df_to_display)

st.subheader(f"{view_option} Raw Data")
st.dataframe(df_to_display.tail(10))

