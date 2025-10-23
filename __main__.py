import streamlit as st

# Import data loaders
from data.ind_stocks import load_individual_stocks
from data.indices import load_indices
from data.natural_resources import load_resources
from pages_utils import fetch_all_data

# ========== Streamlit App ==========
st.set_page_config(page_title="Market Dashboard", layout="wide")
st.title("📊 Market Data Dashboard")

# ========== Data Fetching ==========
data = fetch_all_data()

# ========== Main Page Description ==========
st.write("""
Welcome to the Market Data Dashboard! This app provides insights into market data, including:
- Price history of stocks, indices, and natural resources.
- Volume trends.
- Moving averages and other technical indicators.

Use the sidebar to navigate to different pages for detailed analysis.
""")

