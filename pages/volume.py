import streamlit as st
from .pages_utils import fetch_all_data

data = fetch_all_data()

# ========== Volume Page ==========
st.title("📊 Volume Data")

# Sidebar controls
st.sidebar.header("Volume Settings")
st.sidebar.write("Select options for volume visualization.")

view_option = st.sidebar.selectbox("Choose dataset:", ["Stocks", "Indices", "Resources"])
selected_tickers = st.sidebar.multiselect(
    "Select tickers:",
    options=list(data[view_option.lower()].columns),
    default=list(data[view_option.lower()].columns)[:5]
)

# Main content
st.subheader(f"{view_option} Volume")
# Placeholder for volume data visualization (requires volume data in `data` dictionary)
st.write("Volume data visualization is not implemented yet.")