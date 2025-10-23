import streamlit as st
from pages_utils import fetch_all_data

data = fetch_all_data()

# ========== Price History Page ==========
st.title("📈 Price History")

# Sidebar controls
st.sidebar.header("Price History Settings")
st.sidebar.write("Select options for price history visualization.")

view_option = st.sidebar.selectbox("Choose dataset:", ["Stocks", "Indices", "Resources"])
selected_tickers = st.sidebar.multiselect(
    "Select tickers:",
    options=list(data[view_option.lower()].columns),
    default=list(data[view_option.lower()].columns)[:5]
)

# Main content
st.subheader(f"{view_option} Prices")
df_to_display = data[view_option.lower()][selected_tickers]
st.line_chart(df_to_display)

st.subheader(f"{view_option} Raw Data")
st.dataframe(df_to_display.tail(10))