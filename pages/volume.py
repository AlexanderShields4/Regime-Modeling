import streamlit as st
from pages_utils import fetch_all_data
from data.data_utils import get_volume_data

data = fetch_all_data()

# ========== Volume Page ==========
st.title("📊 Volume Data")

# Sidebar controls
st.sidebar.header("Volume Settings")
st.sidebar.write("Select options for volume visualization.")

view_option = st.sidebar.selectbox("Choose dataset:", ["Stocks", "Indices", "Resources"])
available_cols = list(data[view_option.lower()].columns)

selected_tickers = st.sidebar.multiselect(
    "Select tickers:",
    options=available_cols,
    default=available_cols[:5]
)

# Allow the user to choose the period for volume data (defaults to shorter than 1y)
period = st.sidebar.selectbox("Select period:", ["1mo", "3mo", "6mo", "1y"], index=1)

if not selected_tickers:
    st.warning("Please select at least one ticker to display volume.")
else:
    st.subheader(f"{view_option} Volume (period={period})")

    # Fetch volume data for the chosen tickers and period
    try:
        volume_df = get_volume_data(tickers=selected_tickers, period=period, interval="1d")
    except Exception as e:
        st.error(f"Failed to fetch volume data: {e}")
        volume_df = None

    if volume_df is None or volume_df.empty:
        st.info("No volume data available for the selected tickers/period.")
    else:
        st.line_chart(volume_df)
        st.subheader("Volume Data (last 10 rows)")
        st.dataframe(volume_df.tail(10))