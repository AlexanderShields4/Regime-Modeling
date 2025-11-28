import streamlit as st
import json
import pandas as pd
from pathlib import Path
from datetime import datetime


# Import data loader
from pages_utils import fetch_all_data

# ========== Streamlit App ==========
st.set_page_config(page_title="Market Dashboard", layout="wide")

# Helper function to load Plotly HTML
def load_plotly_html(path, height=600):
    with open(path, "r") as f:
        html = f.read()
    st.components.v1.html(html, height=height, scrolling=False)

# ========== Sidebar Navigation ==========
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "HMM Regime Analysis", "Portfolio Backtest", "Price History", "Moving Averages"])

# ========== Home Page ==========
if page == "Home":
    st.title("📊 Market Data Dashboard")

    # ========== Data Fetching ==========
    data = fetch_all_data()

    # ========== Main Page Description ==========
    st.write("""
    Welcome to the Market Data Dashboard! This app provides insights into market data, including:
    - **HMM Regime Analysis**: View market regime predictions and portfolio allocations
    - **Price History**: Historical price data for stocks, indices, and natural resources
    - **Moving Averages**: Technical indicators and moving average analysis

    Use the sidebar to navigate to different pages for detailed analysis.
    """)

    st.info("💡 Tip: Run the HMM model using `python -m model` to generate regime analysis data, then view it in the 'HMM Regime Analysis' page!")

# ========== HMM Regime Analysis Page ==========
elif page == "HMM Regime Analysis":
    st.title("🔮 HMM Regime Analysis Dashboard")

    dashboard_dir = Path("dashboard_outputs")

    if not dashboard_dir.exists():
        st.warning("⚠️ No HMM analysis data found. Please run the model first:")
        st.code("python -m model", language="bash")
        st.stop()

    # Regime statistics
    stats_file = dashboard_dir / "regime_stats.json"
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            regime_stats = json.load(f)

        st.header("📊 Regime Statistics")
        cols = st.columns(6)
        labels = ['Total Periods', 'Extreme Bull Periods', 'Moderate Bull Periods', 
                  'Extreme Bear Periods', 'Moderate Bear Periods', 'Sideways Periods']
        for col, label in zip(cols, labels):
            value = regime_stats.get(label, 0)
            pct_label = f" ({regime_stats.get(label.replace('Periods','%'), '')})" if 'Periods' in label else ''
            col.metric(label, f"{value}{pct_label}")

        col1, col2 = st.columns(2)
        col1.metric("Regime Changes", regime_stats.get('Regime Changes', 0))
        col2.metric("Avg Duration", f"{regime_stats.get('Average Duration', 0):.1f} periods")

    # Model performance
    perf_file = dashboard_dir / "performance_metrics.json"
    if perf_file.exists():
        with open(perf_file, 'r') as f:
            perf_metrics = json.load(f)

        st.header("📈 Model Performance")
        cols = st.columns(6)
        perf_labels = ["Train Log-Likelihood", "Test Log-Likelihood", "Degradation", "Test AIC", "Test BIC", "Decision"]
        for col, label in zip(cols, perf_labels):
            value = perf_metrics.get(label, 0)
            if label == "Degradation":
                value = f"{value:.1f}%"
            elif label == "Decision":
                value = "✅" if "YES" in value else "⚠️" if "CAUTION" in value else "❌"
            else:
                value = f"{value:.0f}"
            col.metric(label, value)

    # Plotly visualizations
    plot_files = {
        "Regime Timeline": dashboard_dir / "regime_timeline.html",
        "Regime Distribution": dashboard_dir / "regime_distribution.html",
        "Portfolio Allocations": dashboard_dir / "portfolio_allocation_chart.html",
        "Feature Importance Heatmap": dashboard_dir / "feature_importance.html"
    }

    for title, path in plot_files.items():
        if path.exists():
            st.header(f"📊 {title}")
            load_plotly_html(path, height=450 if "Timeline" not in title else 600)

    # Portfolio allocation table
    alloc_csv = dashboard_dir / "portfolio_allocations.csv"
    if alloc_csv.exists():
        st.header("📋 Allocation Recommendations")
        allocations = pd.read_csv(alloc_csv)
        st.dataframe(allocations, use_container_width=True, hide_index=True)

    # Risk metrics
    risk_file = dashboard_dir / "risk_metrics.json"
    if risk_file.exists():
        with open(risk_file, 'r') as f:
            risk_metrics = json.load(f)
        st.header("⚠️ Risk Metrics by Regime")
        risk_df = pd.DataFrame(risk_metrics).T.reset_index().rename(columns={'index':'Regime'})
        st.dataframe(
            risk_df.style.format({
                'Average Return': '{:.6f}',
                'Volatility': '{:.6f}',
                'Sharpe Ratio': '{:.3f}',
                'Max Drawdown': '{:.3%}',
                'Periods': '{:.0f}'
            }),
            use_container_width=True,
            hide_index=True
        )

    # Download section
    st.header("💾 Download Data")
    col1, col2, col3 = st.columns(3)
    if alloc_csv.exists():
        with open(alloc_csv, 'rb') as f:
            col1.download_button("📥 Allocations CSV", f, "portfolio_allocations.csv", "text/csv")
    perf_csv = dashboard_dir / "performance_summary.csv"
    if perf_csv.exists():
        with open(perf_csv, 'rb') as f:
            col2.download_button("📥 Performance CSV", f, "performance_summary.csv", "text/csv")
    if risk_file.exists():
        with open(risk_file, 'rb') as f:
            col3.download_button("📥 Risk Metrics JSON", f, "risk_metrics.json", "application/json")

# ========== Portfolio Backtest Page ==========
elif page == "Portfolio Backtest":
    st.title("💰 Portfolio Backtest Analysis")

    backtest_dir = Path("dashboard_outputs/backtest_results")
    if not backtest_dir.exists():
        st.warning("⚠️ No backtest results found. Please run the model first:")
        st.code("python -m model", language="bash")
        st.stop()

    # Metadata
    metadata_file = backtest_dir / "backtest_metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        st.header("📊 Backtest Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Date Range", f"{metadata['date_range']['start'][:10]} to {metadata['date_range']['end'][:10]}")
        col2.metric("Best Strategy", metadata['best_strategy'])
        col3.metric("Transaction Cost", f"{metadata['transaction_cost']*100:.2f}%")

    # Metrics comparison
    metrics_file = backtest_dir / "data/metrics_summary.csv"
    if metrics_file.exists():
        st.header("📈 Performance Metrics Comparison")
        metrics_df = pd.read_csv(metrics_file, index_col=0)
        st.dataframe(metrics_df.style.format({
            'Total Return': '{:.2%}',
            'CAGR': '{:.2%}',
            'Volatility': '{:.2%}',
            'Sharpe Ratio': '{:.3f}',
            'Sortino Ratio': '{:.3f}',
            'Max Drawdown': '{:.2%}',
            'Average Drawdown': '{:.2%}',
            'Recovery Time': '{:.1f}',
            'Win Rate': '{:.2%}',
            'Calmar Ratio': '{:.3f}'
        }), use_container_width=True)

    # Backtest charts (Plotly)
    chart_files = {
        "Portfolio Value Over Time": backtest_dir / "charts/portfolio_value_comparison.html",
        "Cumulative Returns": backtest_dir / "charts/cumulative_returns.html",
        "Drawdown Analysis": backtest_dir / "charts/drawdown_analysis.html",
        "Sharpe Ratio Comparison": backtest_dir / "charts/sharpe_ratio_comparison.html",
        "Annual Returns by Strategy": backtest_dir / "charts/annual_returns_by_strategy.html",
        "Regime Timeline with Portfolio": backtest_dir / "charts/regime_timeline_portfolio.html"
    }

    for title, path in chart_files.items():
        if path.exists():
            st.header(f"📉 {title}")
            load_plotly_html(path, height=450 if "Timeline" not in title else 600)

    # Regime Allocation History
    alloc_history_file = backtest_dir / "data/regime_allocations.csv"
    if alloc_history_file.exists():
        st.header("📋 Regime Allocation History")
        alloc_history = pd.read_csv(alloc_history_file)
        st.write("**Regime Distribution:**")
        regime_counts = alloc_history['Regime_Name'].value_counts()
        for regime, count in regime_counts.items():
            pct = count / len(alloc_history) * 100
            st.write(f"- {regime}: {count} periods ({pct:.1f}%)")
        with st.expander("View Allocation History Details"):
            st.dataframe(alloc_history.head(50), use_container_width=True, hide_index=True)

    # Download section
    col1, col2, col3 = st.columns(3)
    if metrics_file.exists():
        with open(metrics_file, 'rb') as f:
            col1.download_button("📥 Metrics CSV", f, "backtest_metrics.csv", "text/csv")
    results_file = backtest_dir / "data/strategy_results.csv"
    if results_file.exists():
        with open(results_file, 'rb') as f:
            col2.download_button("📥 Portfolio Values", f, "portfolio_values.csv", "text/csv")
    if metadata_file.exists():
        with open(metadata_file, 'rb') as f:
            col3.download_button("📥 Metadata JSON", f, "backtest_metadata.json", "application/json")

# ========== Price History Page ==========
elif page == "Price History":
    st.title("📈 Price History")
    st.info("Navigate to pages/price_history.py for price history analysis")

# ========== Moving Averages Page ==========
elif page == "Moving Averages":
    st.title("📊 Moving Averages")
    st.info("Navigate to pages/moving_averages.py for moving average analysis")
