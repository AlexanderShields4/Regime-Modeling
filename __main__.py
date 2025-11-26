import streamlit as st
import json
import pandas as pd
from pathlib import Path

# Import data loader
from pages_utils import fetch_all_data

# ========== Streamlit App ==========
st.set_page_config(page_title="Market Dashboard", layout="wide")

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

    # Check if dashboard outputs exist
    if not dashboard_dir.exists():
        st.warning("⚠️ No HMM analysis data found. Please run the model first:")
        st.code("python -m model", language="bash")
        st.stop()

    # Load and display regime statistics
    stats_file = dashboard_dir / "regime_stats.json"
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            regime_stats = json.load(f)

        st.header("📊 Regime Statistics")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Periods", regime_stats['Total Periods'])
        with col2:
            st.metric("Bull Periods", f"{regime_stats['Bull Periods']} ({regime_stats['Bull %']})")
        with col3:
            st.metric("Bear Periods", f"{regime_stats['Bear Periods']} ({regime_stats['Bear %']})")
        with col4:
            st.metric("Sideways Periods", f"{regime_stats['Sideways Periods']} ({regime_stats['Sideways %']})")

        col5, col6 = st.columns(2)
        with col5:
            st.metric("Regime Changes", regime_stats['Regime Changes'])
        with col6:
            st.metric("Avg Duration", f"{regime_stats['Average Duration']:.1f} periods")

    # Display performance metrics if available
    perf_file = dashboard_dir / "performance_metrics.json"
    if perf_file.exists():
        with open(perf_file, 'r') as f:
            perf_metrics = json.load(f)

        st.header("📈 Model Performance")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Train Log-Likelihood", f"{perf_metrics['Train Log-Likelihood']:.0f}")
        with col2:
            st.metric("Test Log-Likelihood", f"{perf_metrics['Test Log-Likelihood']:.0f}")
        with col3:
            st.metric("Degradation", f"{perf_metrics['Degradation %']:.1f}%")

        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("Test AIC", f"{perf_metrics['Test AIC']:.0f}")
        with col5:
            st.metric("Test BIC", f"{perf_metrics['Test BIC']:.0f}")
        with col6:
            decision_emoji = "✅" if "YES" in perf_metrics['Decision'] else "⚠️" if "CAUTION" in perf_metrics['Decision'] else "❌"
            st.metric("Decision", f"{decision_emoji} {perf_metrics['Decision'].split()[-1]}")

    # Display regime timeline
    timeline_file = dashboard_dir / "regime_timeline.html"
    if timeline_file.exists():
        st.header("📅 Regime Timeline")
        with open(timeline_file, 'r') as f:
            st.components.v1.html(f.read(), height=500, scrolling=True)

    # Display regime distribution
    col1, col2 = st.columns(2)

    with col1:
        dist_file = dashboard_dir / "regime_distribution.html"
        if dist_file.exists():
            st.header("🥧 Regime Distribution")
            with open(dist_file, 'r') as f:
                st.components.v1.html(f.read(), height=450, scrolling=True)

    with col2:
        alloc_file = dashboard_dir / "portfolio_allocation_chart.html"
        if alloc_file.exists():
            st.header("💼 Portfolio Allocations")
            with open(alloc_file, 'r') as f:
                st.components.v1.html(f.read(), height=450, scrolling=True)

    # Display portfolio allocation recommendations
    alloc_csv = dashboard_dir / "portfolio_allocations.csv"
    if alloc_csv.exists():
        st.header("📋 Allocation Recommendations")
        allocations = pd.read_csv(alloc_csv)
        st.dataframe(allocations, use_container_width=True, hide_index=True)

    # Display risk metrics
    risk_file = dashboard_dir / "risk_metrics.json"
    if risk_file.exists():
        with open(risk_file, 'r') as f:
            risk_metrics = json.load(f)

        st.header("⚠️ Risk Metrics by Regime")

        risk_df = pd.DataFrame(risk_metrics).T
        risk_df.index.name = 'Regime'
        risk_df = risk_df.reset_index()

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

    # Display feature importance
    feature_file = dashboard_dir / "feature_importance.html"
    if feature_file.exists():
        st.header("🔍 Feature Importance Heatmap")
        with open(feature_file, 'r') as f:
            st.components.v1.html(f.read(), height=600, scrolling=True)

    # Download section
    st.header("💾 Download Data")
    col1, col2, col3 = st.columns(3)

    with col1:
        if alloc_csv.exists():
            with open(alloc_csv, 'rb') as f:
                st.download_button(
                    label="📥 Download Allocations CSV",
                    data=f,
                    file_name="portfolio_allocations.csv",
                    mime="text/csv"
                )

    with col2:
        perf_csv = dashboard_dir / "performance_summary.csv"
        if perf_csv.exists():
            with open(perf_csv, 'rb') as f:
                st.download_button(
                    label="📥 Download Performance CSV",
                    data=f,
                    file_name="performance_summary.csv",
                    mime="text/csv"
                )

    with col3:
        if risk_file.exists():
            with open(risk_file, 'rb') as f:
                st.download_button(
                    label="📥 Download Risk Metrics JSON",
                    data=f,
                    file_name="risk_metrics.json",
                    mime="application/json"
                )

# ========== Portfolio Backtest Page ==========
elif page == "Portfolio Backtest":
    st.title("💰 Portfolio Backtest Analysis")

    backtest_dir = Path("dashboard_outputs/backtest_results")

    # Check if backtest results exist
    if not backtest_dir.exists():
        st.warning("⚠️ No backtest results found. Please run the model first:")
        st.code("python -m model", language="bash")
        st.info("The model will automatically run portfolio backtesting and generate visualizations.")
        st.stop()

    # Load metadata
    metadata_file = backtest_dir / "backtest_metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)

        # Display summary
        st.header("📊 Backtest Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Date Range", f"{metadata['date_range']['start'][:10]} to {metadata['date_range']['end'][:10]}")
        with col2:
            st.metric("Best Strategy", metadata['best_strategy'])
        with col3:
            st.metric("Transaction Cost", f"{metadata['transaction_cost']*100:.2f}%")

    # Display metrics comparison
    metrics_file = backtest_dir / "data/metrics_summary.csv"
    if metrics_file.exists():
        st.header("📈 Performance Metrics Comparison")
        metrics_df = pd.read_csv(metrics_file, index_col=0)

        # Format the dataframe for better display
        formatted_metrics = metrics_df.style.format({
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
        })

        st.dataframe(formatted_metrics, use_container_width=True)

    # Display visualizations
    st.header("📉 Portfolio Performance Visualizations")

    charts_dir = backtest_dir / "charts"

    # Chart 1: Portfolio Value Over Time (full width)
    portfolio_chart = charts_dir / "portfolio_value_comparison.png"
    if portfolio_chart.exists():
        st.subheader("Portfolio Value Over Time")
        st.image(str(portfolio_chart), use_column_width=True)
    else:
        st.warning("Portfolio value chart not found")

    # Chart 2 & 3: Cumulative Returns and Drawdowns (side by side)
    col1, col2 = st.columns(2)

    with col1:
        returns_chart = charts_dir / "cumulative_returns.png"
        if returns_chart.exists():
            st.subheader("Cumulative Returns")
            st.image(str(returns_chart), use_column_width=True)

    with col2:
        drawdown_chart = charts_dir / "drawdown_analysis.png"
        if drawdown_chart.exists():
            st.subheader("Drawdown Analysis")
            st.image(str(drawdown_chart), use_column_width=True)

    # Chart 4 & 5: Sharpe and Annual Returns (side by side)
    col1, col2 = st.columns(2)

    with col1:
        sharpe_chart = charts_dir / "sharpe_ratio_comparison.png"
        if sharpe_chart.exists():
            st.subheader("Sharpe Ratio Comparison")
            st.image(str(sharpe_chart), use_column_width=True)

    with col2:
        annual_chart = charts_dir / "annual_returns_by_strategy.png"
        if annual_chart.exists():
            st.subheader("Annual Returns by Strategy")
            st.image(str(annual_chart), use_column_width=True)

    # Chart 6: Regime Timeline with Portfolio (full width)
    regime_chart = charts_dir / "regime_timeline_portfolio.png"
    if regime_chart.exists():
        st.subheader("Regime Timeline with Portfolio Value")
        st.image(str(regime_chart), use_column_width=True)

    # Regime Allocation History
    alloc_history_file = backtest_dir / "data/regime_allocations.csv"
    if alloc_history_file.exists():
        st.header("📋 Regime Allocation History")
        alloc_history = pd.read_csv(alloc_history_file)

        # Show summary statistics
        st.write("**Regime Distribution:**")
        regime_counts = alloc_history['Regime_Name'].value_counts()
        for regime, count in regime_counts.items():
            pct = count / len(alloc_history) * 100
            st.write(f"- {regime}: {count} periods ({pct:.1f}%)")

        # Show sample of allocation history
        with st.expander("View Allocation History Details"):
            st.dataframe(alloc_history.head(50), use_container_width=True, hide_index=True)

    # Download section
    st.header("💾 Download Results")
    col1, col2, col3 = st.columns(3)

    with col1:
        if metrics_file.exists():
            with open(metrics_file, 'rb') as f:
                st.download_button(
                    label="📥 Download Metrics CSV",
                    data=f,
                    file_name="backtest_metrics.csv",
                    mime="text/csv"
                )

    with col2:
        results_file = backtest_dir / "data/strategy_results.csv"
        if results_file.exists():
            with open(results_file, 'rb') as f:
                st.download_button(
                    label="📥 Download Portfolio Values",
                    data=f,
                    file_name="portfolio_values.csv",
                    mime="text/csv"
                )

    with col3:
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                st.download_button(
                    label="📥 Download Metadata JSON",
                    data=f,
                    file_name="backtest_metadata.json",
                    mime="application/json"
                )

# ========== Price History Page (placeholder) ==========
elif page == "Price History":
    st.title("📈 Price History")
    st.info("Navigate to pages/price_history.py for price history analysis")

# ========== Moving Averages Page (placeholder) ==========
elif page == "Moving Averages":
    st.title("📊 Moving Averages")
    st.info("Navigate to pages/moving_averages.py for moving average analysis")
