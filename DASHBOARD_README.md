# HMM Regime Analysis Dashboard

This dashboard provides comprehensive visualization and analysis of market regime predictions from the HMM (Hidden Markov Model).

## Quick Start

### 1. Run the HMM Model

First, generate the regime analysis data by running the model:

```bash
python -m model
```

This will:
- Train the HMM model on market data
- Generate regime predictions
- Create dashboard outputs in the `dashboard_outputs/` folder
- Calculate risk metrics and portfolio allocations

### 2. Launch the Dashboard

Start the Streamlit dashboard:

```bash
streamlit run __main__.py
```

### 3. View Results

Navigate to the "HMM Regime Analysis" page in the sidebar to see:

- **Regime Statistics**: Distribution of bull, bear, and sideways markets
- **Model Performance**: Training/testing metrics, AIC, BIC, degradation
- **Regime Timeline**: Visual timeline showing regime changes over time
- **Regime Distribution**: Pie chart of regime percentages
- **Portfolio Allocations**: Recommended stock/bond allocations by regime
- **Risk Metrics**: Sharpe ratio, volatility, max drawdown by regime
- **Feature Importance**: Heatmap showing which features drive each regime

## Dashboard Features

### Regime Statistics
- Total periods analyzed
- Number of periods in each regime (bull, bear, sideways)
- Regime change frequency
- Average regime duration

### Model Performance Metrics
- Train/Test Log-Likelihood
- Degradation percentage (out-of-sample performance)
- AIC/BIC model selection criteria
- Portfolio readiness decision (✅/⚠️/❌)

### Portfolio Recommendations
Each regime has specific allocation recommendations:
- **Bull Market**: 80% stocks, 20% bonds (growth-focused)
- **Bear Market**: 30% stocks, 70% bonds (risk-averse)
- **Sideways Market**: 50% stocks, 50% bonds (balanced)

### Risk Metrics by Regime
- Average return per regime
- Volatility (standard deviation)
- Sharpe ratio (risk-adjusted returns)
- Maximum drawdown

## Files Generated

The model automatically creates these files in `dashboard_outputs/`:

| File | Description |
|------|-------------|
| `regime_timeline.html` | Interactive timeline visualization |
| `regime_distribution.html` | Pie chart of regime distribution |
| `portfolio_allocation_chart.html` | Bar chart of allocations |
| `feature_importance.html` | Heatmap of feature means |
| `regime_stats.json` | Regime statistics data |
| `performance_metrics.json` | Model performance data |
| `risk_metrics.json` | Risk metrics by regime |
| `portfolio_allocations.csv` | Allocation recommendations table |
| `performance_summary.csv` | Performance metrics table |

## Refreshing Dashboard Data

Every time you run `python -m model`, the `dashboard_outputs/` folder is completely replaced with fresh data. This ensures:
- No stale data in the dashboard
- Latest model results are always shown
- Consistent visualization across runs

## Running with Different Configurations

You can modify parameters in `model.py` or use the grid search:

```python
# Run with custom parameters
from model import run_hmm_model

run_hmm_model(
    n_stocks=15,
    n_indices=5,
    include_returns=True,
    include_volatility=True,
    include_rsi=False,
    include_momentum=True,
    include_market_breadth=True,
    backtest=True
)
```

## Tips

1. **Backtesting**: Set `backtest=True` to get performance metrics
2. **Feature Selection**: Use different feature combinations to improve model fit
3. **Download Data**: Use the download buttons to export CSV/JSON files
4. **Regime Interpretation**:
   - Bull: Positive returns, lower volatility
   - Bear: Negative returns, higher volatility
   - Sideways: Neutral returns, moderate volatility
