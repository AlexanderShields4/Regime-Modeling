# Regime-Modeling Project Guide

## Project Overview

Adaptive regime-switching investment strategy using Hidden Markov Models (HMM) to detect market regimes (Bull/Bear/Sideways) and dynamically allocate portfolios. Built by Jack Bray, Alex Shields, and Getchell Gibbons.

**Core pipeline:** yfinance data → feature engineering → HMM training → regime detection → portfolio backtesting → performance evaluation.

## Architecture

### Current Structure (flat scripts)

```
.
├── data/                    # Data fetching modules (yfinance wrappers)
│   ├── data_utils.py        # Core ticker lists & download logic
│   ├── ind_stocks.py        # Individual stock loader
│   ├── indices.py           # Market indices loader
│   ├── natural_resources.py # Commodity/resource loader
│   └── moving_averages.py   # Moving average calculations
├── models/                  # Saved HMM models (pickle files)
├── dashboard_outputs/       # Generated analysis outputs (JSON, CSV, charts)
├── data_cache/              # Cached raw data (pkl + csv)
├── Milestones/              # Project documentation (Proposal, M1, M2)
├── model.py                 # HMM model, grid search, scoring (1245 lines - MONOLITH)
├── features.py              # Feature engineering & data caching (302 lines)
├── portfolio_backtest.py    # PortfolioMetrics class & backtesting (378 lines)
├── portfolio_visualizer.py  # Matplotlib/Plotly chart generation
├── inspect_regimes.py       # Regime analysis & transition reporting
├── pages_utils.py           # Data fetch orchestration (thin wrapper)
├── cache_data.py            # Data caching utility
├── run_best_config.py       # Re-run best HMM configuration
├── overnight_grid_search.py # Full grid search runner (~1500 configs)
├── test_grid_search.py      # Quick grid search validation
└── requirements.txt         # Dependencies (has duplicates, needs cleanup)
```

### Target Structure (after refactoring)

```
.
├── src/regime_modeling/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py      # Centralized config (tickers, defaults, paths)
│   │   └── grid_search.py   # Grid search parameter definitions
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetcher.py        # yfinance data fetching (unified)
│   │   ├── cache.py          # Caching logic with proper invalidation
│   │   └── validation.py     # Data quality checks
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engineering.py    # Feature calculation (returns, vol, RSI, etc.)
│   │   └── selection.py      # Feature selection & combination logic
│   ├── models/
│   │   ├── __init__.py
│   │   ├── hmm.py            # HMM training, fitting, decoding
│   │   ├── regime.py         # Regime classification (Bull/Bear/Sideways)
│   │   └── persistence.py    # Model save/load with versioning
│   ├── portfolio/
│   │   ├── __init__.py
│   │   ├── backtest.py       # Backtesting engine
│   │   ├── metrics.py        # PortfolioMetrics (CAGR, Sharpe, Sortino, etc.)
│   │   ├── allocation.py     # Regime-based allocation strategies
│   │   └── visualization.py  # Portfolio charts (Plotly/Matplotlib)
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── transitions.py    # Transition matrix analysis
│   │   ├── regime_stats.py   # Regime statistics & reporting
│   │   └── scoring.py        # Composite score calculation
│   └── grid_search/
│       ├── __init__.py
│       ├── runner.py          # Parallel grid search execution
│       └── reporting.py       # Grid search results & ranking
├── scripts/
│   ├── run_best_config.py
│   ├── run_grid_search.py
│   └── cache_data.py
├── tests/
│   ├── __init__.py
│   ├── test_features.py
│   ├── test_hmm.py
│   ├── test_backtest.py
│   └── test_scoring.py
├── outputs/                   # All generated outputs (gitignored)
├── pyproject.toml             # Modern Python packaging
├── CLAUDE.md
└── README.md (if needed)
```

## Key Technical Details

### HMM Implementation
- **Library:** `hmmlearn.GaussianHMM` with 3 hidden states
- **Decoding:** Viterbi algorithm
- **Training:** Expectation-Maximization (EM), 3000-7000 iterations
- **Covariance types:** `full` (preferred) or `tied`
- **Regime classification:** By Sharpe ratio ranking (Bull > Sideways > Bear)

### Feature Engineering Pipeline
1. Raw prices fetched via yfinance (43 stocks, 8 indices, 12 resources, 3 bond ETFs)
2. Features computed: log returns, rolling volatility, RSI, momentum, market breadth
3. Features concatenated, NaN-dropped, StandardScaler applied
4. Train/test split is chronological (no shuffle) to prevent look-ahead bias

### Grid Search
- Parallel execution via `multiprocessing.Pool` (50% CPU cores)
- Composite score: 60% portfolio performance + 40% statistical quality
- Lower score = better
- Best config: 18 stocks, 3 indices, vol_window=10, returns+volatility features

### Portfolio Backtesting
- Regime-based allocation strategies (e.g., Bull: 60% equity / 30% bond / 10% gold)
- Benchmarks: 60/40, equal-weight, S&P 500 buy-and-hold
- Metrics: CAGR, Sharpe, Sortino, max drawdown, Calmar, win rate
- Rebalancing frequencies tested: daily, weekly, monthly

## Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Run best configuration
python run_best_config.py

# Run quick grid search test (~5-10 min)
python test_grid_search.py

# Run full overnight grid search (~hours)
python overnight_grid_search.py

# Cache fresh data
python cache_data.py

# Inspect regime analysis
python inspect_regimes.py
```

## Known Issues & Technical Debt

### Critical
- **model.py is 1245 lines** - monolithic, handles HMM training, grid search, scoring, backtesting orchestration, and dashboard output all in one file
- **No type hints anywhere** - makes refactoring risky without tests
- **No unit tests** - only a functional integration test (test_grid_search.py)
- **requirements.txt has duplicate entries** and missing version pins

### Important
- Hardcoded magic numbers scattered (risk-free rate 0.02, random_state 42, cache expiry 24h)
- No logging framework - all print() statements
- Unused dependencies: cvxpy, streamlit (installed but not imported)
- Model persistence uses raw pickle with no versioning or validation
- No data validation/schema checks on loaded market data

### Design Decisions
- Feature engineering happens before train/test split (scaler fitted on train only - correct)
- Forward-fill only for missing data (no backward-fill to avoid data leakage)
- 3 regimes chosen (Bull/Bear/Sideways) - hardcoded, not configurable

## Conventions

- Python 3.14 (in venv)
- pandas for all tabular data
- Plotly for interactive charts, Matplotlib for static publication charts
- Pickle for data caching and model persistence
- CSV for analysis outputs and grid search results
- JSON for dashboard metadata
