"""Centralized configuration constants for regime_modeling."""

# ---------------------------------------------------------------------------
# Ticker lists
# ---------------------------------------------------------------------------

# Excluded tickers with insufficient history (pre-2000): GOOGL, META, TSLA, NFLX, V, MA, PYPL, TOT, BABA
STOCK_TICKERS = [
    # Tech
    "AAPL", "MSFT", "AMZN", "NVDA", "AMD", "INTC",
    # Finance
    "JPM", "BAC", "GS", "MS", "C", "WFC", "AXP",
    # Healthcare
    "JNJ", "PFE", "UNH", "MRK", "ABT", "TMO", "LLY", "BMY", "AMGN",
    # Industrials
    "BA", "CAT", "GE", "MMM", "HON", "LMT", "UPS", "FDX",
    # Consumer
    "PG", "KO", "PEP", "NKE", "MCD", "DIS", "SBUX", "COST", "HD", "WMT",
    # Energy
    "XOM", "CVX", "BP", "COP", "SLB",
    # Global examples
    "TM", "NSRGY", "TSM", "RIO", "SAP",
]

INDEX_TICKERS = [
    "^GSPC",  # S&P 500
    "^IXIC",  # NASDAQ
    "^DJI",   # Dow Jones
    "^RUT",   # Russell 2000
    "^VIX",   # VIX
    "^FTSE",  # FTSE 100
    "^N225",  # Nikkei 225
    "^HSI",   # Hang Seng
]

# Excluded tickers with insufficient history: PSX, MPC, WEAT, SOYB, CORN, UNG, SLV, USO, CF, GLD, AGI, VALE, BG
NATURAL_RESOURCES_TICKERS = [
    # Energy
    "XOM", "CVX", "BP", "COP", "SLB", "EOG", "VLO",
    # Metals & Mining
    "FCX", "NEM", "RIO", "BHP", "AA", "CLF",
    # Agriculture
    "ADM", "TSN", "CAG", "MOS", "IP", "WY",
]

BOND_ETF_TICKERS = [
    "TLT",   # iShares 20+ Year Treasury Bond ETF (primary)
    "IEF",   # iShares 7-10 Year Treasury Bond ETF (intermediate)
    "AGG",   # iShares Core U.S. Aggregate Bond ETF (broad market)
]

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

DEFAULT_START_DATE = "2000-01-01"
DEFAULT_INTERVAL = "1d"

# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

CACHE_DIR = "data_cache"
CACHE_MAX_AGE_HOURS = 24

# ---------------------------------------------------------------------------
# Feature engineering defaults
# ---------------------------------------------------------------------------

DEFAULT_VOLATILITY_WINDOW = 20
DEFAULT_RSI_PERIOD = 14
DEFAULT_MOMENTUM_PERIOD = 10
DEFAULT_MOVING_AVERAGE_WINDOW = 20

# ---------------------------------------------------------------------------
# HMM model
# ---------------------------------------------------------------------------

N_REGIMES = 3
DEFAULT_HMM_PARAMS = {
    "n_iter": 5000,
    "covariance_type": "full",
    "random_state": 42,
    "train_ratio": 0.8,
}

# ---------------------------------------------------------------------------
# Grid search defaults
# ---------------------------------------------------------------------------

DEFAULT_GRID_SEARCH_RANGES = {
    "n_stocks_range": [5, 7, 10, 15],
    "n_indices_range": [0, 3, 5],
    "volatility_window_range": [10, 20, 30],
    "rsi_period_range": [10, 14, 20],
    "momentum_period_range": [5, 10, 15],
    "n_iter_range": [5000],
    "covariance_type_range": ["full"],
}

# ---------------------------------------------------------------------------
# Portfolio / backtesting
# ---------------------------------------------------------------------------

RISK_FREE_RATE = 0.02
TRADING_DAYS_PER_YEAR = 252
DEFAULT_TRANSACTION_COST = 0.001
DEFAULT_INITIAL_CAPITAL = 100_000

REGIME_ALLOCATIONS = {
    "Bull":     {"stocks": 1.0, "bonds": 0.0, "cash": 0.0},
    "Bear":     {"stocks": 0.3, "bonds": 0.5, "cash": 0.2},
    "Sideways": {"stocks": 0.8, "bonds": 0.1, "cash": 0.1},
    "Unknown":  {"stocks": 0.6, "bonds": 0.3, "cash": 0.1},
}

# ---------------------------------------------------------------------------
# Composite scoring (lower = better)
# ---------------------------------------------------------------------------

SCORING = {
    # Portfolio score targets and penalty multipliers
    # Targets are calibrated to realistic top-tier values that align with
    # the DECISION_THRESHOLDS "excellent" category, so scoring differentiates
    # good configs rather than penalising everything equally.
    "target_cagr": 0.15,
    "cagr_penalty_multiplier": 250,
    "target_sharpe": 1.2,
    "sharpe_penalty_multiplier": 25,
    "drawdown_penalty_multiplier": 150,
    "target_sortino": 1.8,
    "sortino_penalty_multiplier": 15,
    "target_calmar": 0.75,
    "calmar_penalty_multiplier": 20,

    # Statistical score penalties
    "regime_usage_penalty_multiplier": 30,
    "insufficient_switching_penalty": 20,
    "excessive_switching_penalty": 0.5,
    "short_duration_threshold": 5,
    "short_duration_penalty_multiplier": 25,
    "long_duration_threshold": 50,
    "long_duration_penalty_multiplier": 3,
    "degradation_penalty_multiplier": 8,
    "bic_normalization_divisor": 1000,
    "diversity_penalty_multiplier": 50,

    # Normalization factors
    "portfolio_normalization_divisor": 250,
    "statistical_normalization_divisor": 150,
    "portfolio_weight": 600,
    "statistical_weight": 400,
}

DECISION_THRESHOLDS = {
    "ready_degradation": 10,
    "ready_min_duration": 5,
    "ready_max_duration": 50,
    "caution_degradation": 30,

    # Grid search quality categories
    "excellent_sharpe": 1.5,
    "excellent_max_dd": -0.25,
    "excellent_cagr": 0.12,
    "excellent_degradation": 10,

    "good_sharpe": 1.0,
    "good_max_dd": -0.30,
    "good_cagr": 0.08,
    "good_degradation": 15,

    "acceptable_sharpe": 0.5,
    "acceptable_max_dd": -0.40,
    "acceptable_cagr": 0.04,
}

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

OUTPUT_DIR = "dashboard_outputs"
BACKTEST_OUTPUT_DIR = "dashboard_outputs/backtest_results"
MODELS_DIR = "models"
BEST_CONFIG_FILE = "best_config.txt"
