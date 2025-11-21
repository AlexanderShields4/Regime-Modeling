# Hidden Markov Model for Market Regime Detection

A statistical framework for identifying latent market regimes using Hidden Markov Models trained on equity price data. The system infers three distinct market states (bull, bear, sideways) from observable features including returns, volatility, and technical indicators.

## Model Architecture

The core implementation uses a 3-state Gaussian HMM where each hidden state represents a market regime. Observable emissions consist of engineered features from stock and index price series. State transitions follow a learned Markov process, with the Viterbi algorithm used for optimal state sequence decoding.

Key components:
- **State space**: 3 latent states corresponding to market regimes
- **Observations**: Multidimensional Gaussian emissions (12-21 features)
- **Covariance**: Full covariance matrix captures feature dependencies within states
- **Training**: EM algorithm with configurable iteration counts (default 5000)

### Feature Engineering

Raw price data undergoes transformation into stationary time series and technical indicators:

**Primary features** (always included):
- Log returns for selected equities and indices
- Rolling volatility (standard deviation of returns over configurable window)

**Optional features**:
- RSI (Relative Strength Index) on key assets
- Momentum (rate of change over period)
- Market breadth metrics (percentage positive, cross-sectional volatility)

The feature space typically spans 12-21 dimensions depending on the number of assets selected and which optional features are enabled. Feature scaling uses StandardScaler to normalize observations before model training.

## Data Pipeline

Market data flows through several stages:

1. **Acquisition**: Price series fetched from financial APIs (yfinance)
2. **Caching**: Raw data persisted to avoid repeated API calls during experimentation
3. **Feature engineering**: Transformation of prices into model-ready features
4. **Scaling**: Standardization of features for HMM input
5. **Training**: EM algorithm learns state parameters and transition probabilities

The caching layer stores merged dataframes containing stock, index, and resource prices along with their moving averages. Cache invalidation occurs manually when fresh data is needed.

## Parameter Optimization

Grid search functionality tests combinations of:
- Asset selection (n_stocks, n_indices)
- Feature engineering parameters (volatility_window, rsi_period, momentum_period)
- Feature inclusion flags (which technical indicators to use)
- HMM hyperparameters (n_iter, covariance_type)

Each configuration undergoes train/test backtesting with an 80/20 temporal split. Model quality metrics include:

- **Degradation**: Percentage difference in per-sample log-likelihood between train and test sets
- **AIC/BIC**: Information criteria penalizing model complexity
- **Regime stability**: Average duration of decoded states and frequency of transitions

The search runs in parallel across CPU cores, with results ranked by a composite score. Configurations with degradation under 10% and regime durations between 5-50 periods are considered production-ready.

## Model Evaluation

Backtesting splits time series data chronologically. The model trains on earlier data and evaluates on held-out recent periods. Key metrics:

**Log-likelihood**: Measures how well the model explains test data. Reported both as total and per-sample averages to enable fair comparison across different data lengths.

**Degradation**: Calculated as the percentage change in per-sample log-likelihood from training to test set. Positive values indicate worse test performance. Values under 10% suggest good generalization.

**Regime characteristics**: Decoded states are analyzed for economic interpretation by examining mean returns and volatility during each regime. States naturally cluster into bull (positive returns, low volatility), bear (negative returns), and sideways (near-zero returns or high volatility) patterns.

**Transition stability**: Excessive regime switching indicates noisy predictions unsuitable for portfolio management. Optimal configurations maintain state persistence while remaining responsive to genuine market shifts.

## File Organization

Core modules:
- `model.py` - HMM training, backtesting, grid search implementation
- `features.py` - Feature engineering pipeline and data caching
- `pages_utils.py` - API integration for data acquisition

Execution scripts:
- `cache_data.py` - Pre-cache market data
- `quick_grid_test.py` - Small parameter sweep for validation
- `overnight_grid_search.py` - Comprehensive search across parameter space
- `run_grid_search.py` - Example configurations for grid search

The `data_cache/` directory stores pickled dataframes for fast loading. The `models/` directory contains serialized trained models with their scalers and parameter configurations. Grid search results output to timestamped CSV files and `best_config.txt`.

## Implementation Notes

The system addresses several practical challenges in regime detection:

**Stationarity**: Raw prices are non-stationary. The feature engineering pipeline transforms prices into log returns and other indicators that exhibit more stable statistical properties suitable for HMM modeling.

**Overfitting**: Train/test validation catches models that memorize training data. The degradation metric quantifies generalization. Information criteria (AIC/BIC) penalize model complexity.

**Computation**: Grid search can test hundreds of configurations. Parallel processing distributes work across cores. Data caching eliminates redundant API calls that would trigger rate limiting.

**Parameter sensitivity**: Different assets, time periods, and feature combinations produce varying results. Grid search automates the exploration of this space. The scoring function balances multiple objectives (fit quality, generalization, regime stability).

**Reproducibility**: Random seeds ensure consistent results. All parameters are logged with trained models. The best configuration gets saved as executable Python code.

## Model Persistence

Trained models serialize to pickle files containing:
- Fitted GaussianHMM object
- StandardScaler with learned parameters
- Feature engineering configuration
- Performance metrics from backtesting
- Training timestamp

Loading a saved model restores the complete pipeline needed to process new data and generate regime predictions. The scaler must be applied to new features using the transform method (not fit_transform) to maintain consistency with training.

## Dependencies

Core requirements:
- hmmlearn (HMM implementation)
- scikit-learn (scaling, metrics)
- pandas (data manipulation)
- numpy (numerical operations)
- yfinance (market data)

The system runs on Python 3.x. Parallel processing uses the standard library multiprocessing module.

## Results Tracking

Each backtest run appends to `Results.md` with timestamp, parameters, performance metrics, and portfolio readiness assessment. Grid searches generate CSV files with all tested configurations sorted by score. This creates an audit trail of experiments and facilitates comparison across parameter settings.

The best configuration from each grid search gets written to `best_config.txt` as ready-to-run Python code, making it easy to retrain the optimal model or deploy it for predictions.
