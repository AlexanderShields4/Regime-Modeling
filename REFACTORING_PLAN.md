# Refactoring Plan: Regime-Modeling

## Current State Assessment

| Aspect | Score | Notes |
|--------|-------|-------|
| Architecture | 7/10 | Clear data pipeline, but monolithic files |
| HMM Implementation | 8/10 | Correct hmmlearn usage, proper Viterbi decoding |
| Testing | 2/10 | Only one functional integration test |
| Documentation | 5/10 | Milestones exist, no inline docs |
| Error Handling | 5/10 | Some try/except, but silent failures |
| Maintainability | 5/10 | No type hints, magic numbers, no logging |
| Production Readiness | 4/10 | Research code, needs hardening |

---

## Phase 1: Foundation (No Behavior Changes)

Goal: Set up proper project structure and tooling without changing any logic.

### 1.1 — Fix requirements.txt & modernize packaging

- Remove duplicate entries from requirements.txt
- Pin all dependency versions
- Remove unused dependencies (cvxpy, streamlit)
- Create `pyproject.toml` with proper metadata and dependency groups
  - `[project.dependencies]` for core deps
  - `[project.optional-dependencies.dev]` for test/lint tools

### 1.2 — Create package structure

- Create `src/regime_modeling/` directory with `__init__.py`
- Move files into the target structure (see CLAUDE.md)
- Update all import paths
- Create `scripts/` directory for entry points
- Verify everything still runs with `python scripts/run_best_config.py`

### 1.3 — Centralize configuration

- Extract all hardcoded values into `src/regime_modeling/config/settings.py`:
  - Ticker lists (currently in data_utils.py)
  - Default HMM parameters (n_iter, covariance_type, random_state)
  - Feature defaults (volatility_window, rsi_period, momentum_period)
  - Portfolio constants (risk_free_rate, rebalancing frequencies)
  - Cache settings (max_age_hours, cache directory paths)
  - Output paths (models dir, dashboard_outputs dir)
- Use dataclasses or Pydantic models for config validation

### 1.4 — Add logging

- Replace all `print()` calls with `logging` module
- Configure log levels (DEBUG for grid search details, INFO for normal, WARNING for issues)
- Add file handler for persistent logs

---

## Phase 2: Break Up Monoliths

Goal: Split model.py (1245 lines) and other large files into focused modules.

### 2.1 — Extract HMM core from model.py

- `src/regime_modeling/models/hmm.py`: GaussianHMM wrapper
  - `train_hmm(features, n_components, n_iter, covariance_type)` → trained model
  - `decode_regimes(model, features)` → regime labels
  - `evaluate_model(model, train_features, test_features)` → metrics dict
- `src/regime_modeling/models/regime.py`: Regime classification
  - `classify_regimes(features, labels)` → {0: "Bull", 1: "Bear", 2: "Sideways"}
  - `analyze_regime_characteristics(features, labels)` → stats per regime

### 2.2 — Extract scoring & grid search

- `src/regime_modeling/analysis/scoring.py`: Composite scoring logic
  - `calculate_composite_score(model_metrics, portfolio_metrics)` → float
  - `rank_configurations(results_list)` → sorted DataFrame
- `src/regime_modeling/grid_search/runner.py`: Parallel execution
  - `run_grid_search(param_grid, n_workers)` → results DataFrame
  - `_run_single_config(config_id, params)` → single result dict
- `src/regime_modeling/grid_search/reporting.py`: Results output
  - `save_results(results_df, output_path)`
  - `print_top_configs(results_df, n=10)`

### 2.3 — Extract portfolio modules

- `src/regime_modeling/portfolio/metrics.py`: PortfolioMetrics class (already clean)
- `src/regime_modeling/portfolio/allocation.py`: Regime-to-allocation mapping
- `src/regime_modeling/portfolio/backtest.py`: Backtesting engine
- `src/regime_modeling/portfolio/visualization.py`: Chart generation

### 2.4 — Extract analysis/reporting

- `src/regime_modeling/analysis/transitions.py`: Transition matrix computation
- `src/regime_modeling/analysis/regime_stats.py`: Dashboard output generation

---

## Phase 3: Add Type Safety & Validation

Goal: Make the codebase safe to refactor further.

### 3.1 — Add type hints to all public functions

- Start with data layer (inputs/outputs are DataFrames with known columns)
- Add to model layer (feature arrays, label arrays, metric dicts)
- Add to portfolio layer (allocation dicts, metrics)
- Use `TypedDict` for structured dictionaries (model results, config dicts)

### 3.2 — Add data validation

- Validate raw data from yfinance (expected columns, date range, no all-NaN)
- Validate features before HMM training (no NaN, no infinite, correct shape)
- Validate model outputs (regime labels in expected range, probabilities sum to 1)
- Validate portfolio inputs (allocations sum to 1.0, no negative weights)

### 3.3 — Improve error handling

- Replace silent `try/except` in grid search with proper error propagation
- Add meaningful error messages with context (which config failed, why)
- Add retry logic for yfinance API calls (transient network failures)

---

## Phase 4: Testing

Goal: Build confidence for future changes.

### 4.1 — Unit tests for feature engineering

- Test each feature calculator (returns, volatility, RSI, momentum, breadth)
- Test with known input/output pairs
- Test edge cases (single row, all NaN, constant prices)

### 4.2 — Unit tests for HMM wrapper

- Test model training with synthetic data (known regime structure)
- Test regime classification logic (ensure Bull has highest Sharpe)
- Test model evaluation metrics (log-likelihood, AIC, BIC calculations)

### 4.3 — Unit tests for portfolio

- Test PortfolioMetrics calculations against known values
- Test allocation strategies (weights sum to 1.0)
- Test backtesting engine with synthetic regime labels

### 4.4 — Integration test

- Migrate existing test_grid_search.py to pytest
- Add end-to-end pipeline test (small data → features → HMM → backtest → score)

---

## Phase 5: Model Persistence & Reproducibility

### 5.1 — Model versioning

- Save models with metadata (parameters, training date, data hash, metrics)
- Use joblib instead of raw pickle for sklearn-compatible serialization
- Add model loading with validation (check parameter compatibility)

### 5.2 — Reproducibility

- Fix random seeds throughout pipeline (numpy, hmmlearn)
- Log exact data used for training (date range, ticker list, feature set)
- Save scaler alongside model (needed for inference on new data)

---

## Phase 6: API Layer (if frontend planned)

### 6.1 — Define service interfaces

- `DataService`: fetch, cache, validate market data
- `ModelService`: train, evaluate, save/load HMM models
- `BacktestService`: run backtests, compute metrics, generate reports
- `GridSearchService`: configure, execute, report grid searches

### 6.2 — Build REST API (FastAPI)

- `GET /api/data/status` — cache status, last update
- `POST /api/model/train` — train new HMM with parameters
- `GET /api/model/{id}/regimes` — current regime predictions
- `POST /api/backtest/run` — run backtest with strategy config
- `GET /api/backtest/{id}/results` — backtest results & metrics
- `POST /api/grid-search/run` — launch grid search (async)
- `GET /api/grid-search/{id}/status` — grid search progress

---

## Execution Order & Dependencies

```
Phase 1.1 (requirements) ──┐
Phase 1.3 (config)       ──┤
Phase 1.4 (logging)      ──┼──► Phase 1.2 (restructure) ──► Phase 2 (split monoliths)
                            │                                        │
                            │                                        ▼
                            │                               Phase 3 (types & validation)
                            │                                        │
                            │                                        ▼
                            │                               Phase 4 (testing)
                            │                                        │
                            │                                        ▼
                            │                               Phase 5 (persistence)
                            │                                        │
                            │                                        ▼
                            └──────────────────────────────► Phase 6 (API layer)
```

## Guiding Principles

1. **No behavior changes during restructuring** — every phase should produce identical outputs to the current code
2. **One phase at a time** — complete and verify each phase before starting the next
3. **Run existing tests after every change** — `python test_grid_search.py` must pass (or its migrated equivalent)
4. **Commit after each sub-phase** — easy to revert if something breaks
5. **Preserve the current best_config.txt and grid search results** — these are valuable artifacts
