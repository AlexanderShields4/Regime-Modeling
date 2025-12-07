"""
Re-run the best configuration to regenerate regime allocations with fixed labels.
This is much faster than re-running the full grid search.
"""

from model import run_hmm_model

print("Running best configuration from grid search...")
print("This will regenerate regime_allocations.csv with correct Bull/Bear/Sideways labels\n")

# Best configuration from grid search
results = run_hmm_model(
    n_stocks=18,
    n_indices=3,
    volatility_window=10,
    rsi_period=14,
    momentum_period=10,
    include_returns=True,
    include_volatility=True,
    include_rsi=False,
    include_momentum=False,
    include_market_breadth=False,
    n_iter=10000,
    covariance_type='full',
    random_state=42,
    backtest=True,
    train_ratio=0.6,
    generate_outputs=True
)

print("\n" + "="*60)
print("DONE! Check dashboard_outputs/backtest_results/data/regime_allocations.csv")
print("The regime labels should now show Bull/Bear/Sideways instead of all Sideways")
print("="*60)
