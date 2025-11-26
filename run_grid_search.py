"""
Grid Search Example - Find Optimal HMM Parameters

This script runs parallel grid search to find the best parameter combination
for your HMM model. It will test many configurations in parallel using all
available CPU cores.

WARNING: This may take 10-60 minutes depending on your parameter ranges
and number of CPU cores.
"""

from model import grid_search_parameters

# Example 1: Quick search (recommended to start)
print("EXAMPLE 1: QUICK GRID SEARCH")
print("-" * 70)

results_quick = grid_search_parameters(
    n_stocks_range=[5, 7, 10],           # Test 3 stock counts
    n_indices_range=[0, 3],              # Test with/without indices
    volatility_window_range=[20],        # Keep fixed for quick test
    rsi_period_range=[14],               # Keep fixed
    momentum_period_range=[10],          # Keep fixed
    feature_combinations='auto',         # Test 6 smart feature combos
    train_ratio=0.8,                     # 80/20 split
    n_processes=None,                    # Auto-detect cores
    top_n=5                              # Show top 5 results
)

# Total: 3 stocks × 2 indices × 6 features = 36 configurations
# On 8 cores: ~2-5 minutes


# Example 2: Comprehensive search (more thorough)
print("\n\nEXAMPLE 2: COMPREHENSIVE GRID SEARCH")
print("-" * 70)

# Uncomment to run comprehensive search:
"""
results_comprehensive = grid_search_parameters(
    n_stocks_range=[5, 7, 10, 15, 20],   # Test 5 stock counts
    n_indices_range=[0, 3, 5, 7],        # Test 4 index counts
    volatility_window_range=[10, 20, 30], # Test 3 window sizes
    rsi_period_range=[14, 21],           # Test 2 RSI periods
    momentum_period_range=[10, 20],      # Test 2 momentum periods
    feature_combinations='auto',         # 6 feature combos
    train_ratio=0.8,
    n_processes=None,
    top_n=10
)

# Total: 5 x 4 x 3 x 2 x 2 x 6 = 1,440 configurations
# On 8 cores: ~30-90 minutes
"""


# Example 3: Custom feature combinations
print("\n\nEXAMPLE 3: CUSTOM FEATURE COMBINATIONS")
print("-" * 70)

# Define your own feature combinations
custom_features = [
    {'returns': True, 'volatility': True, 'rsi': False, 'momentum': False, 'market_breadth': False},
    {'returns': True, 'volatility': True, 'rsi': True, 'momentum': True, 'market_breadth': True},
]

# Uncomment to run with custom features:
"""
results_custom = grid_search_parameters(
    n_stocks_range=[7, 10],
    n_indices_range=[0],
    volatility_window_range=[20],
    rsi_period_range=[14],
    momentum_period_range=[10],
    feature_combinations=custom_features,  # Use custom
    n_processes=None,
    top_n=5
)
"""


# After grid search completes:
print("\n" + "="*70)
print("WHAT HAPPENS NEXT")
print("="*70)
print("""
1. Results are saved to:
   - grid_search_results_TIMESTAMP.csv (all configurations)
   - best_config.txt (ready-to-use code for best config)

2. Review the results:
   - Lower Score = Better overall performance
   - Check Degradation% (want < 10%)
   - Check Avg_Duration (want 5-50 periods)
   - Check Decision column

3. Use the best configuration:
   - Copy code from best_config.txt
   - Run with backtest=True to validate
   - If satisfied, run with backtest=False to train on all data

4. For portfolio management:
   - Train final model with best config on all data
   - Save model and scaler (see portfolio_manager.py)
   - Deploy for live trading

TIP: Start with Example 1 (quick search) to get a sense of performance.
     Then run Example 2 (comprehensive) overnight for thorough optimization.
""")

print("\n✓ Grid search example complete!")
print("✓ Check the generated CSV file for full results")
print("✓ Check best_config.txt for the winning configuration\n")
