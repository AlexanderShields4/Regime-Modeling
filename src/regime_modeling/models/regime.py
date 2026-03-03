import pandas as pd
import numpy as np
import warnings
from typing import List

def _extract_return_columns(data: pd.DataFrame) -> List[str]:
    """Extract return columns, excluding engineered features."""
    valid_prefixes = ('stock_', 'index_', 'resource_')
    exclude_patterns = ['_ma_', '_vol', 'momentum', '_mom', 'rsi', '_rsi',
                       '_ema', '_sma', '_ewma']

    return_cols = [
        col for col in data.columns
        if col.lower().startswith(valid_prefixes) and not any(
            pattern in col.lower() for pattern in exclude_patterns
        )
    ]

    return return_cols


def analyze_regime_characteristics(states: np.ndarray, data: pd.DataFrame, n_states: int) -> List[str]:
    """Analyze regime characteristics and classify as Bull/Bear/Sideways."""

    return_cols = _extract_return_columns(data)

    regime_types = []
    if not return_cols:
        warnings.warn(
            "No return columns found for regime classification. "
            "Check include_returns=True and n_stocks>0 in config. "
            "Using conservative allocation (60/30/10) for all regimes.",
            UserWarning,
            stacklevel=2
        )
        return ["Unknown"] * n_states
        
    regime_stats = []
    for regime in range(n_states):
        mask = states == regime
        subset = data[mask]

        if len(subset) == 0:
            regime_stats.append({'mean_return': 0, 'std_return': 1, 'sharpe': 0, 'count': 0})
            continue

        regime_returns = subset[return_cols].mean(axis=1)
        mean_ret = regime_returns.mean()
        std_ret = regime_returns.std()
        sharpe = mean_ret / std_ret if std_ret > 0 else 0

        regime_stats.append({
            'mean_return': mean_ret,
            'std_return': std_ret,
            'sharpe': sharpe,
            'count': len(subset)
        })

    valid_regimes = [(i, stats) for i, stats in enumerate(regime_stats) if stats['count'] > 0]

    if len(valid_regimes) == 0:
        return ["Unknown"] * n_states

    sorted_regimes = sorted(valid_regimes, key=lambda x: x[1]['sharpe'], reverse=True)
    regime_labels = {}
    
    if len(sorted_regimes) == 3:
        # Best Sharpe (high return/low vol) = Bull
        # Worst Sharpe (low return/high vol) = Bear
        # Middle = Sideways
        regime_labels[sorted_regimes[0][0]] = "Bull"
        regime_labels[sorted_regimes[1][0]] = "Sideways"
        regime_labels[sorted_regimes[2][0]] = "Bear"
    elif len(sorted_regimes) == 2:
        # Best = Bull, Worst = Bear
        regime_labels[sorted_regimes[0][0]] = "Bull"
        regime_labels[sorted_regimes[1][0]] = "Bear"
    else:
        # Only 1 regime
        regime_labels[sorted_regimes[0][0]] = "Sideways"

    # Build final list in original order
    for regime in range(n_states):
        if regime in regime_labels:
            regime_types.append(regime_labels[regime])
        else:
            regime_types.append("Unknown")

    return regime_types
