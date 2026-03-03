import numpy as np
from typing import Dict, Any, List

from regime_modeling.config import SCORING, N_REGIMES

def calculate_composite_score(results: Dict[str, Any]) -> float:
    """
    Calculate composite score balancing portfolio and statistical quality.
    Lower score = better.
    """
    S = SCORING

    cagr = results.get('portfolio_cagr', 0)
    returns_penalty = max(0, (S['target_cagr'] - cagr) * S['cagr_penalty_multiplier'])

    sharpe = results.get('portfolio_sharpe', 0)
    sharpe_penalty = max(0, (S['target_sharpe'] - sharpe) * S['sharpe_penalty_multiplier'])

    max_dd = abs(results.get('portfolio_max_dd', -0.5))
    drawdown_penalty = max_dd * S['drawdown_penalty_multiplier']

    sortino = results.get('portfolio_sortino', 0)
    sortino_penalty = max(0, (S['target_sortino'] - sortino) * S['sortino_penalty_multiplier'])

    calmar = results.get('portfolio_calmar', 0)
    calmar_penalty = max(0, (S['target_calmar'] - calmar) * S['calmar_penalty_multiplier'])

    portfolio_score = (
        returns_penalty +
        sharpe_penalty +
        drawdown_penalty +
        sortino_penalty +
        calmar_penalty
    )

    test_states = results['test_states']
    n_regime_changes = results['n_regime_changes']
    avg_duration = results['avg_regime_duration']

    state_counts = np.bincount(test_states)
    state_probs = state_counts / len(test_states)
    regime_entropy = -np.sum(state_probs[state_probs > 0] * np.log(state_probs[state_probs > 0]))
    regime_diversity_score = regime_entropy / np.log(N_REGIMES)

    n_unique_regimes = len(np.unique(test_states))
    regime_usage_penalty = (N_REGIMES - n_unique_regimes) * S['regime_usage_penalty_multiplier'] if n_unique_regimes < N_REGIMES else 0

    switching_penalty = 0
    if n_regime_changes < N_REGIMES:
        switching_penalty = (N_REGIMES - n_regime_changes) * S['insufficient_switching_penalty']
    elif n_regime_changes > len(test_states) / N_REGIMES:
        switching_penalty = (n_regime_changes - len(test_states) / N_REGIMES) * S['excessive_switching_penalty']

    duration_penalty = 0
    if avg_duration < S['short_duration_threshold']:
        duration_penalty = (S['short_duration_threshold'] - avg_duration) * S['short_duration_penalty_multiplier']
    elif avg_duration > S['long_duration_threshold']:
        duration_penalty = (avg_duration - S['long_duration_threshold']) * S['long_duration_penalty_multiplier']

    degradation_penalty = abs(results['degradation']) * S['degradation_penalty_multiplier']
    bic_score = results['test_bic'] / S['bic_normalization_divisor']
    diversity_penalty = (1 - regime_diversity_score) * S['diversity_penalty_multiplier']

    statistical_score = (
        degradation_penalty +
        regime_usage_penalty +
        switching_penalty +
        duration_penalty +
        bic_score +
        diversity_penalty
    )

    portfolio_normalized = portfolio_score / S['portfolio_normalization_divisor']
    statistical_normalized = statistical_score / S['statistical_normalization_divisor']

    composite_score = (
        portfolio_normalized * S['portfolio_weight'] +
        statistical_normalized * S['statistical_weight']
    )

    results['portfolio_score_component'] = portfolio_score
    results['statistical_score_component'] = statistical_score
    results['regime_diversity'] = regime_diversity_score
    results['n_unique_regimes'] = n_unique_regimes

    return composite_score
