import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
import shutil

from regime_modeling.config import OUTPUT_DIR
from regime_modeling.models.regime import analyze_regime_characteristics


def generate_all_outputs(
    model: Any, 
    test_data: pd.DataFrame, 
    test_states: np.ndarray, 
    scaler: Any, 
    regime_types: List[str],
    output_dir: str = OUTPUT_DIR
) -> Optional[Dict]:
    """Generate dashboard, risk metrics, and portfolio backtest."""
    import logging
    logger = logging.getLogger(__name__)

    # Imported here to avoid circular dependencies
    from regime_modeling.portfolio.backtest import calculate_risk_metrics

    dashboard_info = generate_dashboard_outputs(
        model, test_data, test_states, {},
        regime_types=regime_types, 
        output_dir=output_dir
    )
    logger.info(f"Dashboard files saved to: {dashboard_info['output_dir']}/")

    risk_metrics = calculate_risk_metrics(test_data, test_states, regime_types=regime_types)
    if risk_metrics:
        with open(f"{dashboard_info['output_dir']}/risk_metrics.json", 'w') as f:
            json.dump(risk_metrics, f, indent=2)
        logger.info("Risk metrics saved")

    logger.info("\nRunning portfolio backtest on test data...")
    try:
        from regime_modeling.portfolio.backtest import run_portfolio_backtest
        backtest_results = run_portfolio_backtest(test_states, test_data, output_dir=f'{output_dir}/backtest_results')
        logger.info("Portfolio backtest complete!")
        logger.info(f"Best rebalancing frequency: {backtest_results['best_rebalance_freq']}")
        logger.info(f"Visualizations saved to: {backtest_results['output_dir']}/charts/\n")
        return backtest_results
    except Exception as e:
        logger.error(f"Portfolio backtest failed: {e}")
        logger.info("Continuing without backtest results...\n")
        return None


def generate_dashboard_outputs(
    model: Any, 
    data: pd.DataFrame, 
    hidden_states: np.ndarray, 
    backtest_results: Dict[str, Any], 
    regime_types: List[str] = None, 
    output_dir: str = OUTPUT_DIR
) -> Dict[str, Any]:
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if regime_types is not None:
        regime_names = regime_types
    else:
        n_states = len(np.unique(hidden_states))
        regime_names = analyze_regime_characteristics(hidden_states, data, n_states)
        
    color_map = {
        'Bull': 'rgba(0,200,0,0.25)',
        'Bear': 'rgba(220,0,0,0.25)',
        'Sideways': 'rgba(120,120,120,0.25)'
    }

    marker_map = {
        'Bull': 'green',
        'Bear': 'darkred',
        'Sideways': 'gray'
    }

    fig_regime = go.Figure()

    if len(hidden_states) == 0:
        fig_regime.add_annotation(text="No regime data", xref="paper", yref="paper", showarrow=False)
    else:
        regime_changes = np.where(np.diff(hidden_states) != 0)[0]
        starts = np.insert(regime_changes + 1, 0, 0)
        ends = np.append(regime_changes, len(hidden_states) - 1)

        for s, e in zip(starts, ends):
            regime_idx = int(hidden_states[s])
            # guard: if index out of range, fallback to 'Sideways'
            if 0 <= regime_idx < len(regime_names):
                regime_name = regime_names[regime_idx]
            else:
                regime_name = 'Sideways'

            fill_color = color_map.get(regime_name, 'rgba(120,120,120,0.25)')

            fig_regime.add_vrect(
                x0=data.index[s], x1=data.index[e],
                fillcolor=fill_color, layer="below", line_width=0,
                annotation_text=regime_name if (e - s) <= 100 else None,  # optional label for short spans
                annotation_position="top left"
            )

    # Add invisible/dummy traces purely for legend (one per regime)
    for name in regime_names:
        fig_regime.add_trace(go.Scatter(
            x=[data.index[0] if len(data.index) else None],
            y=[0],
            mode='markers',
            marker=dict(size=10, color=marker_map.get(name, 'gray')),
            name=f"{name} Market",
            visible=True,
            hoverinfo='none'  # these are just legend entries
        ))

    fig_regime.update_layout(
        title='Market Regime Timeline',
        xaxis_title='Date',
        yaxis=dict(visible=False),  # timeline is background colored; hide numeric axis
        hovermode='x',
        height=400,
        showlegend=True,
        margin=dict(t=50, b=40, l=40, r=20)
    )
    
    # Intentionally omitted saving the interactive chart here to maintain legacy behavior. The visualizer will do what it needs.

    # -------------------------
    # Regime statistics
    # -------------------------
    total_periods = len(hidden_states)
    
    # Count periods in each regime by name
    regime_counts = {name: 0 for name in ['Bull', 'Bear', 'Sideways']}
    for i, name in enumerate(regime_names):
        mask = hidden_states == i
        regime_counts[name] = int(np.sum(mask))

    regime_stats = {
        'Total Periods': total_periods,
        'Bull Periods': regime_counts['Bull'],
        'Bear Periods': regime_counts['Bear'],
        'Sideways Periods': regime_counts['Sideways'],
        'Regime Changes': int(np.sum(np.diff(hidden_states) != 0)),
        'Average Duration': float(total_periods / (np.sum(np.diff(hidden_states) != 0) + 1))
    }

    # percentages
    for key in ['Bull', 'Bear', 'Sideways']:
        regime_stats[f'{key} %'] = f"{(regime_stats[f'{key} Periods'] / total_periods * 100):.1f}%" if total_periods > 0 else "0.0%"

    # -------------------------
    # Save regime_stats JSON
    # -------------------------
    with open(f'{output_dir}/regime_stats.json', 'w') as f:
        json.dump(regime_stats, f, indent=2)

    # -------------------------
    # Performance metrics (backtest results)
    # -------------------------
    if backtest_results:
        performance_metrics = {
            'Train Log-Likelihood': float(backtest_results.get('train_log_prob', np.nan)),
            'Test Log-Likelihood': float(backtest_results.get('test_log_prob', np.nan)),
            'Test AIC': float(backtest_results.get('test_aic', np.nan)),
            'Test BIC': float(backtest_results.get('test_bic', np.nan)),
            'Degradation %': float(backtest_results.get('degradation', np.nan)),
            'Average Regime Duration': float(backtest_results.get('avg_regime_duration', np.nan)),
            'Number of Regime Changes': int(backtest_results.get('n_regime_changes', 0)),
            'Decision': backtest_results.get('decision', '')
        }

        with open(f'{output_dir}/performance_metrics.json', 'w') as f:
            json.dump(performance_metrics, f, indent=2)

        # Generate performance summary CSV
        metrics_df = pd.DataFrame([performance_metrics]).T
        metrics_df.columns = ['Value']
        metrics_df.to_csv(f'{output_dir}/performance_summary.csv')

    # -------------------------
    # Return metadata
    # -------------------------
    return {
        'output_dir': output_dir,
        'regime_stats': regime_stats,
        'files_created': os.listdir(output_dir)
    }
