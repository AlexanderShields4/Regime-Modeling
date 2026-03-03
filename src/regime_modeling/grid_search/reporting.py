import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from regime_modeling.config import DECISION_THRESHOLDS, BEST_CONFIG_FILE
from regime_modeling.models.persistence import save_best_model

logger = logging.getLogger(__name__)


def generate_grid_search_report(results_list: List[Dict], top_n: int = 10) -> Optional[pd.DataFrame]:
    """
    Process raw results from grid search into a polished DataFrame,
    save best models, output CSV logs, and print reports.
    """
    if not results_list:
        return None
        
    summary_data = []
    for r in results_list:
        p = r['params']
        features = []
        if p['include_returns']: features.append('Ret')
        if p['include_volatility']: features.append('Vol')
        if p['include_rsi']: features.append('RSI')
        if p['include_momentum']: features.append('Mom')
        if p['include_market_breadth']: features.append('MB')
        feature_str = '+'.join(features)

        sharpe = r.get('portfolio_sharpe', 0)
        max_dd = r.get('portfolio_max_dd', -0.5)
        cagr = r.get('portfolio_cagr', 0)
        degradation = abs(r['degradation'])

        DT = DECISION_THRESHOLDS
        if sharpe > DT['excellent_sharpe'] and max_dd > DT['excellent_max_dd'] and cagr > DT['excellent_cagr'] and degradation < DT['excellent_degradation']:
            decision = '✅ EXCELLENT'
        elif sharpe > DT['good_sharpe'] and max_dd > DT['good_max_dd'] and cagr > DT['good_cagr'] and degradation < DT['good_degradation']:
            decision = '✓ GOOD'
        elif sharpe > DT['acceptable_sharpe'] and max_dd > DT['acceptable_max_dd'] and cagr > DT['acceptable_cagr']:
            decision = '○ ACCEPTABLE'
        else:
            decision = '✗ POOR'

        summary_data.append({
            'Config_ID': r['config_id'],
            'Score': r['score'],
            'CAGR%': r.get('portfolio_cagr', 0) * 100,
            'Sharpe': r.get('portfolio_sharpe', 0),
            'Sortino': r.get('portfolio_sortino', 0),
            'MaxDD%': r.get('portfolio_max_dd', -0.5) * 100,
            'Calmar': r.get('portfolio_calmar', 0),
            'Volatility%': r.get('portfolio_volatility', 0.3) * 100,
            'WinRate%': r.get('portfolio_win_rate', 0.5) * 100,
            'BestRebalance': r.get('best_rebalance_freq', 'unknown'),
            'Regime_Diversity': r.get('regime_diversity', 0),
            'N_Regimes_Used': r.get('n_unique_regimes', 0),
            'Regime_Changes': r['n_regime_changes'],
            'Avg_Duration': r['avg_regime_duration'],
            'Degradation%': r['degradation'],
            'Test_LogL': r['test_log_prob'],
            'Test_BIC': r['test_bic'],
            'n_stocks': p['n_stocks'],
            'n_indices': p['n_indices'],
            'vol_window': p['volatility_window'],
            'Features': feature_str,
            'Regimes': '|'.join(r['regime_types']),
            'Decision': decision
        })

    results_df = pd.DataFrame(summary_data)
    results_df = results_df.sort_values('Score')
    logger.info("\n" + "="*70)
    logger.info(f"TOP {top_n} CONFIGURATIONS (Lower Score = Better)")
    logger.info("="*70)
    logger.info(results_df.head(top_n).to_string(index=False))

    # Save full results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'grid_search_results_{timestamp}.csv'
    results_df.to_csv(csv_filename, index=False)
    logger.info(f"\n✓ Full results saved to: {csv_filename}")

    best = results_df.iloc[0]
    logger.info("\n" + "="*70)
    logger.info("BEST CONFIGURATION DETAILS")
    logger.info("="*70)
    logger.info(f"Config ID: {best['Config_ID']}")
    logger.info(f"Composite Score: {best['Score']:.2f} (lower is better)")
    logger.info(f"\nPORTFOLIO PERFORMANCE:")
    logger.info(f"  CAGR: {best['CAGR%']:.2f}%")
    logger.info(f"  Sharpe Ratio: {best['Sharpe']:.2f}")
    logger.info(f"  Sortino Ratio: {best['Sortino']:.2f}")
    logger.info(f"  Max Drawdown: {best['MaxDD%']:.2f}%")
    logger.info(f"  Calmar Ratio: {best['Calmar']:.2f}")
    logger.info(f"  Volatility: {best['Volatility%']:.2f}%")
    logger.info(f"  Win Rate: {best['WinRate%']:.1f}%")
    logger.info(f"  Best Rebalance Freq: {best['BestRebalance']}")
    logger.info(f"\nPARAMETERS:")
    logger.info(f"  n_stocks: {best['n_stocks']}")
    logger.info(f"  n_indices: {best['n_indices']}")
    logger.info(f"  volatility_window: {best['vol_window']}")
    logger.info(f"  Features: {best['Features']}")
    logger.info(f"\nREGIME QUALITY:")
    logger.info(f"  Regimes Used: {best['N_Regimes_Used']}/3")
    logger.info(f"  Regime Diversity: {best['Regime_Diversity']:.3f} (1.0 = perfectly balanced)")
    logger.info(f"  Regime Changes: {best['Regime_Changes']}")
    logger.info(f"  Avg Regime Duration: {best['Avg_Duration']:.1f} periods")
    logger.info(f"  Regime Types: {best['Regimes']}")
    logger.info(f"\nSTATISTICAL QUALITY:")
    logger.info(f"  Degradation: {best['Degradation%']:.1f}%")
    logger.info(f"  Test Log-Likelihood: {best['Test_LogL']:.0f}")
    logger.info(f"  Test BIC: {best['Test_BIC']:.0f}")
    logger.info(f"\nOVERALL DECISION: {best['Decision']}")
    logger.info("="*70 + "\n")

    best_config_file = BEST_CONFIG_FILE
    best_params = [r for r in results_list if r['config_id'] == best['Config_ID']][0]['params']

    with open(best_config_file, 'w') as f:
        f.write("# Best HMM Configuration (from grid search)\n")
        f.write(f"# Generated: {timestamp}\n\n")
        f.write("run_hmm_model(\n")
        f.write(f"    n_stocks={best_params['n_stocks']},\n")
        f.write(f"    n_indices={best_params['n_indices']},\n")
        f.write(f"    volatility_window={best_params['volatility_window']},\n")
        f.write(f"    rsi_period={best_params['rsi_period']},\n")
        f.write(f"    momentum_period={best_params['momentum_period']},\n")
        f.write(f"    include_returns={best_params['include_returns']},\n")
        f.write(f"    include_volatility={best_params['include_volatility']},\n")
        f.write(f"    include_rsi={best_params['include_rsi']},\n")
        f.write(f"    include_momentum={best_params['include_momentum']},\n")
        f.write(f"    include_market_breadth={best_params['include_market_breadth']},\n")
        f.write(f"    n_iter={best_params['n_iter']},\n")
        f.write(f"    covariance_type='{best_params['covariance_type']}',\n")
        f.write(f"    random_state={best_params['random_state']},\n")
        f.write(f"    backtest=True\n")
        f.write(")\n")

    logger.info(f"✓ Best configuration saved to: {best_config_file}")

    best_result = [r for r in results_list if r['config_id'] == best['Config_ID']][0]

    if best_result.get('model') is not None and best_result.get('scaler') is not None:
        logger.info("\nSaving best model...")

        best_metrics = {
            'score': best_result['score'],
            'degradation': best_result['degradation'],
            'train_log_prob': best_result['train_log_prob'],
            'test_log_prob': best_result['test_log_prob'],
            'test_aic': best_result['test_aic'],
            'test_bic': best_result['test_bic'],
            'avg_regime_duration': best_result['avg_regime_duration'],
            'n_regime_changes': best_result['n_regime_changes'],
            'regime_types': best_result['regime_types'],
            'decision': best_result['decision']
        }

        saved_paths = save_best_model(
            model=best_result['model'],
            scaler=best_result['scaler'],
            params=best_result['params'],
            metrics=best_metrics
        )

        logger.info(f"✓ Best model saved to: {saved_paths['model_path']}")
        logger.info(f"✓ Latest model link: {saved_paths['latest_path']}")
        logger.info("\nTo load the model later:")
        logger.info("  from model import load_model")
        logger.info("  data = load_model()")
        logger.info("  model = data['model']")
        logger.info("  scaler = data['scaler']\n")
    else:
        logger.info("    To train the best config, re-run with the saved parameters in best_config.txt\n")

    logger.info("\n" + "="*70)
    logger.info("PORTFOLIO PERFORMANCE SUMMARY")
    logger.info("="*70)
    logger.info(f"\nBest Sharpe Ratio: {results_df['Sharpe'].max():.2f} (Config {results_df.loc[results_df['Sharpe'].idxmax(), 'Config_ID']})")
    logger.info(f"Best CAGR: {results_df['CAGR%'].max():.1f}% (Config {results_df.loc[results_df['CAGR%'].idxmax(), 'Config_ID']})")
    logger.info(f"Best Max Drawdown: {results_df['MaxDD%'].max():.1f}% (Config {results_df.loc[results_df['MaxDD%'].idxmax(), 'Config_ID']})")
    logger.info(f"Best Calmar Ratio: {results_df['Calmar'].max():.2f} (Config {results_df.loc[results_df['Calmar'].idxmax(), 'Config_ID']})")

    logger.info(f"\n Configuration Quality Distribution:")
    decision_counts = results_df['Decision'].value_counts()
    for decision in ['✅ EXCELLENT', '✓ GOOD', '○ ACCEPTABLE', '✗ POOR']:
        count = decision_counts.get(decision, 0)
        pct = count / len(results_df) * 100
        logger.info(f"  {decision}: {count}/{len(results_df)} ({pct:.1f}%)")

    excellent_count = (results_df['Decision'] == '✅ EXCELLENT').sum()
    good_count = (results_df['Decision'] == '✓ GOOD').sum()
    ready_count = excellent_count + good_count

    logger.info(f"\nConfigs ready for live trading: {ready_count}/{len(results_df)} ({ready_count/len(results_df)*100:.1f}%)")
    logger.info(f"  (Sharpe > 1.0, MaxDD > -30%, CAGR > 8%, Degradation < 15%)")

    logger.info("\n" + "="*70)
    logger.info("SCORING BREAKDOWN (for top config)")
    logger.info("="*70)
    # best_result variable reused from above
    logger.info(f"Composite Score: {best_result['score']:.2f}")
    logger.info(f"  Portfolio Component: {best_result.get('portfolio_score_component', 0):.2f} (normalized × 600 = {best_result.get('portfolio_score_component', 0) / 250 * 600:.2f})")
    logger.info(f"  Statistical Component: {best_result.get('statistical_score_component', 0):.2f} (normalized × 400 = {best_result.get('statistical_score_component', 0) / 150 * 400:.2f})")
    logger.info(f"\nScore Interpretation:")
    logger.info(f"  Lower score = better overall quality")
    logger.info(f"  60% weight on portfolio performance (returns, risk, Sharpe)")
    logger.info(f"  40% weight on statistical quality (BIC, degradation, regime diversity)")
    logger.info("="*70 + "\n")

    return results_df

def grid_search_parameters(
    n_stocks_range=None,
    n_indices_range=None,
    volatility_window_range=None,
    rsi_period_range=None,
    momentum_period_range=None,
    n_iter_range=None,
    covariance_type_range=None,
    random_state=None,  # defaulted in runner
    feature_combinations='auto',
    train_ratio=None,   # defaulted in runner
    n_processes=None,
    top_n=10,
    show_progress=False
) -> Optional[pd.DataFrame]:
    """
    Convenience wrapper that runs parallel grid search and then generates the report.
    This replaces the monolithic behavior from the legacy file.
    """
    from regime_modeling.grid_search.runner import run_grid_search
    from regime_modeling.config import DEFAULT_HMM_PARAMS
    
    if random_state is None:
        random_state = DEFAULT_HMM_PARAMS['random_state']
    if train_ratio is None:
        train_ratio = DEFAULT_HMM_PARAMS['train_ratio']

    results_list = run_grid_search(
        n_stocks_range=n_stocks_range,
        n_indices_range=n_indices_range,
        volatility_window_range=volatility_window_range,
        rsi_period_range=rsi_period_range,
        momentum_period_range=momentum_period_range,
        n_iter_range=n_iter_range,
        covariance_type_range=covariance_type_range,
        random_state=random_state,
        feature_combinations=feature_combinations,
        train_ratio=train_ratio,
        n_processes=n_processes,
        show_progress=show_progress
    )

    return generate_grid_search_report(results_list, top_n)
