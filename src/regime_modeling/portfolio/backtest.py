import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

from regime_modeling.config import (
    RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
    DEFAULT_TRANSACTION_COST,
    DEFAULT_INITIAL_CAPITAL,
)


class PortfolioMetrics:

    def __init__(self, risk_free_rate=RISK_FREE_RATE):
        self.risk_free_rate = risk_free_rate
        self.daily_rf_rate = (1 + risk_free_rate) ** (1/TRADING_DAYS_PER_YEAR) - 1

    def calculate_returns(self, portfolio_values):
        returns = portfolio_values.pct_change().dropna()
        return returns

    def calculate_total_return(self, portfolio_values):
        return (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1

    def calculate_cagr(self, portfolio_values):
        total_return = self.calculate_total_return(portfolio_values)
        n_years = len(portfolio_values) / TRADING_DAYS_PER_YEAR
        if n_years == 0:
            return 0.0
        cagr = (1 + total_return) ** (1 / n_years) - 1
        return cagr

    def calculate_volatility(self, returns, annualize=True):
        vol = returns.std()
        if annualize:
            vol = vol * np.sqrt(TRADING_DAYS_PER_YEAR)
        return vol

    def calculate_sharpe_ratio(self, returns):
        returns = returns.dropna()
        if len(returns) == 0:
            return float('nan')

        excess_returns = returns - self.daily_rf_rate

        vol = returns.std()
        if vol == 0 or np.isnan(vol):
            return float('nan')

        sharpe = (excess_returns.mean() / vol) * np.sqrt(TRADING_DAYS_PER_YEAR)
        return float(sharpe)

    def calculate_sortino_ratio(self, returns):
        excess_returns = returns - self.daily_rf_rate
        # Frank Sortino original formula: RMS of negative excess returns (semi-deviation)
        downside = np.minimum(excess_returns, 0)
        downside_dev = np.sqrt(np.mean(downside ** 2)) * np.sqrt(TRADING_DAYS_PER_YEAR)
        if downside_dev == 0:
            return 0.0
        return (excess_returns.mean() * TRADING_DAYS_PER_YEAR) / downside_dev

    def calculate_drawdowns(self, portfolio_values):
        cumulative_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - cumulative_max) / cumulative_max
        return drawdowns

    def calculate_max_drawdown(self, portfolio_values):
        drawdowns = self.calculate_drawdowns(portfolio_values)
        return drawdowns.min()

    def calculate_average_drawdown(self, portfolio_values):
        drawdowns = self.calculate_drawdowns(portfolio_values)
        return drawdowns[drawdowns < 0].mean() if len(drawdowns[drawdowns < 0]) > 0 else 0.0

    def calculate_recovery_time(self, portfolio_values):
        drawdowns = self.calculate_drawdowns(portfolio_values)
        recovery_times = []

        in_drawdown = False
        drawdown_start = None

        for i in range(len(drawdowns)):
            if drawdowns.iloc[i] < 0 and not in_drawdown:
                in_drawdown = True
                drawdown_start = i
            elif drawdowns.iloc[i] >= -1e-10 and in_drawdown:
                recovery_times.append(i - drawdown_start)
                in_drawdown = False
                drawdown_start = None

        return np.mean(recovery_times) if recovery_times else 0.0

    def calculate_win_rate(self, returns):
        positive_returns = returns[returns > 0]
        return len(positive_returns) / len(returns) if len(returns) > 0 else 0.0

    def calculate_calmar_ratio(self, portfolio_values):
        cagr = self.calculate_cagr(portfolio_values)
        max_dd = abs(self.calculate_max_drawdown(portfolio_values))
        if max_dd == 0:
            return 0.0
        return cagr / max_dd

    def get_all_metrics(self, portfolio_values):
        returns = self.calculate_returns(portfolio_values)

        metrics = {
            'Total Return': self.calculate_total_return(portfolio_values),
            'CAGR': self.calculate_cagr(portfolio_values),
            'Volatility': self.calculate_volatility(returns),
            'Sharpe Ratio': self.calculate_sharpe_ratio(returns),
            'Sortino Ratio': self.calculate_sortino_ratio(returns),
            'Max Drawdown': self.calculate_max_drawdown(portfolio_values),
            'Average Drawdown': self.calculate_average_drawdown(portfolio_values),
            'Recovery Time': self.calculate_recovery_time(portfolio_values),
            'Win Rate': self.calculate_win_rate(returns),
            'Calmar Ratio': self.calculate_calmar_ratio(portfolio_values)
        }

        return metrics


class PortfolioBacktester:

    def __init__(self, stock_data, bond_data, regime_predictions=None,
                 transaction_cost=DEFAULT_TRANSACTION_COST, initial_capital=DEFAULT_INITIAL_CAPITAL):
        self.stock_data = stock_data
        self.bond_data = bond_data

        self.dates = stock_data.index.intersection(bond_data.index)
        self.stock_prices = stock_data.loc[self.dates]
        self.bond_prices = bond_data.loc[self.dates]

        self.regime_predictions = regime_predictions
        if regime_predictions is not None and len(regime_predictions) > len(self.dates):
            self.regime_predictions = regime_predictions[:len(self.dates)]

        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital
        self.metrics_calculator = PortfolioMetrics()

    def get_rebalance_dates(self, frequency):
        dates = self.dates

        if frequency == 'regime_change':
            if self.regime_predictions is None:
                raise ValueError("regime_predictions required for regime_change frequency")

            rebalance_indices = [0]
            for i in range(1, len(self.regime_predictions)):
                if self.regime_predictions[i] != self.regime_predictions[i-1]:
                    rebalance_indices.append(i)

            return [dates[i] for i in rebalance_indices]

        elif frequency == 'monthly':
            rebalance_dates = []
            current_month = None
            for date in dates:
                if current_month is None or date.month != current_month:
                    rebalance_dates.append(date)
                    current_month = date.month
            return rebalance_dates

        elif frequency == 'weekly':
            rebalance_dates = []
            current_week = None
            for date in dates:
                week = date.isocalendar()[1]
                year_week = (date.year, week)
                if current_week is None or year_week != current_week:
                    rebalance_dates.append(date)
                    current_week = year_week
            return rebalance_dates

        elif frequency == 'quarterly':
            rebalance_dates = []
            current_quarter = None
            for date in dates:
                quarter = (date.month - 1) // 3
                if current_quarter is None or quarter != current_quarter:
                    rebalance_dates.append(date)
                    current_quarter = quarter
            return rebalance_dates

        elif frequency == 'yearly':
            rebalance_dates = []
            current_year = None
            for date in dates:
                if current_year is None or date.year != current_year:
                    rebalance_dates.append(date)
                    current_year = date.year
            return rebalance_dates

        else:
            raise ValueError(f"Unknown frequency: {frequency}")

    def apply_transaction_costs(self, old_weights, new_weights, portfolio_value):
        assets_to_charge = ['stocks', 'bonds']
        turnover = sum(abs(new_weights.get(asset, 0) - old_weights.get(asset, 0))
                      for asset in assets_to_charge)

        cost = turnover * portfolio_value * self.transaction_cost
        return cost

    def calculate_portfolio_value(self, weight_schedule, rebalance_dates):
        portfolio_values = pd.Series(index=self.dates, dtype=float)
        portfolio_values.iloc[0] = self.initial_capital

        stock_shares = 0.0
        bond_shares = 0.0
        cash = self.initial_capital
        first_date = self.dates[0]
        if first_date in weight_schedule:
            weights = weight_schedule[first_date]
            stock_value = self.initial_capital * weights.get('stocks', 0.0)
            bond_value = self.initial_capital * weights.get('bonds', 0.0)
            cash = self.initial_capital * weights.get('cash', 0.0)

            stock_shares = stock_value / self.stock_prices.iloc[0] if stock_value > 0 else 0.0
            bond_shares = bond_value / self.bond_prices.iloc[0] if bond_value > 0 else 0.0

        for i in range(1, len(self.dates)):
            date = self.dates[i]

            current_value = (stock_shares * self.stock_prices.iloc[i] +
                           bond_shares * self.bond_prices.iloc[i] +
                           cash)

            if date in rebalance_dates:
                weights = weight_schedule[date]

                stock_value_old = stock_shares * self.stock_prices.iloc[i]
                bond_value_old = bond_shares * self.bond_prices.iloc[i]
                cash_old = cash
                old_weights = {
                    'stocks': stock_value_old / current_value if current_value > 0 else 0,
                    'bonds': bond_value_old / current_value if current_value > 0 else 0,
                    'cash': cash_old / current_value if current_value > 0 else 0
                }

                transaction_cost = self.apply_transaction_costs(old_weights, weights, current_value)
                current_value -= transaction_cost

                target_stock_value = current_value * weights.get('stocks', 0.0)
                target_bond_value = current_value * weights.get('bonds', 0.0)
                cash = current_value * weights.get('cash', 0.0)

                stock_shares = target_stock_value / self.stock_prices.iloc[i] if target_stock_value > 0 else 0.0
                bond_shares = target_bond_value / self.bond_prices.iloc[i] if target_bond_value > 0 else 0.0

            portfolio_values.iloc[i] = current_value

        return portfolio_values

    def run_buy_stock_and_hold_strategy(self, stock_pct = 1.0, rebalance_freq='yearly'):
        rebalance_dates = self.get_rebalance_dates(rebalance_freq)
        weight_schedule = {date: {'stocks': stock_pct, 'bonds': 0.0}
                          for date in rebalance_dates}
        portfolio_values = self.calculate_portfolio_value(weight_schedule, rebalance_dates)
        return portfolio_values

    def run_buy_and_hold_strategy(self, stock_pct=0.6, bond_pct=0.4, rebalance_freq='yearly'):
        rebalance_dates = self.get_rebalance_dates(rebalance_freq)

        weight_schedule = {date: {'stocks': stock_pct, 'bonds': bond_pct}
                          for date in rebalance_dates}

        portfolio_values = self.calculate_portfolio_value(weight_schedule, rebalance_dates)
        return portfolio_values

    def run_bond_only_strategy(self):
        weight_schedule = {self.dates[0]: {'stocks': 0.0, 'bonds': 1.0}}
        rebalance_dates = [self.dates[0]]

        portfolio_values = self.calculate_portfolio_value(weight_schedule, rebalance_dates)
        return portfolio_values

    def get_weighted_average_regime(self, end_idx, lookback_days):
        start_idx = max(0, end_idx - lookback_days)
        regime_slice = self.regime_predictions[start_idx:end_idx + 1]

        if len(regime_slice) == 0:
            return int(self.regime_predictions[end_idx])

        unique_regimes, counts = np.unique(regime_slice, return_counts=True)

        most_common_idx = np.argmax(counts)
        return int(unique_regimes[most_common_idx])

    def get_weighted_regime_allocation(self, end_idx, lookback_days, regime_allocations):
        start_idx = max(0, end_idx - lookback_days)
        regime_slice = self.regime_predictions[start_idx:end_idx + 1]

        if len(regime_slice) == 0:
            regime = int(self.regime_predictions[end_idx])
            return regime_allocations[regime]

        unique_regimes, counts = np.unique(regime_slice, return_counts=True)
        weights = counts / len(regime_slice)

        weighted_allocation = {'stocks': 0.0, 'bonds': 0.0, 'cash': 0.0}

        for regime, weight in zip(unique_regimes, weights):
            regime_alloc = regime_allocations[int(regime)]
            for asset in weighted_allocation.keys():
                weighted_allocation[asset] += regime_alloc.get(asset, 0.0) * weight

        return weighted_allocation

    def run_regime_based_strategy(self, regime_allocations, rebalance_freq='regime_change'):
        if self.regime_predictions is None:
            raise ValueError("regime_predictions required for regime-based strategy")

        rebalance_dates = self.get_rebalance_dates(rebalance_freq)

        weight_schedule = {}

        lookback_map = {
            'weekly': 5,
            'monthly': 21,
            'quarterly': 63
        }

        for date in rebalance_dates:
            date_idx = self.dates.get_loc(date)

            if rebalance_freq in lookback_map:
                lookback_days = lookback_map[rebalance_freq]
                weight_schedule[date] = self.get_weighted_regime_allocation(
                    date_idx, lookback_days, regime_allocations
                )
            else:
                regime = int(self.regime_predictions[date_idx])
                weight_schedule[date] = regime_allocations[regime]

        portfolio_values = self.calculate_portfolio_value(weight_schedule, rebalance_dates)
        return portfolio_values

    def compare_all_strategies(self, regime_allocations):
        results = {}
        metrics = {}
        results['Buy & Hold 60/40'] = self.run_buy_and_hold_strategy(0.6, 0.4, 'yearly')
        metrics['Buy & Hold 60/40'] = self.metrics_calculator.get_all_metrics(results['Buy & Hold 60/40'])

        # Strategy 2: Bond Only
        results['Bond Only 100%'] = self.run_bond_only_strategy()
        metrics['Bond Only 100%'] = self.metrics_calculator.get_all_metrics(results['Bond Only 100%'])

        # Strategy 3: Buy Stock and Hold
        results['Buy Stock and Hold 100%'] = self.run_buy_stock_and_hold_strategy(1.0, 'yearly')
        metrics['Buy Stock and Hold 100%'] = self.metrics_calculator.get_all_metrics(results['Buy Stock and Hold 100%'])

        # Strategy 3: Regime-Based with different rebalancing frequencies
        rebalancing_frequencies = ['regime_change', 'weekly', 'monthly', 'quarterly']
        regime_strategies = {}

        for freq in rebalancing_frequencies:
            strategy_name = f'Regime-Based ({freq})'
            try:
                regime_strategies[freq] = self.run_regime_based_strategy(regime_allocations, freq)
                results[strategy_name] = regime_strategies[freq]
                metrics[strategy_name] = self.metrics_calculator.get_all_metrics(regime_strategies[freq])
            except Exception as e:
                logger.error(f"Error running {strategy_name}: {e}")

        # Select best rebalancing frequency based on Sharpe ratio
        best_freq = None
        best_sharpe = -np.inf
        for freq in rebalancing_frequencies:
            strategy_name = f'Regime-Based ({freq})'
            if strategy_name in metrics:
                sharpe = metrics[strategy_name]['Sharpe Ratio']
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_freq = freq

        return {
            'strategy_results': results,
            'metrics': metrics,
            'best_rebalance_freq': best_freq,
            'best_sharpe': best_sharpe
        }


def run_portfolio_backtest(
    test_states: np.ndarray,
    test_data: pd.DataFrame,
    generate_visualizations: bool = True,
    output_dir: str = None
) -> dict:
    """
    Run portfolio backtest on test period data.

    Fetches SPY (stock proxy) and TLT (bond proxy) for the test date range,
    then runs all strategy comparisons via PortfolioBacktester.

    Returns:
        dict with keys: metrics, best_rebalance_freq, strategy_results, best_sharpe,
        and optionally output_dir if one was provided.
    """
    from regime_modeling.data.data_utils import get_individual_stocks, get_bond_data
    from regime_modeling.config import REGIME_ALLOCATIONS, N_REGIMES
    from regime_modeling.models.regime import analyze_regime_characteristics

    _logger = logging.getLogger(__name__)

    start_date = str(test_data.index[0].date())
    end_date = str((test_data.index[-1] + pd.Timedelta(days=5)).date())

    # Fetch SPY as equity market proxy
    stock_df = get_individual_stocks(['SPY'], start=start_date, end=end_date)
    if stock_df.empty or 'SPY' not in stock_df.columns:
        raise ValueError("Could not fetch SPY data for the backtest period")
    stock_series = stock_df['SPY'].dropna()

    # Fetch bond data (TLT with fallbacks)
    bond_series = get_bond_data(start=start_date, end=end_date)
    if isinstance(bond_series, pd.DataFrame):
        if bond_series.empty:
            raise ValueError("Could not fetch bond data for the backtest period")
        bond_series = bond_series.iloc[:, 0]
    bond_series = bond_series.dropna()

    # Normalize timezone info so intersection works regardless of yfinance tz behaviour
    def _strip_tz(idx):
        if hasattr(idx, 'tz') and idx.tz is not None:
            return idx.tz_localize(None)
        return idx

    stock_series = stock_series.copy()
    stock_series.index = _strip_tz(stock_series.index)
    bond_series = bond_series.copy()
    bond_series.index = _strip_tz(bond_series.index)
    test_index = _strip_tz(test_data.index)

    aligned_dates = test_index.intersection(stock_series.index).intersection(bond_series.index)
    if len(aligned_dates) == 0:
        raise ValueError("No overlapping dates between test_data, SPY, and bond data")

    # Map test_states to aligned dates only
    test_date_positions = [test_index.get_loc(d) for d in aligned_dates]
    regime_predictions = test_states[test_date_positions]

    stock_aligned = stock_series.loc[aligned_dates]
    bond_aligned = bond_series.loc[aligned_dates]

    # Build integer-indexed allocations from string-keyed REGIME_ALLOCATIONS
    regime_types = analyze_regime_characteristics(test_states, test_data, N_REGIMES)
    int_allocations = {
        i: REGIME_ALLOCATIONS.get(regime_types[i], REGIME_ALLOCATIONS['Unknown'])
        for i in range(len(regime_types))
    }

    backtester = PortfolioBacktester(
        stock_data=stock_aligned,
        bond_data=bond_aligned,
        regime_predictions=regime_predictions,
    )
    comparison = backtester.compare_all_strategies(int_allocations)

    result = {
        'strategy_results': comparison['strategy_results'],
        'metrics': comparison['metrics'],
        'best_rebalance_freq': comparison['best_rebalance_freq'],
        'best_sharpe': comparison['best_sharpe'],
    }

    if output_dir is not None:
        result['output_dir'] = output_dir
        charts_dir = os.path.join(output_dir, 'charts')
        os.makedirs(charts_dir, exist_ok=True)

        if generate_visualizations:
            try:
                from regime_modeling.portfolio.visualizer import PortfolioVisualizer
                viz = PortfolioVisualizer(output_dir=charts_dir)
                viz.save_all_visualizations(comparison)
            except Exception as e:
                _logger.warning(f"Visualization generation failed: {e}")

    return result


def calculate_risk_metrics(
    test_data: pd.DataFrame,
    test_states: np.ndarray,
    regime_types: list = None
) -> dict:
    """
    Compute per-regime risk statistics from the test feature data.

    Returns a JSON-serializable dict keyed by regime name with annualised
    return, volatility, Sharpe, worst/best day, and period count.
    """
    from regime_modeling.models.regime import analyze_regime_characteristics, _extract_return_columns
    from regime_modeling.config import N_REGIMES, TRADING_DAYS_PER_YEAR

    if regime_types is None:
        regime_types = analyze_regime_characteristics(test_states, test_data, N_REGIMES)

    return_cols = _extract_return_columns(test_data)
    if not return_cols:
        return {}

    risk_metrics = {}
    for i, regime_name in enumerate(regime_types):
        mask = test_states == i
        if not mask.any():
            continue

        regime_returns = test_data.loc[mask, return_cols].mean(axis=1)
        mean_ret = float(regime_returns.mean())
        std_ret = float(regime_returns.std())

        risk_metrics[regime_name] = {
            'count': int(mask.sum()),
            'mean_daily_return': round(mean_ret, 6),
            'annualized_return': round(mean_ret * TRADING_DAYS_PER_YEAR, 4),
            'daily_volatility': round(std_ret, 6),
            'annualized_volatility': round(std_ret * np.sqrt(TRADING_DAYS_PER_YEAR), 4),
            'worst_day': round(float(regime_returns.min()), 6),
            'best_day': round(float(regime_returns.max()), 6),
            'sharpe': round(mean_ret / std_ret * np.sqrt(TRADING_DAYS_PER_YEAR), 4) if std_ret > 0 else 0.0,
        }

    return risk_metrics
