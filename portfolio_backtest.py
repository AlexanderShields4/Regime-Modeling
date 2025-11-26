import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class PortfolioMetrics:
    # Calculate comprehensive performance metrics for portfolio strategies

    def __init__(self, risk_free_rate=0.02):
        self.risk_free_rate = risk_free_rate  # Annual risk-free rate
        self.daily_rf_rate = (1 + risk_free_rate) ** (1/252) - 1  # Daily risk-free rate

    def calculate_returns(self, portfolio_values):
        # Calculate daily returns from portfolio values
        returns = portfolio_values.pct_change().dropna()
        return returns

    def calculate_total_return(self, portfolio_values):
        # Total return from start to end
        return (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1

    def calculate_cagr(self, portfolio_values):
        # Compound Annual Growth Rate
        total_return = self.calculate_total_return(portfolio_values)
        n_years = len(portfolio_values) / 252  # 252 trading days per year
        if n_years == 0:
            return 0.0
        cagr = (1 + total_return) ** (1 / n_years) - 1
        return cagr

    def calculate_volatility(self, returns, annualize=True):
        # Return volatility (standard deviation)
        vol = returns.std()
        if annualize:
            vol = vol * np.sqrt(252)  # Annualize daily volatility
        return vol

    def calculate_sharpe_ratio(self, returns):
        # Risk-adjusted return (Sharpe ratio)
        excess_returns = returns - self.daily_rf_rate
        if returns.std() == 0:
            return 0.0
        sharpe = excess_returns.mean() / returns.std() * np.sqrt(252)  # Annualized
        return sharpe

    def calculate_sortino_ratio(self, returns):
        # Downside risk-adjusted return (Sortino ratio)
        excess_returns = returns - self.daily_rf_rate
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(252)  # Annualized
        return sortino

    def calculate_drawdowns(self, portfolio_values):
        # Calculate drawdown series (running drawdown from peak)
        cumulative_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - cumulative_max) / cumulative_max
        return drawdowns

    def calculate_max_drawdown(self, portfolio_values):
        # Maximum peak-to-trough decline
        drawdowns = self.calculate_drawdowns(portfolio_values)
        return drawdowns.min()

    def calculate_average_drawdown(self, portfolio_values):
        # Average of all drawdown periods
        drawdowns = self.calculate_drawdowns(portfolio_values)
        return drawdowns[drawdowns < 0].mean() if len(drawdowns[drawdowns < 0]) > 0 else 0.0

    def calculate_recovery_time(self, portfolio_values):
        # Average days to recover from drawdown to new peak
        drawdowns = self.calculate_drawdowns(portfolio_values)
        recovery_times = []

        in_drawdown = False
        drawdown_start = None

        for i in range(len(drawdowns)):
            if drawdowns.iloc[i] < 0 and not in_drawdown:
                # Entering drawdown
                in_drawdown = True
                drawdown_start = i
            elif drawdowns.iloc[i] == 0 and in_drawdown:
                # Recovered to new peak
                recovery_times.append(i - drawdown_start)
                in_drawdown = False
                drawdown_start = None

        return np.mean(recovery_times) if recovery_times else 0.0

    def calculate_win_rate(self, returns):
        # Percentage of positive return periods
        positive_returns = returns[returns > 0]
        return len(positive_returns) / len(returns) if len(returns) > 0 else 0.0

    def calculate_calmar_ratio(self, portfolio_values):
        # CAGR / Max Drawdown (return per unit of max drawdown)
        cagr = self.calculate_cagr(portfolio_values)
        max_dd = abs(self.calculate_max_drawdown(portfolio_values))
        if max_dd == 0:
            return 0.0
        return cagr / max_dd

    def get_all_metrics(self, portfolio_values):
        # Calculate all metrics and return as dictionary
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
    # Main backtesting engine for comparing portfolio strategies

    def __init__(self, stock_data, bond_data, regime_predictions=None,
                 transaction_cost=0.001, initial_capital=100000):
        # Ensure data is aligned (same dates)
        self.stock_data = stock_data
        self.bond_data = bond_data

        # Align dates using inner join
        self.dates = stock_data.index.intersection(bond_data.index)
        self.stock_prices = stock_data.loc[self.dates]
        self.bond_prices = bond_data.loc[self.dates]

        self.regime_predictions = regime_predictions
        if regime_predictions is not None and len(regime_predictions) > len(self.dates):
            # Trim regime predictions to match data length
            self.regime_predictions = regime_predictions[:len(self.dates)]

        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital
        self.metrics_calculator = PortfolioMetrics()

    def get_rebalance_dates(self, frequency='monthly'):
        # Generate rebalancing schedule based on frequency
        dates = self.dates

        if frequency == 'regime_change':
            # Rebalance on each regime change
            if self.regime_predictions is None:
                raise ValueError("regime_predictions required for regime_change frequency")

            rebalance_indices = [0]  # Always start at beginning
            for i in range(1, len(self.regime_predictions)):
                if self.regime_predictions[i] != self.regime_predictions[i-1]:
                    rebalance_indices.append(i)

            return [dates[i] for i in rebalance_indices]

        elif frequency == 'monthly':
            # Rebalance first trading day of each month
            rebalance_dates = []
            current_month = None
            for date in dates:
                if current_month is None or date.month != current_month:
                    rebalance_dates.append(date)
                    current_month = date.month
            return rebalance_dates

        elif frequency == 'quarterly':
            # Rebalance first trading day of each quarter
            rebalance_dates = []
            current_quarter = None
            for date in dates:
                quarter = (date.month - 1) // 3
                if current_quarter is None or quarter != current_quarter:
                    rebalance_dates.append(date)
                    current_quarter = quarter
            return rebalance_dates

        elif frequency == 'yearly':
            # Rebalance first trading day of each year
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
        # Calculate transaction costs when rebalancing portfolio
        turnover = sum(abs(new_weights.get(asset, 0) - old_weights.get(asset, 0))
                      for asset in set(list(old_weights.keys()) + list(new_weights.keys())))

        cost = turnover * portfolio_value * self.transaction_cost
        return cost

    def calculate_portfolio_value(self, weight_schedule, rebalance_dates):
        # Core portfolio simulation engine
        # weight_schedule: dict of {date: {'stocks': weight, 'bonds': weight}}

        portfolio_values = pd.Series(index=self.dates, dtype=float)
        portfolio_values.iloc[0] = self.initial_capital

        # Track current holdings
        stock_shares = 0.0
        bond_shares = 0.0
        cash = self.initial_capital

        # Initial allocation
        first_date = self.dates[0]
        if first_date in weight_schedule:
            weights = weight_schedule[first_date]
            stock_value = self.initial_capital * weights['stocks']
            bond_value = self.initial_capital * weights['bonds']

            stock_shares = stock_value / self.stock_prices.iloc[0]
            bond_shares = bond_value / self.bond_prices.iloc[0]
            cash = 0.0

        # Simulate day by day
        for i in range(1, len(self.dates)):
            date = self.dates[i]

            # Calculate current portfolio value
            current_value = (stock_shares * self.stock_prices.iloc[i] +
                           bond_shares * self.bond_prices.iloc[i] +
                           cash)

            # Check if rebalancing needed
            if date in rebalance_dates:
                # Get target weights
                weights = weight_schedule[date]

                # Calculate old weights
                stock_value_old = stock_shares * self.stock_prices.iloc[i]
                bond_value_old = bond_shares * self.bond_prices.iloc[i]
                old_weights = {
                    'stocks': stock_value_old / current_value if current_value > 0 else 0,
                    'bonds': bond_value_old / current_value if current_value > 0 else 0
                }

                # Apply transaction costs
                transaction_cost = self.apply_transaction_costs(old_weights, weights, current_value)
                current_value -= transaction_cost

                # Rebalance to target weights
                target_stock_value = current_value * weights['stocks']
                target_bond_value = current_value * weights['bonds']

                stock_shares = target_stock_value / self.stock_prices.iloc[i]
                bond_shares = target_bond_value / self.bond_prices.iloc[i]
                cash = 0.0

            # Update portfolio value
            portfolio_values.iloc[i] = current_value

        return portfolio_values

    def run_buy_and_hold_strategy(self, stock_pct=0.6, bond_pct=0.4, rebalance_freq='yearly'):
        # 60/40 stock/bond benchmark with annual rebalancing
        rebalance_dates = self.get_rebalance_dates(rebalance_freq)

        # Create weight schedule (constant weights, rebalanced periodically)
        weight_schedule = {date: {'stocks': stock_pct, 'bonds': bond_pct}
                          for date in rebalance_dates}

        portfolio_values = self.calculate_portfolio_value(weight_schedule, rebalance_dates)
        return portfolio_values

    def run_bond_only_strategy(self):
        # 100% bonds baseline (no rebalancing needed)
        weight_schedule = {self.dates[0]: {'stocks': 0.0, 'bonds': 1.0}}
        rebalance_dates = [self.dates[0]]

        portfolio_values = self.calculate_portfolio_value(weight_schedule, rebalance_dates)
        return portfolio_values

    def run_regime_based_strategy(self, regime_allocations, rebalance_freq='regime_change'):
        # HMM-driven allocation based on regime predictions
        # regime_allocations: dict mapping regime state to weights
        # e.g., {0: {'stocks': 0.8, 'bonds': 0.2}, 1: {'stocks': 0.3, 'bonds': 0.7}, ...}

        if self.regime_predictions is None:
            raise ValueError("regime_predictions required for regime-based strategy")

        rebalance_dates = self.get_rebalance_dates(rebalance_freq)

        # Create weight schedule based on regime at each rebalance date
        weight_schedule = {}
        for date in rebalance_dates:
            date_idx = self.dates.get_loc(date)
            regime = self.regime_predictions[date_idx]
            weight_schedule[date] = regime_allocations[regime]

        portfolio_values = self.calculate_portfolio_value(weight_schedule, rebalance_dates)
        return portfolio_values

    def compare_all_strategies(self, regime_allocations):
        # Run all strategies and compare results

        results = {}
        metrics = {}

        # Strategy 1: Buy & Hold 60/40
        results['Buy & Hold 60/40'] = self.run_buy_and_hold_strategy(0.6, 0.4, 'yearly')
        metrics['Buy & Hold 60/40'] = self.metrics_calculator.get_all_metrics(results['Buy & Hold 60/40'])

        # Strategy 2: Bond Only
        results['Bond Only 100%'] = self.run_bond_only_strategy()
        metrics['Bond Only 100%'] = self.metrics_calculator.get_all_metrics(results['Bond Only 100%'])

        # Strategy 3: Regime-Based with different rebalancing frequencies
        rebalancing_frequencies = ['regime_change', 'monthly', 'quarterly']
        regime_strategies = {}

        for freq in rebalancing_frequencies:
            strategy_name = f'Regime-Based ({freq})'
            try:
                regime_strategies[freq] = self.run_regime_based_strategy(regime_allocations, freq)
                results[strategy_name] = regime_strategies[freq]
                metrics[strategy_name] = self.metrics_calculator.get_all_metrics(regime_strategies[freq])
            except Exception as e:
                print(f"Error running {strategy_name}: {e}")

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
