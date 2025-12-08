import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import os


class PortfolioVisualizer:
    # Create PNG visualizations for portfolio backtesting results using matplotlib

    def __init__(self, output_dir='dashboard_outputs/backtest_results/charts'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set default style
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = {
            'Buy & Hold 60/40': '#1f77b4',              # Blue
            'Bond Only 100%': '#2ca02c',                # Green
            'Buy Stock and Hold 100%': '#8c564b',       # Brown
            'Regime-Based (regime_change)': '#d62728',  # Red
            'Regime-Based (weekly)': '#17becf',         # Cyan
            'Regime-Based (monthly)': '#ff7f0e',        # Orange
            'Regime-Based (quarterly)': '#9467bd'       # Purple
        }

    def save_figure(self, fig, filename, dpi=300):
        # Helper to save matplotlib figure as high-quality PNG
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return filepath

    def plot_portfolio_values(self, strategies_dict, save_path='portfolio_value_comparison.png'):
        # Line chart showing portfolio values over time for all strategies (semi-log y-axis)
        fig, ax = plt.subplots(figsize=(12, 6))

        for strategy_name, portfolio_values in strategies_dict.items():
            color = self.colors.get(strategy_name, None)
            linewidth = 2.5 if 'Regime-Based' in strategy_name else 1.5

            ax.plot(portfolio_values.index, portfolio_values.values,
                   label=strategy_name, color=color, linewidth=linewidth, alpha=0.8)

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.set_title('Portfolio Value Comparison Over Time (Log Scale)', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(loc='upper left', fontsize=10) #ax to log st the y axis is semi scaled
        ax.grid(True, alpha=0.3, which='both')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)

        return self.save_figure(fig, save_path)

    def plot_cumulative_returns(self, strategies_dict, save_path='cumulative_returns.png'):
        # Line chart showing cumulative returns (%) from baseline
        fig, ax = plt.subplots(figsize=(12, 6))

        for strategy_name, portfolio_values in strategies_dict.items():
            # Calculate cumulative return from initial value
            cumulative_returns = (portfolio_values / portfolio_values.iloc[0] - 1) * 100
            color = self.colors.get(strategy_name, None)
            linewidth = 2.5 if 'Regime-Based' in strategy_name else 1.5

            ax.plot(cumulative_returns.index, cumulative_returns.values,
                   label=strategy_name, color=color, linewidth=linewidth, alpha=0.8)

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)

        return self.save_figure(fig, save_path)

    def plot_drawdowns(self, strategies_dict, save_path='drawdown_analysis.png'):
        # Underwater plot showing drawdowns from peak
        fig, ax = plt.subplots(figsize=(12, 6))

        for strategy_name, portfolio_values in strategies_dict.items():
            # Calculate drawdowns
            cumulative_max = portfolio_values.expanding().max()
            drawdowns = (portfolio_values - cumulative_max) / cumulative_max * 100

            color = self.colors.get(strategy_name, None)
            linewidth = 2.5 if 'Regime-Based' in strategy_name else 1.5

            ax.plot(drawdowns.index, drawdowns.values,
                   label=strategy_name, color=color, linewidth=linewidth, alpha=0.7)

        ax.fill_between(drawdowns.index, drawdowns.values, 0,
                        where=(drawdowns.values < 0), alpha=0.1, color='red')

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.set_title('Drawdown Analysis (Underwater Plot)', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)

        return self.save_figure(fig, save_path)

    def plot_sharpe_comparison(self, metrics_dict, save_path='sharpe_ratio_comparison.png'):
        # Horizontal bar chart comparing Sharpe ratios
        fig, ax = plt.subplots(figsize=(10, 6))

        strategies = list(metrics_dict.keys())
        sharpe_ratios = [metrics_dict[s]['Sharpe Ratio'] for s in strategies]

        # Color bars based on positive/negative
        colors = ['green' if sr > 0 else 'red' for sr in sharpe_ratios]

        y_pos = np.arange(len(strategies))
        ax.barh(y_pos, sharpe_ratios, color=colors, alpha=0.7, edgecolor='black')

        # Add value labels on bars
        for i, v in enumerate(sharpe_ratios):
            ax.text(v + 0.05 if v > 0 else v - 0.05, i, f'{v:.2f}',
                   va='center', ha='left' if v > 0 else 'right', fontsize=10)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(strategies, fontsize=10)
        ax.set_xlabel('Sharpe Ratio', fontsize=12)
        ax.set_title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

        return self.save_figure(fig, save_path)

    def plot_annual_returns(self, strategies_dict, save_path='annual_returns_by_strategy.png'):
        # Grouped bar chart showing annual returns for each strategy
        fig, ax = plt.subplots(figsize=(20, 6))

        # Calculate annual returns for each strategy
        annual_returns_data = {}

        for strategy_name, portfolio_values in strategies_dict.items():
            returns = portfolio_values.pct_change()
            # Group by year and calculate annual return
            annual_returns = returns.groupby(returns.index.year).apply(
                lambda x: (1 + x).prod() - 1
            ) * 100
            annual_returns_data[strategy_name] = annual_returns

        # Create DataFrame for easier plotting
        annual_df = pd.DataFrame(annual_returns_data)

        # Plot grouped bars
        years = annual_df.index.values
        x = np.arange(len(years))
        width = 0.07  # Width of bars

        for i, strategy_name in enumerate(annual_df.columns):
            offset = (i - len(annual_df.columns) / 2) * width
            color = self.colors.get(strategy_name, None)
            ax.bar(x + offset, annual_df[strategy_name].values,
                  width, label=strategy_name, color=color, alpha=0.8)

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Annual Return (%)', fontsize=12)
        ax.set_title('Annual Returns by Strategy', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(years, rotation=45)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

        return self.save_figure(fig, save_path)

    def plot_regime_timeline_portfolio(self, regime_predictions, stock_market_data,
                                      dates, save_path='regime_timeline_portfolio.png',
                                      regime_labels_map=None):
        # Dual-axis chart: regime background + stock market overlay
        fig, ax1 = plt.subplots(figsize=(14, 6))

        # Load actual regime labels from regime_allocations.csv if not provided
        if regime_labels_map is None:
            try:
                import pandas as pd
                regime_alloc = pd.read_csv('dashboard_outputs/backtest_results/data/regime_allocations.csv')
                # Create mapping from regime number to regime name
                regime_labels_map = regime_alloc.groupby('Regime')['Regime_Name'].first().to_dict()
            except:
                # Fallback to default mapping if file doesn't exist
                regime_labels_map = {0: 'Sideways', 1: 'Bull', 2: 'Bear'}

        # Assign colors based on regime NAME (not number)
        # This ensures colors match the actual regime characteristics
        name_to_color = {
            'Bull': 'lightgreen',      # Green for bull markets
            'Bear': 'lightcoral',      # Red for bear markets
            'Sideways': 'lightgray'    # Gray for sideways/choppy
        }

        # Create regime colors dict using actual labels
        regime_colors = {num: name_to_color.get(name, 'lightgray')
                        for num, name in regime_labels_map.items()}

        regime_names = regime_labels_map

        # Track which regimes we've added to legend
        added_to_legend = set()

        current_regime = regime_predictions[0]
        start_idx = 0

        for i in range(1, len(regime_predictions)):
            if regime_predictions[i] != current_regime or i == len(regime_predictions) - 1:
                # Plot shaded region for this regime
                end_idx = i if i < len(regime_predictions) - 1 else i

                # Get color and name, with fallback for unknown regimes
                color = regime_colors.get(current_regime, 'lightgray')
                name = regime_names.get(current_regime, f'Regime {current_regime}')

                # Only add label if this regime hasn't been added to legend yet
                label = name if current_regime not in added_to_legend else None
                if current_regime not in added_to_legend:
                    added_to_legend.add(current_regime)

                ax1.axvspan(dates[start_idx], dates[end_idx], alpha=0.3, color=color, label=label)

                current_regime = regime_predictions[i]
                start_idx = i

        # Plot stock market on primary axis
        ax1.plot(dates, stock_market_data.values, color='darkblue', linewidth=2.5, label='Stock Market')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Stock Market Price ($)', fontsize=12, color='darkblue')
        ax1.tick_params(axis='y', labelcolor='darkblue')
        ax1.grid(True, alpha=0.3)

        # Add regime change markers
        for i in range(1, len(regime_predictions)):
            if regime_predictions[i] != regime_predictions[i-1]:
                ax1.axvline(x=dates[i], color='black', linestyle='--', linewidth=0.5, alpha=0.5)

        ax1.set_title('Regime Timeline with Stock Market Performance', fontsize=14, fontweight='bold')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)

        # Add legend (remove duplicates)
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=10)

        return self.save_figure(fig, save_path)

    def save_all_visualizations(self, backtest_results, regime_predictions=None, dates=None, stock_data=None):
        # Generate all PNG charts from backtest results
        strategy_results = backtest_results['strategy_results']
        metrics = backtest_results['metrics']

        visualization_paths = []

        # Chart 1: Portfolio values
        path = self.plot_portfolio_values(strategy_results)
        visualization_paths.append(path)

        # Chart 2: Cumulative returns
        path = self.plot_cumulative_returns(strategy_results)
        visualization_paths.append(path)

        # Chart 3: Drawdowns
        path = self.plot_drawdowns(strategy_results)
        visualization_paths.append(path)

        # Chart 4: Sharpe ratio comparison
        path = self.plot_sharpe_comparison(metrics)
        visualization_paths.append(path)

        # Chart 5: Annual returns
        path = self.plot_annual_returns(strategy_results)
        visualization_paths.append(path)

        # Chart 6: Regime timeline (if regime data provided)
        if regime_predictions is not None and dates is not None and stock_data is not None:
            # Plot stock market performance against regime timeline
            path = self.plot_regime_timeline_portfolio(
                regime_predictions,
                stock_data,
                dates
            )
            visualization_paths.append(path)

        return visualization_paths
