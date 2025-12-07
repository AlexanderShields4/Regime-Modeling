"""
Inspect and analyze what the HMM model has learned about market regimes.
Generates a detailed report showing regime characteristics, patterns, and examples.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def analyze_regimes():
    print("\n" + "="*80)
    print("REGIME ANALYSIS REPORT")
    print("="*80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load data
    print("Loading data...")
    df = pd.read_csv('merged_data_with_features.csv', index_col=0, parse_dates=True)
    regime_df = pd.read_csv('dashboard_outputs/backtest_results/data/regime_allocations.csv')
    regime_df['Date'] = pd.to_datetime(regime_df['Date'])
    regime_df_indexed = regime_df.set_index('Date')

    # Extract feature columns
    valid_prefixes = ('stock_', 'index_', 'resource_')
    exclude_patterns = ['_ma_', '_vol', 'momentum', '_mom', 'rsi', '_rsi', '_ema', '_sma', '_ewma']

    return_cols = [col for col in df.columns
                   if col.lower().startswith(valid_prefixes)
                   and not any(pattern in col.lower() for pattern in exclude_patterns)]

    vol_cols = [col for col in df.columns if '_vol' in col.lower()]
    momentum_cols = [col for col in df.columns if 'momentum' in col.lower()]

    # Align dates
    common_dates = df.index.intersection(regime_df_indexed.index)
    df_aligned = df.loc[common_dates]
    regime_aligned = regime_df_indexed.loc[common_dates]

    print(f"Analyzing {len(common_dates)} trading days from {common_dates[0].date()} to {common_dates[-1].date()}\n")

    # Overall statistics
    print("="*80)
    print("REGIME DISTRIBUTION")
    print("="*80)
    regime_counts = regime_aligned['Regime_Name'].value_counts()
    for regime_name, count in regime_counts.items():
        pct = count / len(regime_aligned) * 100
        print(f"{regime_name:10s}: {count:4d} days ({pct:5.1f}%)")

    # Detailed regime analysis
    print("\n" + "="*80)
    print("REGIME CHARACTERISTICS")
    print("="*80)

    regime_stats = {}

    for regime_name in ['Bull', 'Sideways', 'Bear']:
        mask = regime_aligned['Regime_Name'] == regime_name
        regime_data = df_aligned[mask]
        regime_num = regime_aligned[mask]['Regime'].iloc[0] if len(regime_aligned[mask]) > 0 else None

        if len(regime_data) == 0:
            continue

        # Calculate returns
        daily_returns = regime_data[return_cols].mean(axis=1)
        mean_return = daily_returns.mean()
        std_return = daily_returns.std()
        median_return = daily_returns.median()
        sharpe = mean_return / std_return if std_return > 0 else 0

        # Volatility
        if vol_cols:
            mean_vol = regime_data[vol_cols].mean().mean()
        else:
            mean_vol = None

        # Momentum
        if momentum_cols:
            mean_momentum = regime_data[momentum_cols].mean().mean()
        else:
            mean_momentum = None

        # Return distribution
        positive_days = (daily_returns > 0).sum()
        negative_days = (daily_returns < 0).sum()

        # Store stats
        regime_stats[regime_name] = {
            'regime_num': regime_num,
            'days': len(regime_data),
            'mean_return': mean_return,
            'std_return': std_return,
            'sharpe': sharpe,
            'mean_vol': mean_vol,
            'positive_days': positive_days,
            'negative_days': negative_days
        }

        # Print detailed stats
        print(f"\n{regime_name.upper()} MARKET (Regime {regime_num})")
        print("-" * 80)
        print(f"Occurrences:     {len(regime_data)} days ({len(regime_data)/len(df_aligned)*100:.1f}% of time)")
        print(f"\nReturns:")
        print(f"  Mean daily:    {mean_return:+.6f} ({mean_return*252*100:+6.1f}% annualized)")
        print(f"  Median daily:  {median_return:+.6f}")
        print(f"  Std deviation: {std_return:.6f} ({std_return*np.sqrt(252)*100:.1f}% annualized)")
        print(f"  Sharpe ratio:  {sharpe:+.3f}")
        print(f"  Min daily:     {daily_returns.min():+.6f} ({daily_returns.min()*100:+.2f}%)")
        print(f"  Max daily:     {daily_returns.max():+.6f} ({daily_returns.max()*100:+.2f}%)")
        print(f"\nDay Distribution:")
        print(f"  Positive days: {positive_days} ({positive_days/len(regime_data)*100:.1f}%)")
        print(f"  Negative days: {negative_days} ({negative_days/len(regime_data)*100:.1f}%)")

        if mean_vol is not None:
            print(f"\nVolatility:")
            print(f"  Avg volatility: {mean_vol:.6f}")

        if mean_momentum is not None:
            print(f"\nMomentum:")
            print(f"  Avg momentum:   {mean_momentum:.6f}")

    # Regime transitions
    print("\n" + "="*80)
    print("REGIME TRANSITIONS")
    print("="*80)

    regime_series = regime_aligned['Regime'].values
    transitions = np.diff(regime_series)
    n_transitions = np.sum(transitions != 0)
    avg_duration = len(regime_series) / (n_transitions + 1)

    print(f"Total regime changes: {n_transitions}")
    print(f"Average regime duration: {avg_duration:.1f} days")

    # Transition matrix
    transition_counts = {}
    for i in range(len(regime_series) - 1):
        from_regime = int(regime_series[i])
        to_regime = int(regime_series[i + 1])
        key = (from_regime, to_regime)
        transition_counts[key] = transition_counts.get(key, 0) + 1

    print("\nTransition Matrix (from → to):")
    regime_names_by_num = {v['regime_num']: k for k, v in regime_stats.items()}

    for from_num in sorted(regime_names_by_num.keys()):
        for to_num in sorted(regime_names_by_num.keys()):
            if from_num != to_num:
                count = transition_counts.get((from_num, to_num), 0)
                from_name = regime_names_by_num[from_num]
                to_name = regime_names_by_num[to_num]
                print(f"  {from_name:10s} → {to_name:10s}: {count:3d} times")

    # Recent periods analysis
    print("\n" + "="*80)
    print("EXAMPLE PERIODS FOR EACH REGIME")
    print("="*80)

    for regime_name in ['Bull', 'Sideways', 'Bear']:
        mask = regime_aligned['Regime_Name'] == regime_name
        regime_dates = regime_aligned[mask].index

        if len(regime_dates) == 0:
            continue

        print(f"\n{regime_name.upper()} REGIME - Sample Periods:")
        print("-" * 80)

        # Find continuous periods
        periods = []
        current_period_start = None

        for i, date in enumerate(regime_dates):
            if current_period_start is None:
                current_period_start = date
            elif i > 0 and (date - regime_dates[i-1]).days > 1:
                # Period ended
                periods.append((current_period_start, regime_dates[i-1]))
                current_period_start = date

        # Add last period
        if current_period_start is not None:
            periods.append((current_period_start, regime_dates[-1]))

        # Show longest periods
        periods_sorted = sorted(periods, key=lambda x: (x[1] - x[0]).days, reverse=True)

        print(f"Longest {regime_name} periods:")
        for i, (start, end) in enumerate(periods_sorted[:5]):
            duration = (end - start).days + 1
            period_data = df_aligned.loc[start:end]
            period_return = period_data[return_cols].mean(axis=1).sum()
            print(f"  {i+1}. {start.date()} to {end.date()} ({duration:3d} days, {period_return*100:+.1f}% cumulative)")

        # Show recent periods
        print(f"\nMost recent {regime_name} periods:")
        for i, (start, end) in enumerate(periods_sorted[-5:][::-1]):
            duration = (end - start).days + 1
            period_data = df_aligned.loc[start:end]
            period_return = period_data[return_cols].mean(axis=1).sum()
            print(f"  {start.date()} to {end.date()} ({duration:3d} days, {period_return*100:+.1f}% cumulative)")

    # Feature importance
    print("\n" + "="*80)
    print("FEATURE ANALYSIS")
    print("="*80)

    print(f"\nFeatures used in HMM training:")
    print(f"  Return features:    {len(return_cols)}")
    print(f"  Volatility features: {len(vol_cols)}")
    print(f"  Momentum features:   {len(momentum_cols)}")
    print(f"  Total features:      {len(df.columns)}")

    print(f"\nAssets tracked:")
    for col in return_cols:
        print(f"  - {col}")

    print("\n" + "="*80)
    print("PORTFOLIO ALLOCATION STRATEGY")
    print("="*80)

    allocations = regime_aligned[['Regime_Name', 'Stock_Pct', 'Bond_Pct', 'Cash_Pct']].drop_duplicates()
    for _, row in allocations.iterrows():
        print(f"\n{row['Regime_Name']:10s}: {row['Stock_Pct']*100:4.0f}% stocks, "
              f"{row['Bond_Pct']*100:4.0f}% bonds, {row['Cash_Pct']*100:4.0f}% cash")

    # Save summary to file
    print("\n" + "="*80)
    print("Saving detailed report to regime_analysis_report.txt...")
    print("="*80 + "\n")

    # Create summary dataframe
    summary_df = pd.DataFrame(regime_stats).T
    summary_df['annualized_return_%'] = summary_df['mean_return'] * 252 * 100
    summary_df['annualized_vol_%'] = summary_df['std_return'] * np.sqrt(252) * 100
    summary_df['win_rate_%'] = summary_df['positive_days'] / summary_df['days'] * 100

    summary_df = summary_df[['regime_num', 'days', 'annualized_return_%', 'annualized_vol_%',
                             'sharpe', 'win_rate_%', 'mean_vol']]

    print("\nSUMMARY TABLE:")
    print(summary_df.to_string())

    # Save to CSV
    summary_df.to_csv('regime_summary.csv')
    print("\n✓ Summary saved to regime_summary.csv")

if __name__ == "__main__":
    analyze_regimes()
