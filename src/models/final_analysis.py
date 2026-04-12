"""
Final Analysis: Statistical tests, equity curves, confusion matrices.
Run AFTER winrate_experiment.py completes.

Usage:
    python src/models/final_analysis.py
"""
import sys, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
WINRATE_DIR = os.path.join(PROJECT_ROOT, 'results', 'winrate')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

pairs = ['EURUSD', 'GBPJPY', 'XAUUSD']


def statistical_significance():
    """Paired t-test + bootstrap CI on ShiftGuard vs Technical."""
    print("=" * 60)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 60)

    results = {}
    for pair in pairs:
        df = pd.read_csv(os.path.join(WINRATE_DIR, f'{pair}_winrate_trades.csv'))

        # Per-trade PnL for each strategy
        tech_pnl = df['tech_signal'] * df['actual_return']
        sg_pnl = df['sg_signal'] * df['actual_return']

        # Only compare bars where BOTH strategies traded
        both_traded = (df['tech_signal'] != 0) & (df['sg_signal'] != 0)
        tech_both = tech_pnl[both_traded].values
        sg_both = sg_pnl[both_traded].values

        if len(tech_both) < 30:
            print(f"\n  {pair}: Not enough overlapping trades for test")
            continue

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(sg_both, tech_both)

        # Bootstrap CI for mean difference
        n_bootstrap = 10000
        diffs = sg_both - tech_both
        boot_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(diffs, size=len(diffs), replace=True)
            boot_means.append(sample.mean())
        ci_lower = np.percentile(boot_means, 2.5)
        ci_upper = np.percentile(boot_means, 97.5)

        # Win rate comparison
        sg_trades = df[df['sg_signal'] != 0]
        tech_trades = df[df['tech_signal'] != 0]
        sg_wins = ((sg_trades['sg_signal'] * sg_trades['actual_return']) > 0).mean() * 100
        tech_wins = ((tech_trades['tech_signal'] * tech_trades['actual_return']) > 0).mean() * 100

        significant = "YES" if p_value < 0.05 else "NO"

        print(f"\n  {pair}:")
        print(f"    Tech win rate:  {tech_wins:.1f}%")
        print(f"    SG win rate:    {sg_wins:.1f}%")
        print(f"    Paired t-test:  t={t_stat:.3f}, p={p_value:.6f}")
        print(f"    Significant:    {significant} (p {'<' if p_value < 0.05 else '>'} 0.05)")
        print(f"    Bootstrap 95% CI for mean PnL difference: [{ci_lower:.6f}, {ci_upper:.6f}]")
        print(f"    CI excludes zero: {'YES' if (ci_lower > 0 or ci_upper < 0) else 'NO'}")

        results[pair] = {
            'tech_win_rate': round(tech_wins, 2),
            'sg_win_rate': round(sg_wins, 2),
            't_stat': round(t_stat, 4),
            'p_value': round(p_value, 6),
            'significant': p_value < 0.05,
            'ci_lower': round(ci_lower, 6),
            'ci_upper': round(ci_upper, 6),
            'n_overlapping_trades': int(both_traded.sum()),
        }

    with open(os.path.join(FIGURES_DIR, 'statistical_tests.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


def equity_curves():
    """Plot cumulative return for Technical vs ShiftGuard vs Buy&Hold."""
    print("\n" + "=" * 60)
    print("EQUITY CURVES")
    print("=" * 60)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)

    for idx, pair in enumerate(pairs):
        df = pd.read_csv(os.path.join(WINRATE_DIR, f'{pair}_winrate_trades.csv'))
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])

        # Cumulative PnL (starting at $10,000)
        capital = 10000
        tech_pnl = df['tech_signal'] * df['actual_return']
        sg_pnl = df['sg_signal'] * df['actual_return']
        bh_pnl = df['actual_return']  # buy and hold

        tech_equity = capital * (1 + tech_pnl.cumsum())
        sg_equity = capital * (1 + sg_pnl.cumsum())
        bh_equity = capital * (1 + bh_pnl.cumsum())

        ax = axes[idx]
        ax.plot(df['datetime_utc'], tech_equity, label='Technical (RSI/MACD)', color='gray', alpha=0.7, linewidth=1)
        ax.plot(df['datetime_utc'], bh_equity, label='Buy & Hold', color='blue', alpha=0.5, linewidth=1, linestyle='--')
        ax.plot(df['datetime_utc'], sg_equity, label='ShiftGuard', color='green', linewidth=2)

        ax.set_title(f'{pair} — Equity Curve ($10K initial)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=capital, color='red', linestyle=':', alpha=0.5, label='Break-even')

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())

        final_tech = tech_equity.iloc[-1]
        final_sg = sg_equity.iloc[-1]
        ax.annotate(f'${final_tech:,.0f}', xy=(df['datetime_utc'].iloc[-1], final_tech),
                    fontsize=9, color='gray', ha='right')
        ax.annotate(f'${final_sg:,.0f}', xy=(df['datetime_utc'].iloc[-1], final_sg),
                    fontsize=9, color='green', fontweight='bold', ha='right')

        print(f"  {pair}: Tech ${final_tech:,.0f} | SG ${final_sg:,.0f} | B&H ${bh_equity.iloc[-1]:,.0f}")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'equity_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")


def win_rate_bar_chart():
    """Bar chart comparing win rates across all strategies and pairs."""
    print("\n" + "=" * 60)
    print("WIN RATE BAR CHART")
    print("=" * 60)

    data = {}
    for pair in pairs:
        with open(os.path.join(WINRATE_DIR, f'{pair}_winrate_summary.json')) as f:
            r = json.load(f)
        data[pair] = {
            'Technical': r['Technical (RSI/MACD)']['win_rate'],
            'ShiftGuard': r['ShiftGuard (Regime-Filtered)']['win_rate'],
        }

    x = np.arange(len(pairs))
    width = 0.3

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, [data[p]['Technical'] for p in pairs], width, label='Technical (RSI/MACD)', color='gray', alpha=0.7)
    bars2 = ax.bar(x + width/2, [data[p]['ShiftGuard'] for p in pairs], width, label='ShiftGuard', color='green')

    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_title('Win Rate Comparison: Technical vs ShiftGuard (4H bars)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(45, 65)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10, color='gray')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10, color='green', fontweight='bold')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'winrate_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def profit_factor_chart():
    """Profit factor comparison."""
    print("\n" + "=" * 60)
    print("PROFIT FACTOR CHART")
    print("=" * 60)

    data = {}
    for pair in pairs:
        with open(os.path.join(WINRATE_DIR, f'{pair}_winrate_summary.json')) as f:
            r = json.load(f)
        data[pair] = {
            'Technical': r['Technical (RSI/MACD)']['profit_factor'],
            'ShiftGuard': r['ShiftGuard (Regime-Filtered)']['profit_factor'],
        }

    x = np.arange(len(pairs))
    width = 0.3

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, [data[p]['Technical'] for p in pairs], width, label='Technical', color='gray', alpha=0.7)
    bars2 = ax.bar(x + width/2, [data[p]['ShiftGuard'] for p in pairs], width, label='ShiftGuard', color='green')

    ax.set_ylabel('Profit Factor', fontsize=12)
    ax.set_title('Profit Factor: Technical vs ShiftGuard', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, fontsize=12)
    ax.legend(fontsize=11)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Break-even')
    ax.grid(axis='y', alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'profit_factor_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


if __name__ == '__main__':
    statistical_significance()
    equity_curves()
    win_rate_bar_chart()
    profit_factor_chart()

    print(f"\n{'='*60}")
    print("All analysis complete. Figures saved to results/figures/")
    print(f"{'='*60}")
