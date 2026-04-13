"""Generate dashboard-facing statistical summaries and figures from current winrate outputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[2] / '.mpl-cache'))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WINRATE_DIR = PROJECT_ROOT / 'results' / 'winrate'
FIGURES_DIR = PROJECT_ROOT / 'results' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_trade_data(pairs: list[str]) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        path = WINRATE_DIR / f'{pair}_winrate_trades.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], errors='coerce')
        df = df.dropna(subset=['datetime_utc']).sort_values('datetime_utc').reset_index(drop=True)
        data[pair] = df
    return data


def load_winrate_overall() -> dict[str, dict[str, dict[str, float]]]:
    path = WINRATE_DIR / 'winrate_overall.json'
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def strategy_curve(df: pd.DataFrame, signal_col: str) -> pd.Series:
    return (df[signal_col].astype(float) * df['actual_return'].astype(float)).cumsum()


def compute_statistical_tests(data: dict[str, pd.DataFrame]) -> dict[str, dict[str, float | int | str]]:
    summaries: dict[str, dict[str, float | int | str]] = {}
    for pair, df in data.items():
        overlap = df[(df['tech_signal'] != 0) & (df['sg_signal'] != 0)].copy()
        if overlap.empty:
            continue

        tech_pnl = overlap['tech_signal'].astype(float) * overlap['actual_return'].astype(float)
        sg_pnl = overlap['sg_signal'].astype(float) * overlap['actual_return'].astype(float)
        diff = sg_pnl - tech_pnl

        t_stat, p_value = stats.ttest_rel(sg_pnl, tech_pnl, nan_policy='omit')
        mean_diff = float(diff.mean())
        sem = stats.sem(diff, nan_policy='omit')
        if len(diff) > 1 and np.isfinite(sem):
            ci_low, ci_high = stats.t.interval(0.95, len(diff) - 1, loc=mean_diff, scale=sem)
        else:
            ci_low = ci_high = mean_diff

        summaries[pair] = {
            'tech_win_rate': round(float((tech_pnl > 0).mean() * 100), 2),
            'sg_win_rate': round(float((sg_pnl > 0).mean() * 100), 2),
            't_stat': round(float(t_stat), 4),
            'p_value': round(float(p_value), 6),
            'significant': str(bool(p_value < 0.05)),
            'ci_lower': round(float(ci_low), 6),
            'ci_upper': round(float(ci_high), 6),
            'n_overlapping_trades': int(len(overlap)),
        }

    with open(FIGURES_DIR / 'statistical_tests.json', 'w') as f:
        json.dump(summaries, f, indent=2)
    return summaries


def plot_winrate_comparison(overall: dict[str, dict[str, dict[str, float]]], pairs: list[str]) -> None:
    strategies = ['Technical (RSI/MACD)', 'ML Direction (XGBoost)', 'ShiftGuard (Regime-Filtered)']
    labels = ['Technical', 'ML Direction', 'ShiftGuard']
    x = np.arange(len(pairs))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, (strategy, label) in enumerate(zip(strategies, labels)):
        values = [overall.get(pair, {}).get(strategy, {}).get('win_rate', 0) for pair in pairs]
        ax.bar(x + (idx - 1) * width, values, width=width, label=label)

    ax.set_title('Win Rate Comparison by Pair')
    ax.set_xticks(x)
    ax.set_xticklabels(pairs)
    ax.set_ylabel('Win Rate (%)')
    ax.legend()
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'winrate_comparison.png', dpi=180)
    plt.close(fig)


def plot_profit_factor_comparison(overall: dict[str, dict[str, dict[str, float]]], pairs: list[str]) -> None:
    strategies = ['Technical (RSI/MACD)', 'ML Direction (XGBoost)', 'ShiftGuard (Regime-Filtered)']
    labels = ['Technical', 'ML Direction', 'ShiftGuard']
    x = np.arange(len(pairs))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, (strategy, label) in enumerate(zip(strategies, labels)):
        values = [overall.get(pair, {}).get(strategy, {}).get('profit_factor', 0) for pair in pairs]
        ax.bar(x + (idx - 1) * width, values, width=width, label=label)

    ax.set_title('Profit Factor Comparison by Pair')
    ax.set_xticks(x)
    ax.set_xticklabels(pairs)
    ax.set_ylabel('Profit Factor')
    ax.legend()
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'profit_factor_comparison.png', dpi=180)
    plt.close(fig)


def plot_equity_curves(data: dict[str, pd.DataFrame], pairs: list[str]) -> None:
    fig, axes = plt.subplots(len(pairs), 1, figsize=(12, 4 * len(pairs)), sharex=False)
    if len(pairs) == 1:
        axes = [axes]

    for ax, pair in zip(axes, pairs):
        df = data[pair]
        ax.plot(df['datetime_utc'], strategy_curve(df, 'tech_signal'), label='Technical', alpha=0.9)
        ax.plot(df['datetime_utc'], strategy_curve(df, 'ml_signal'), label='ML Direction', alpha=0.9)
        ax.plot(df['datetime_utc'], strategy_curve(df, 'sg_signal'), label='ShiftGuard', alpha=0.9)
        ax.set_title(f'Equity Curves - {pair}')
        ax.set_ylabel('Cumulative Return')
        ax.grid(alpha=0.2)
        ax.legend()

    axes[-1].set_xlabel('Datetime (UTC)')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'equity_curves.png', dpi=180)
    plt.close(fig)


def plot_regime_confusion_matrix(data: dict[str, pd.DataFrame]) -> None:
    frames = [df[['target_regime', 'regime']] for df in data.values() if {'target_regime', 'regime'}.issubset(df.columns)]
    if not frames:
        return

    combined = pd.concat(frames, ignore_index=True).dropna()
    if combined.empty:
        return

    y_true = combined['target_regime'].astype(int)
    y_pred = combined['regime'].astype(int)
    labels = [0, 1, 2, 3, 4]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', cbar=False, ax=ax)
    ax.set_title('Regime Classifier Confusion Matrix')
    ax.set_xlabel('Predicted Regime')
    ax.set_ylabel('Target Regime')
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels, rotation=0)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'regime_confusion_matrix.png', dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", default=['EURUSD', 'GBPJPY', 'XAUUSD'])
    args = parser.parse_args()

    data = load_trade_data(args.pairs)
    if not data:
        raise FileNotFoundError("No winrate trade CSVs found. Run src/models/winrate_experiment.py first.")

    overall = load_winrate_overall()
    compute_statistical_tests(data)
    plot_equity_curves(data, list(data.keys()))
    if overall:
        plot_winrate_comparison(overall, list(data.keys()))
        plot_profit_factor_comparison(overall, list(data.keys()))
    plot_regime_confusion_matrix(data)


if __name__ == '__main__':
    main()
