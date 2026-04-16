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

DEFAULT_CAPITAL = 10000
DEFAULT_LEVERAGE = 20
DEFAULT_STOP_LOSS_PCT = 1.0
SPREAD_MAP = {'EURUSD': 0.00010, 'GBPJPY': 0.00020, 'XAUUSD': 0.00030}
COMMISSION_COST = 0.00003
SLIPPAGE_COST = 0.00002
SWAP_COST = 0.000005
TAX_RATE = 0.30


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


def strategy_net_returns(
    df: pd.DataFrame,
    signal_col: str,
    pair: str,
    leverage: float = DEFAULT_LEVERAGE,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
) -> pd.Series:
    signals = df[signal_col].fillna(0).astype(float)
    returns = df['actual_return'].fillna(0).astype(float)
    active = signals != 0
    net = pd.Series(0.0, index=df.index, dtype=float)
    if not active.any():
        return net

    leveraged = signals[active] * returns[active] * leverage
    stop_loss = (stop_loss_pct / 100) * leverage
    capped = leveraged.clip(lower=-stop_loss)
    cost_per_trade = (
        SPREAD_MAP.get(pair, 0.00020) + COMMISSION_COST + SLIPPAGE_COST + SWAP_COST
    ) * leverage
    net.loc[active] = capped - cost_per_trade
    return net


def strategy_equity_curve(
    df: pd.DataFrame,
    signal_col: str,
    pair: str,
    capital: float = DEFAULT_CAPITAL,
) -> pd.Series:
    net = strategy_net_returns(df, signal_col, pair)
    pre_tax_profit = capital * net.cumsum()
    running_tax = pre_tax_profit.clip(lower=0) * TAX_RATE
    equity = (capital + pre_tax_profit - running_tax).clip(lower=0)

    blown = equity <= 0
    if blown.any():
        first_blow_index = blown.idxmax()
        equity.loc[first_blow_index:] = 0
    return equity


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


def plot_market_participation_comparison(
    overall: dict[str, dict[str, dict[str, float]]],
    pairs: list[str],
) -> None:
    strategies = ['Technical (RSI/MACD)', 'ML Direction (XGBoost)', 'ShiftGuard (Regime-Filtered)']
    labels = ['Technical', 'ML Direction', 'ShiftGuard']
    x = np.arange(len(pairs))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, (strategy, label) in enumerate(zip(strategies, labels)):
        values = [overall.get(pair, {}).get(strategy, {}).get('trade_pct', 0) for pair in pairs]
        ax.bar(x + (idx - 1) * width, values, width=width, label=label)

    ax.set_title('Market Participation Comparison by Pair')
    ax.set_xticks(x)
    ax.set_xticklabels(pairs)
    ax.set_ylabel('Trade Percentage (%)')
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
        ax.plot(df['datetime_utc'], strategy_equity_curve(df, 'tech_signal', pair), label='Technical', alpha=0.9)
        ax.plot(df['datetime_utc'], strategy_equity_curve(df, 'ml_signal', pair), label='ML Direction', alpha=0.9)
        ax.plot(df['datetime_utc'], strategy_equity_curve(df, 'sg_signal', pair), label='ShiftGuard', alpha=0.9)
        ax.set_title(f'Equity Curves - {pair} ($10k, 1:20, 1% SL)')
        ax.set_ylabel('Portfolio Value ($)')
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
        plot_market_participation_comparison(overall, list(data.keys()))
        plot_profit_factor_comparison(overall, list(data.keys()))
    plot_regime_confusion_matrix(data)


if __name__ == '__main__':
    main()
