"""
Build final feature matrices for each currency pair.
Loads 4H OHLCV, computes all 4 feature groups, adds targets, saves to data/processed/.

Usage:
    python src/features/build_dataset.py
"""
import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.features.technical import compute_technical_features
from src.features.volatility import compute_volatility_features
from src.features.macro import compute_macro_features
from src.features.sentiment import compute_sentiment_features

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PAIRS = ['EURUSD', 'GBPJPY', 'XAUUSD']
WARMUP_BARS = 60  # drop first N bars (NaN from rolling windows)


# Feature group mapping for SHAP attribution
FEATURE_GROUPS = {
    'technical': [
        'sma_20', 'sma_50', 'ema_12', 'ema_26', 'macd_line', 'macd_signal', 'macd_hist',
        'adx', 'plus_di', 'minus_di', 'ichimoku_conv', 'ichimoku_base',
        'rsi_14', 'stoch_k', 'stoch_d', 'williams_r', 'cci_20', 'roc_10',
        'bb_width', 'bb_pctb', 'atr_14', 'kc_upper', 'kc_lower',
        'bar_return', 'log_return', 'hl_range', 'co_range', 'gap',
    ],
    'volatility': [
        'gk_vol', 'parkinson_vol', 'rolling_std_5', 'rolling_std_20', 'rolling_std_60',
        'vol_of_vol', 'vol_ratio', 'drawdown_20',
    ],
    'macro': [
        'rate_diff', 'rate_diff_delta', 'yield_spread', 'yield_curve',
        'event_surprise', 'event_count', 'days_to_next_event',
        'is_rate_decision_day', 'is_nfp_day', 'is_cpi_day',
    ],
    'sentiment': [
        'vix', 'vix_change', 'vix_above_avg', 'dxy', 'dxy_change',
        'sp500_return', 'oil_return', 'corr_with_sp500', 'corr_with_dxy',
        'news_volume', 'news_spike',
    ],
}


def build_pair(pair_name):
    """Build feature matrix for one currency pair."""
    print(f"\n{'='*60}")
    print(f"Building {pair_name}")
    print(f"{'='*60}")

    # 1. Load 4H bars
    path = os.path.join(DATA_DIR, 'price', f'{pair_name}_4h.csv')
    df = pd.read_csv(path, parse_dates=['datetime_utc'])
    df = df.sort_values('datetime_utc').reset_index(drop=True)
    print(f"  Loaded: {len(df)} 4H bars")

    # 2. Group 1: Technical
    print("  Computing Group 1: Technical...")
    df = compute_technical_features(df)

    # 3. Group 2: Volatility (needs log_return from Group 1)
    print("  Computing Group 2: Volatility...")
    df = compute_volatility_features(df)

    # 4. Group 3: Macro
    print("  Computing Group 3: Macro...")
    df = compute_macro_features(df, pair_name, DATA_DIR)

    # 5. Group 4: Sentiment
    print("  Computing Group 4: Sentiment...")
    df = compute_sentiment_features(df, pair_name, DATA_DIR)

    # 6. Target variable: next-4H log return
    df['target_return'] = np.log(df['close'].shift(-1) / df['close'])
    df['target_direction'] = (df['target_return'] > 0).astype(int)

    # 7. Drop warmup rows
    df = df.iloc[WARMUP_BARS:].reset_index(drop=True)

    # 8. Drop last row (no target available)
    df = df.iloc[:-1].reset_index(drop=True)

    # 9. Report stats
    feature_cols = [c for c in df.columns if c not in
                    ['datetime_utc', 'date', 'session', 'target_return', 'target_direction']]
    nan_pct = df[feature_cols].isna().mean() * 100
    high_nan = nan_pct[nan_pct > 50]
    if len(high_nan) > 0:
        print(f"  WARNING: {len(high_nan)} features with >50% NaN:")
        for col, pct in high_nan.items():
            print(f"    {col}: {pct:.1f}%")

    print(f"  Features: {len(feature_cols)}")
    print(f"  Rows: {len(df)}")
    print(f"  Date range: {df['datetime_utc'].iloc[0]} to {df['datetime_utc'].iloc[-1]}")
    print(f"  Target NaN: {df['target_return'].isna().sum()}")

    # 10. Save
    out_path = os.path.join(OUTPUT_DIR, f'{pair_name}_features.csv')
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    return df


if __name__ == '__main__':
    # Save feature groups mapping
    import json
    groups_path = os.path.join(OUTPUT_DIR, 'feature_groups.json')
    with open(groups_path, 'w') as f:
        json.dump(FEATURE_GROUPS, f, indent=2)
    print(f"Feature groups saved to {groups_path}")

    for pair in PAIRS:
        build_pair(pair)

    print(f"\n{'='*60}")
    print("All pairs built. Output in data/processed/")
    print(f"{'='*60}")
