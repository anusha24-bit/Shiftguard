"""
Win Rate Experiment — DAILY bars
Same logic as winrate_experiment.py but on daily OHLCV.
Macro/sentiment features align naturally with daily bars.

Usage:
    python src/models/winrate_daily.py
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.features.technical import compute_technical_features
from src.features.volatility import compute_volatility_features
from src.features.macro import compute_macro_features
from src.features.sentiment import compute_sentiment_features
from src.features.regime import compute_regime_features, compute_adaptive_regime_labels

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'winrate_daily')
os.makedirs(RESULTS_DIR, exist_ok=True)

META_COLS = ['datetime_utc', 'date', 'session', 'target_return', 'target_direction',
             'volume', 'month', 'regime_label', 'target_regime', 'regime_changed',
             'target_transition_6', 'target_transition_12', 'target_transition_18',
             'bars_since_regime_change', 'market_state', 'target_market_state', 'target_dir']

REGIME_PARAMS = {
    'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 300,
    'reg_alpha': 0.1, 'reg_lambda': 3.0, 'tree_method': 'hist',
    'random_state': 42, 'verbosity': 0, 'objective': 'multi:softprob',
    'num_class': 5, 'eval_metric': 'mlogloss',
}

DIR_PARAMS = {
    'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 500,
    'reg_alpha': 0.1, 'reg_lambda': 5.0, 'tree_method': 'hist',
    'random_state': 42, 'verbosity': 0,
}

TRAIN_DAYS = 252       # 1 year
STEP_DAYS = 21         # 1 month
CONFIDENCE_THRESHOLD = 0.55


def get_feature_cols(df):
    return [c for c in df.columns if c not in META_COLS]


def create_5class_regime(df):
    ret_20 = df['close'].pct_change(20)
    trend_up = ret_20 > 0.01
    trend_down = ret_20 < -0.01
    high_vol = df['atr_pct_short'] > 60
    low_vol = ~high_vol

    labels = np.full(len(df), 2)
    labels[trend_up & low_vol] = 0
    labels[trend_up & high_vol] = 1
    labels[trend_down & low_vol] = 3
    labels[trend_down & high_vol] = 4

    df['market_state'] = labels
    df['target_market_state'] = df['market_state'].shift(-1).ffill().astype(int)
    return df


def strategy_technical(df):
    signals = []
    for _, row in df.iterrows():
        signal = 0
        rsi = row.get('rsi_14', 50)
        macd = row.get('macd_hist', 0)
        if pd.isna(rsi) or pd.isna(macd):
            signals.append(0)
            continue
        if rsi < 30:
            signal = 1
        elif rsi > 70:
            signal = -1
        elif macd > 0:
            signal = 1
        elif macd < 0:
            signal = -1
        signals.append(signal)
    return np.array(signals)


def strategy_ml_direction(dir_model, X_chunk):
    """ML direction: use classifier prediction, not regressor sign."""
    pred = dir_model.predict(X_chunk)
    # Convert: predicted return > 0 → long, else short
    signals = np.where(pred > 0, 1, -1)
    return signals


def evaluate_strategy(signals, actual_returns, name):
    trade_mask = signals != 0
    n_trades = trade_mask.sum()
    if n_trades == 0:
        return {'name': name, 'win_rate': 0, 'n_trades': 0, 'trade_pct': 0,
                'avg_win': 0, 'avg_loss': 0, 'profit_factor': 0, 'total_return': 0}

    trade_pct = n_trades / len(signals) * 100
    pnl = signals[trade_mask] * actual_returns[trade_mask]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    win_rate = len(wins) / n_trades * 100
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else 99
    total_return = pnl.sum()

    return {
        'name': name, 'win_rate': round(win_rate, 2), 'n_trades': int(n_trades),
        'trade_pct': round(trade_pct, 1), 'avg_win': round(float(avg_win), 6),
        'avg_loss': round(float(avg_loss), 6),
        'profit_factor': round(float(min(profit_factor, 99)), 2),
        'total_return': round(float(total_return), 6),
    }


def ensure_all_classes(X, y, n_classes=5):
    unique = np.unique(y)
    for cls in range(n_classes):
        if cls not in unique:
            idx = np.random.randint(0, len(X))
            X = np.vstack([X, X[idx:idx+1]])
            y = np.append(y, cls)
    return X, y


def run_pair(pair_name):
    print(f"\n{'='*60}")
    print(f"Win Rate (DAILY) — {pair_name}")
    print(f"{'='*60}")

    # Load daily OHLCV
    path = os.path.join(DATA_DIR, 'price', f'{pair_name}_ohlcv.csv')
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Rename for compatibility
    df = df.rename(columns={'date': 'datetime_utc'})
    if 'date' not in df.columns:
        df['date'] = df['datetime_utc'].dt.date.astype(str)

    # Ensure OHLC column order
    for col in ['open', 'high', 'low', 'close']:
        if col not in df.columns:
            print(f"  ERROR: missing {col} column")
            return None

    print(f"  Loaded: {len(df)} daily bars")

    # Compute features
    df = compute_technical_features(df)
    df = compute_volatility_features(df)
    df = compute_macro_features(df, pair_name, DATA_DIR)
    df = compute_sentiment_features(df, pair_name, DATA_DIR)
    df = compute_regime_features(df)
    df = compute_adaptive_regime_labels(df)
    df = create_5class_regime(df)

    df['target_return'] = np.log(df['close'].shift(-1) / df['close'])
    df['target_dir'] = (df['target_return'] > 0).astype(int)

    df = df.iloc[260:].reset_index(drop=True)
    df = df.iloc[:-1].reset_index(drop=True)

    feature_cols = get_feature_cols(df)
    print(f"  Rows: {len(df)}, Features: {len(feature_cols)}")

    # Walk-forward
    cursor = TRAIN_DAYS
    total = len(df)
    all_records = []

    train_start = max(0, cursor - TRAIN_DAYS)
    X_init = df.iloc[train_start:cursor][feature_cols].values
    y_ret_init = df.iloc[train_start:cursor]['target_return'].values
    y_regime_init = df.iloc[train_start:cursor]['target_market_state'].values

    # ML direction model — retrain monthly alongside regime
    dir_model = xgb.XGBRegressor(**DIR_PARAMS)
    dir_model.fit(X_init, y_ret_init, verbose=False)

    # Regime model
    X_reg, y_reg = ensure_all_classes(X_init, y_regime_init)
    regime_model = xgb.XGBClassifier(**REGIME_PARAMS)
    regime_model.fit(X_reg, y_reg, verbose=False)

    retrain_count = 0
    step = 0

    while cursor + STEP_DAYS <= total:
        step_end = cursor + STEP_DAYS
        chunk = df.iloc[cursor:step_end]
        X_chunk = chunk[feature_cols].values
        actual_returns = chunk['target_return'].values

        # Strategy 1: Technical
        tech_signals = strategy_technical(chunk)

        # Strategy 2: ML direction (retrained monthly)
        ml_signals = strategy_ml_direction(dir_model, X_chunk)

        # Strategy 3: ShiftGuard
        regime_probs = regime_model.predict_proba(X_chunk)
        regime_pred = regime_model.predict(X_chunk)
        regime_confidence = regime_probs.max(axis=1)

        sg_signals = np.zeros(len(X_chunk))
        for i in range(len(X_chunk)):
            if regime_confidence[i] < CONFIDENCE_THRESHOLD:
                continue
            bars_since = chunk['bars_since_regime_change'].iloc[i]
            if bars_since < 2:
                continue
            state = regime_pred[i]
            if state in [0, 1]:
                sg_signals[i] = 1
            elif state in [3, 4]:
                sg_signals[i] = -1

        for i in range(len(actual_returns)):
            all_records.append({
                'datetime_utc': chunk['datetime_utc'].iloc[i],
                'actual_return': actual_returns[i],
                'tech_signal': tech_signals[i],
                'ml_signal': ml_signals[i],
                'sg_signal': sg_signals[i],
            })

        # Retrain both models monthly
        train_start = max(0, step_end - TRAIN_DAYS)
        X_retrain = df.iloc[train_start:step_end][feature_cols].values
        y_ret_retrain = df.iloc[train_start:step_end]['target_return'].values
        y_reg_retrain = df.iloc[train_start:step_end]['target_market_state'].values

        dir_model = xgb.XGBRegressor(**DIR_PARAMS)
        dir_model.fit(X_retrain, y_ret_retrain, verbose=False)

        X_r, y_r = ensure_all_classes(X_retrain, y_reg_retrain)
        regime_model = xgb.XGBClassifier(**REGIME_PARAMS)
        regime_model.fit(X_r, y_r, verbose=False)
        retrain_count += 1

        step += 1
        cursor = step_end

    # Evaluate
    records_df = pd.DataFrame(all_records)
    actual = records_df['actual_return'].values

    print(f"\n{'='*60}")
    print(f"RESULTS (DAILY) — {pair_name}")
    print(f"{'='*60}")

    results = {}
    for sname, col in [('Technical (RSI/MACD)', 'tech_signal'),
                        ('ML Direction (XGBoost)', 'ml_signal'),
                        ('ShiftGuard (Regime-Filtered)', 'sg_signal')]:
        signals = records_df[col].values
        r = evaluate_strategy(signals, actual, sname)
        results[sname] = r
        print(f"\n  {sname}:")
        print(f"    Win Rate:       {r['win_rate']:.1f}%")
        print(f"    Trades:         {r['n_trades']} ({r['trade_pct']:.1f}% of bars)")
        print(f"    Profit Factor:  {r['profit_factor']:.2f}")
        print(f"    Total Return:   {r['total_return']:.6f}")

    results['regime_retrains'] = retrain_count

    records_df.to_csv(os.path.join(RESULTS_DIR, f'{pair_name}_winrate_daily_trades.csv'), index=False)
    with open(os.path.join(RESULTS_DIR, f'{pair_name}_winrate_daily_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved to results/winrate_daily/")
    return results


if __name__ == '__main__':
    all_results = {}
    for pair in ['EURUSD', 'GBPJPY', 'XAUUSD']:
        results = run_pair(pair)
        if results:
            all_results[pair] = results

    print(f"\n{'='*60}")
    print("WIN RATE (DAILY) — Cross-Pair Summary")
    print(f"{'='*60}")
    print(f"\n{'Pair':<10} {'Tech Win%':<12} {'ML Win%':<12} {'SG Win%':<12} {'SG Trades%':<12} {'SG PF'}")
    print("-" * 60)
    for pair, r in all_results.items():
        tech = r['Technical (RSI/MACD)']
        ml = r['ML Direction (XGBoost)']
        sg = r['ShiftGuard (Regime-Filtered)']
        print(f"{pair:<10} {tech['win_rate']:<12.1f} {ml['win_rate']:<12.1f} "
              f"{sg['win_rate']:<12.1f} {sg['trade_pct']:<12.1f} {sg['profit_factor']:.2f}")

    with open(os.path.join(RESULTS_DIR, 'winrate_daily_overall.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: results/winrate_daily/winrate_daily_overall.json")
