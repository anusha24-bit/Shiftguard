"""
Win Rate Comparison Experiment
Compares 3 approaches to trading:
  1) Technical indicators alone (RSI/MACD signals)
  2) ML alone (XGBoost direction prediction, trade every bar)
  3) ShiftGuard (regime-aware, only trade when confident, sit out during transitions)

The key insight: ShiftGuard trades less but wins more.

Usage:
    python src/models/winrate_experiment.py
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.features.regime import compute_regime_features, compute_adaptive_regime_labels
from src.features.technical import compute_technical_features
from src.features.volatility import compute_volatility_features
from src.features.macro import compute_macro_features
from src.features.sentiment import compute_sentiment_features
from src.models.baseline_technical import generate_technical_signals
from src.models.baseline_ml_direction import train_direction_model, predict_direction_signals

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'winrate')
DETECTION_DIR = os.path.join(PROJECT_ROOT, 'results', 'detection')
ATTRIBUTION_DIR = os.path.join(PROJECT_ROOT, 'results', 'attribution')
os.makedirs(RESULTS_DIR, exist_ok=True)

META_COLS = ['datetime_utc', 'date', 'session', 'target_return', 'target_direction',
             'volume', 'month', 'regime_label', 'target_regime', 'regime_changed',
             'target_transition_6', 'target_transition_12', 'target_transition_18',
             'bars_since_regime_change']

REGIME_PARAMS = {
    'learning_rate': 0.05,
    'max_depth': 5,
    'n_estimators': 300,
    'reg_alpha': 0.1,
    'reg_lambda': 3.0,
    'tree_method': 'hist',
    'random_state': 42,
    'verbosity': 0,
    'objective': 'multi:softprob',
    'num_class': 5,
    'eval_metric': 'mlogloss',
}

TRAIN_BARS = 180 * 6  # 6 months for 4H
STEP_BARS = 6         # 1 day (6 × 4H bars) — check triggers daily
RETRAIN_COOLDOWN = 30 # minimum 30 bars (~5 days) between retrains
CONFIDENCE_THRESHOLD = 0.55  # only trade when regime confidence > this
SHIFT_LOOKBACK_BARS = 30


def get_feature_cols(df):
    return [c for c in df.columns if c not in META_COLS]


def create_5class_regime(df):
    """
    5-class market state:
    0: Trending Up + Low Vol
    1: Trending Up + High Vol
    2: Ranging
    3: Trending Down + Low Vol
    4: Trending Down + High Vol
    """
    # Trend: based on 20-bar return
    ret_20 = df['close'].pct_change(20)
    trend_up = ret_20 > 0.005    # >0.5% over 20 bars
    trend_down = ret_20 < -0.005
    ranging = ~trend_up & ~trend_down

    # Vol: based on ATR percentile
    high_vol = df['atr_pct_short'] > 60
    low_vol = df['atr_pct_short'] <= 60

    labels = np.full(len(df), 2)  # default: ranging
    labels[trend_up & low_vol] = 0   # trending up, low vol
    labels[trend_up & high_vol] = 1  # trending up, high vol
    labels[trend_down & low_vol] = 3 # trending down, low vol
    labels[trend_down & high_vol] = 4 # trending down, high vol

    df['market_state'] = labels
    df['target_market_state'] = df['market_state'].shift(-1).ffill().astype(int)

    return df


def evaluate_strategy(signals, actual_returns, name):
    """
    Evaluate a trading strategy.
    signal: 1 = long, -1 = short, 0 = no trade
    Returns: dict with win rate, trade count, profit factor
    """
    trade_mask = signals != 0
    n_trades = trade_mask.sum()

    if n_trades == 0:
        return {'name': name, 'win_rate': 0, 'n_trades': 0, 'trade_pct': 0,
                'avg_win': 0, 'avg_loss': 0, 'profit_factor': 0, 'total_return': 0}

    trade_pct = n_trades / len(signals) * 100

    # PnL per trade: signal * actual return
    pnl = signals[trade_mask] * actual_returns[trade_mask]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    win_rate = len(wins) / n_trades * 100
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float('inf')
    total_return = pnl.sum()

    return {
        'name': name,
        'win_rate': round(win_rate, 2),
        'n_trades': int(n_trades),
        'trade_pct': round(trade_pct, 1),
        'avg_win': round(float(avg_win), 6),
        'avg_loss': round(float(avg_loss), 6),
        'profit_factor': round(float(min(profit_factor, 99)), 2),
        'total_return': round(float(total_return), 6),
    }


def load_shift_context(pair_name):
    shifts_path = os.path.join(DETECTION_DIR, f'{pair_name}_shifts.csv')
    if not os.path.exists(shifts_path):
        return pd.DataFrame()

    shifts = pd.read_csv(shifts_path)
    shifts['datetime_utc'] = pd.to_datetime(shifts['datetime_utc'], format='mixed', errors='coerce')
    shifts = shifts.dropna(subset=['datetime_utc'])

    attr_path = os.path.join(ATTRIBUTION_DIR, f'{pair_name}_attribution.csv')
    if os.path.exists(attr_path):
        attr = pd.read_csv(attr_path)
        attr['datetime_utc'] = pd.to_datetime(attr['datetime_utc'], format='mixed', errors='coerce')
        attr = attr.dropna(subset=['datetime_utc'])
        shifts = shifts.merge(
            attr[['datetime_utc', 'dominant_group']].drop_duplicates('datetime_utc'),
            on='datetime_utc',
            how='left',
        )

    if 'dominant_group' not in shifts.columns:
        shifts['dominant_group'] = 'unknown'
    shifts['dominant_group'] = shifts['dominant_group'].fillna('unknown')
    return shifts.sort_values('datetime_utc').reset_index(drop=True)


def get_recent_shift_context(shifts_df, current_dt, lookback_bars=SHIFT_LOOKBACK_BARS):
    if shifts_df.empty:
        return None

    lookback_window = pd.Timedelta(hours=4 * lookback_bars)
    recent = shifts_df[
        (shifts_df['datetime_utc'] <= current_dt) &
        (shifts_df['datetime_utc'] >= current_dt - lookback_window)
    ].copy()
    if recent.empty:
        return None

    sort_cols = ['datetime_utc']
    ascending = [False]
    if 'severity' in recent.columns:
        sort_cols = ['severity', 'datetime_utc']
        ascending = [False, False]
    recent = recent.sort_values(sort_cols, ascending=ascending)
    return recent.iloc[0]


def choose_shiftguard_policy(shift_ctx):
    if shift_ctx is None:
        return 'default_regime'

    shift_type = str(shift_ctx.get('type', 'unknown'))
    dominant = str(shift_ctx.get('dominant_group', 'unknown'))

    if dominant == 'technical':
        return 'technical_followthrough'
    if shift_type == 'scheduled' or dominant in {'macro', 'sentiment'}:
        return 'event_alignment'
    if shift_type == 'unexpected' or dominant == 'volatility':
        return 'defensive_shock'
    return 'default_regime'


def adaptive_shiftguard_signal(row, tech_signal, ml_signal, regime_state, regime_conf, shift_ctx):
    policy = choose_shiftguard_policy(shift_ctx)
    bars_since = row.get('bars_since_regime_change', np.inf)

    if policy == 'technical_followthrough':
        if regime_conf < 0.50 or bars_since < 2:
            return 0, policy
        if regime_state in [0, 1]:
            if ml_signal == 1:
                return 1, policy
            if tech_signal == 1:
                return 1, policy
        if regime_state in [3, 4]:
            if ml_signal == -1:
                return -1, policy
            if tech_signal == -1:
                return -1, policy
        return 0, policy

    if policy == 'event_alignment':
        if regime_conf < 0.60 or bars_since < 3:
            return 0, policy
        if regime_state in [0, 1] and ml_signal == 1:
            return 1, policy
        if regime_state in [3, 4] and ml_signal == -1:
            return -1, policy
        return 0, policy

    if policy == 'defensive_shock':
        if regime_conf < 0.68 or bars_since < 5:
            return 0, policy
        if regime_state in [0, 1] and ml_signal == 1 and tech_signal >= 0:
            return 1, policy
        if regime_state in [3, 4] and ml_signal == -1 and tech_signal <= 0:
            return -1, policy
        return 0, policy

    if regime_conf < CONFIDENCE_THRESHOLD or bars_since < 3:
        return 0, policy
    if regime_state in [0, 1]:
        return 1, policy
    if regime_state in [3, 4]:
        return -1, policy
    return 0, policy


def run_pair(pair_name):
    print(f"\n{'='*60}")
    print(f"Win Rate Experiment — {pair_name}")
    print(f"{'='*60}")

    # Build full dataset with regime features
    print("  Building dataset...")
    path = os.path.join(DATA_DIR, 'price', f'{pair_name}_4h.csv')
    df = pd.read_csv(path, parse_dates=['datetime_utc'])
    df = df.sort_values('datetime_utc').reset_index(drop=True)

    df = compute_technical_features(df)
    df = compute_volatility_features(df)
    df = compute_macro_features(df, pair_name, DATA_DIR)
    df = compute_sentiment_features(df, pair_name, DATA_DIR)
    df = compute_regime_features(df)
    df = compute_adaptive_regime_labels(df)
    df = create_5class_regime(df)

    # Compute target
    df['target_return'] = np.log(df['close'].shift(-1) / df['close'])
    df['target_dir'] = (df['target_return'] > 0).astype(int)

    df = df.iloc[260:].reset_index(drop=True)
    df = df.iloc[:-1].reset_index(drop=True)

    feature_cols = get_feature_cols(df)
    shifts_ctx = load_shift_context(pair_name)
    print(f"  Rows: {len(df)}, Features: {len(feature_cols)}")
    print(f"  Market state distribution: {pd.Series(df['market_state']).value_counts().to_dict()}")

    # Walk-forward
    cursor = TRAIN_BARS
    total = len(df)

    all_records = []
    step = 0

    # Initial training
    train_start = max(0, cursor - TRAIN_BARS)
    X_init = df.iloc[train_start:cursor][feature_cols].values
    y_dir_init = df.iloc[train_start:cursor]['target_dir'].values
    y_regime_init = df.iloc[train_start:cursor]['target_market_state'].values

    # ML direction model — use ONLY Groups 1-4 features (no regime labels)
    # Train on first 6 months, static, never retrain
    dir_model, base_feature_cols = train_direction_model(df, feature_cols, train_start, cursor)
    print(f'  ML baseline features: {len(base_feature_cols)} (no regime features)')

    # Regime classifier (5-class)
    # Ensure all classes present
    unique_classes = np.unique(y_regime_init)
    X_regime = X_init.copy()
    y_regime = y_regime_init.copy()
    for cls in range(5):
        if cls not in unique_classes:
            idx = np.random.randint(0, len(X_regime))
            X_regime = np.vstack([X_regime, X_regime[idx:idx+1]])
            y_regime = np.append(y_regime, cls)

    regime_model = xgb.XGBClassifier(**REGIME_PARAMS)
    regime_model.fit(X_regime, y_regime, verbose=False)

    regime_retrain_count = 0
    bars_since_last_retrain = RETRAIN_COOLDOWN  # allow retrain on first trigger
    retrain_reasons_log = []
    total_steps = (total - TRAIN_BARS) // STEP_BARS

    while cursor + STEP_BARS <= total:
        step_end = cursor + STEP_BARS
        chunk = df.iloc[cursor:step_end]
        X_chunk = chunk[feature_cols].values
        actual_returns = chunk['target_return'].values

        # Progress every 500 steps
        if step % 500 == 0:
            print(f"    step {step}/{total_steps} ...")

        # --- Strategy 1: Technical signals ---
        tech_signals = generate_technical_signals(chunk)

        # --- Strategy 2: ML direction (static classifier, no regime features) ---
        ml_signals = predict_direction_signals(dir_model, chunk, base_feature_cols)

        # --- Strategy 3: ShiftGuard regime-filtered ---
        regime_probs = regime_model.predict_proba(X_chunk)
        regime_pred = regime_model.predict(X_chunk)
        regime_confidence = regime_probs.max(axis=1)

        # Determine trade direction from regime
        # States 0,1 (trending up) → long; States 3,4 (trending down) → short; State 2 (ranging) → skip
        shiftguard_signals = np.zeros(len(X_chunk))
        shiftguard_policies = []
        for i in range(len(X_chunk)):
            row = chunk.iloc[i]
            shift_ctx = get_recent_shift_context(shifts_ctx, row['datetime_utc'])
            signal, policy = adaptive_shiftguard_signal(
                row=row,
                tech_signal=tech_signals[i],
                ml_signal=ml_signals[i],
                regime_state=regime_pred[i],
                regime_conf=regime_confidence[i],
                shift_ctx=shift_ctx,
            )
            shiftguard_signals[i] = signal
            shiftguard_policies.append(policy)
            continue

            if conf < CONFIDENCE_THRESHOLD:
                continue  # low confidence → sit out

            # Check if near regime transition (sit out)
            bars_since = chunk['bars_since_regime_change'].iloc[i]
            if bars_since < 3:
                continue  # just after a transition → sit out

            if state in [0, 1]:   # trending up
                shiftguard_signals[i] = 1
            elif state in [3, 4]:  # trending down
                shiftguard_signals[i] = -1
            # state 2 (ranging) → skip

        # Record
        for i in range(len(actual_returns)):
            all_records.append({
                'datetime_utc': chunk['datetime_utc'].iloc[i],
                'actual_return': actual_returns[i],
                'tech_signal': tech_signals[i],
                'ml_signal': ml_signals[i],
                'sg_signal': shiftguard_signals[i],
                'sg_policy': shiftguard_policies[i],
                'regime': regime_pred[i],
                'regime_confidence': regime_confidence[i],
            })

        # === RETRAIN TRIGGERS (any of these fires → retrain) ===
        should_retrain = False
        retrain_reason = ''

        # Trigger 1: Regime label changed
        if chunk['regime_changed'].sum() > 0:
            should_retrain = True
            retrain_reason = 'regime_change'

        # Trigger 2: Vol spike (vol_ratio > 2.5 = short-term vol is 2.5x long-term)
        if 'vol_ratio' in chunk.columns:
            vol_spikes = (chunk['vol_ratio'] > 2.5).sum()
            if vol_spikes >= 1:
                should_retrain = True
                retrain_reason = 'vol_spike'

        # Trigger 3: Rolling win rate dropped below 45% (model degrading)
        if len(all_records) >= 60:
            recent = all_records[-60:]
            recent_sg = [(r['sg_signal'] * r['actual_return']) > 0
                         for r in recent if r['sg_signal'] != 0]
            if len(recent_sg) >= 10:
                recent_wr = sum(recent_sg) / len(recent_sg)
                if recent_wr < 0.45:
                    should_retrain = True
                    retrain_reason = 'performance_drop'

        # Trigger 4: Large event surprise (calendar-driven shift)
        if 'event_surprise' in chunk.columns:
            big_surprise = (chunk['event_surprise'].abs() > chunk['event_surprise'].abs().quantile(0.95) if len(chunk) > 5 else False)
            if isinstance(big_surprise, pd.Series) and big_surprise.sum() > 0:
                should_retrain = True
                retrain_reason = 'event_surprise'

        if should_retrain and bars_since_last_retrain >= RETRAIN_COOLDOWN:
            train_start = max(0, step_end - TRAIN_BARS)
            X_retrain = df.iloc[train_start:step_end][feature_cols].values
            y_retrain = df.iloc[train_start:step_end]['target_market_state'].values
            # Ensure all classes
            unique_r = np.unique(y_retrain)
            for cls in range(5):
                if cls not in unique_r:
                    idx = np.random.randint(0, len(X_retrain))
                    X_retrain = np.vstack([X_retrain, X_retrain[idx:idx+1]])
                    y_retrain = np.append(y_retrain, cls)
            regime_model = xgb.XGBClassifier(**REGIME_PARAMS)
            regime_model.fit(X_retrain, y_retrain, verbose=False)
            regime_retrain_count += 1
            bars_since_last_retrain = 0
            retrain_reasons_log.append((str(chunk['datetime_utc'].iloc[0]), retrain_reason))

        bars_since_last_retrain += STEP_BARS
        step += 1
        cursor = step_end

    # Evaluate all strategies
    records_df = pd.DataFrame(all_records)
    actual = records_df['actual_return'].values

    print(f"\n{'='*60}")
    print(f"RESULTS — {pair_name}")
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
        print(f"    Avg Win:        {r['avg_win']:.6f}")
        print(f"    Avg Loss:       {r['avg_loss']:.6f}")
        print(f"    Profit Factor:  {r['profit_factor']:.2f}")
        print(f"    Total Return:   {r['total_return']:.6f}")

    results['regime_retrains'] = regime_retrain_count
    results['retrain_log'] = retrain_reasons_log

    # Save
    records_df.to_csv(os.path.join(RESULTS_DIR, f'{pair_name}_winrate_trades.csv'), index=False)
    with open(os.path.join(RESULTS_DIR, f'{pair_name}_winrate_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Regime model retrained: {regime_retrain_count} times (cooldown={RETRAIN_COOLDOWN} bars)")
    if retrain_reasons_log:
        reason_counts = {}
        for _, reason in retrain_reasons_log:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        print(f"  Retrain reasons: {reason_counts}")
    print(f"  Saved to results/winrate/")

    return results


if __name__ == '__main__':
    all_results = {}
    for pair in ['EURUSD', 'GBPJPY', 'XAUUSD']:
        results = run_pair(pair)
        all_results[pair] = results

    # Cross-pair summary
    print(f"\n{'='*60}")
    print("WIN RATE EXPERIMENT — Cross-Pair Summary")
    print(f"{'='*60}")
    print(f"\n{'Pair':<10} {'Tech Win%':<12} {'ML Win%':<12} {'SG Win%':<12} {'SG Trades%':<12} {'SG PF'}")
    print("-" * 60)
    for pair, r in all_results.items():
        tech = r['Technical (RSI/MACD)']
        ml = r['ML Direction (XGBoost)']
        sg = r['ShiftGuard (Regime-Filtered)']
        print(f"{pair:<10} {tech['win_rate']:<12.1f} {ml['win_rate']:<12.1f} "
              f"{sg['win_rate']:<12.1f} {sg['trade_pct']:<12.1f} {sg['profit_factor']:.2f}")

    with open(os.path.join(RESULTS_DIR, 'winrate_overall.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: results/winrate/winrate_overall.json")
