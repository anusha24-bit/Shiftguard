"""
Selective Retraining v3 — SHAP-Guided
Uses SHAP attribution to decide whether to retrain:
  - Technical-dominant shift → SKIP (features self-correct)
  - Macro/Sentiment-dominant shift → RETRAIN (persistent regime change)

Compares 4 strategies:
  A) No retrain
  B) Blind retrain (retrain at every detected shift)
  C) SHAP-guided retrain (retrain only when non-technical shift)
  D) Monthly retrain (fixed schedule, no detection needed)

Usage:
    python src/retraining/selective_v3.py
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'predictions')
DETECTION_DIR = os.path.join(PROJECT_ROOT, 'results', 'detection')
ATTRIBUTION_DIR = os.path.join(PROJECT_ROOT, 'results', 'attribution')
RETRAINING_DIR = os.path.join(PROJECT_ROOT, 'results', 'retraining')
os.makedirs(RETRAINING_DIR, exist_ok=True)

EXCLUDE_COLS = ['datetime_utc', 'date', 'session', 'target_return', 'target_direction', 'volume']

DEFAULT_PARAMS = {
    'learning_rate': 0.01,
    'max_depth': 5,
    'n_estimators': 500,
    'reg_alpha': 0.1,
    'reg_lambda': 5.0,
    'tree_method': 'hist',
    'random_state': 42,
    'verbosity': 0,
}

TRAIN_WINDOW = 180 * 6  # 6 months of 4H bars (~180 trading days × 6 bars)
EVAL_STEP = 180         # predict 30 trading days forward before next decision
ROLLING_W = 30
MONTHLY_BARS = 180      # ~30 trading days × 6 bars


def get_feature_cols(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def load_shifts_with_attribution(pair_name):
    """Load detected shifts + SHAP attribution, merged by datetime."""
    shifts = pd.read_csv(os.path.join(DETECTION_DIR, f'{pair_name}_shifts.csv'))
    shifts['datetime_utc'] = pd.to_datetime(shifts['datetime_utc'])

    attr_path = os.path.join(ATTRIBUTION_DIR, f'{pair_name}_attribution.csv')
    if os.path.exists(attr_path):
        attr = pd.read_csv(attr_path)
        attr['datetime_utc'] = pd.to_datetime(attr['datetime_utc'])
        # Merge dominant group onto shifts
        shifts = shifts.merge(
            attr[['datetime_utc', 'dominant_group']].drop_duplicates(),
            on='datetime_utc', how='left'
        )
    else:
        shifts['dominant_group'] = 'unknown'

    shifts['dominant_group'] = shifts['dominant_group'].fillna('unknown')
    return shifts


def should_retrain_shap(dominant_group):
    """SHAP-guided decision: retrain only if non-technical shift."""
    return dominant_group in ['macro', 'sentiment', 'volatility', 'other', 'unknown']


def train_model(df, feature_cols, start_idx, end_idx, params):
    """Train XGBoost on a slice of data."""
    data = df.iloc[start_idx:end_idx]
    X = data[feature_cols].values
    y = data['target_return'].values
    model = xgb.XGBRegressor(**params)
    model.fit(X, y, verbose=False)
    return model


def walk_forward_evaluate(pair_name):
    """
    Walk-forward evaluation comparing 4 retraining strategies.
    Steps through the test period in EVAL_STEP chunks.
    """
    print(f"\n{'='*60}")
    print(f"SHAP-Guided Retraining — {pair_name}")
    print(f"{'='*60}")

    df = pd.read_csv(os.path.join(PROCESSED_DIR, f'{pair_name}_features.csv'))
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    feature_cols = get_feature_cols(df)

    shifts = load_shifts_with_attribution(pair_name)

    # Test period starts at 2021
    test_start = df[df['datetime_utc'] >= '2021-01-01'].index[0]
    total_bars = len(df)

    print(f"  Total bars: {total_bars}, Test starts at index: {test_start}")
    print(f"  Shifts with attribution: {len(shifts)}")

    # Initialize models for each strategy
    initial_train_end = test_start
    initial_train_start = max(0, initial_train_end - TRAIN_WINDOW)

    model_none = train_model(df, feature_cols, initial_train_start, initial_train_end, DEFAULT_PARAMS)
    model_blind = train_model(df, feature_cols, initial_train_start, initial_train_end, DEFAULT_PARAMS)
    model_shap = train_model(df, feature_cols, initial_train_start, initial_train_end, DEFAULT_PARAMS)
    model_monthly = train_model(df, feature_cols, initial_train_start, initial_train_end, DEFAULT_PARAMS)

    # Track predictions and retraining events
    all_preds = {'none': [], 'blind': [], 'shap_guided': [], 'monthly': []}
    all_actuals = []
    all_dates = []
    retrain_counts = {'none': 0, 'blind': 0, 'shap_guided': 0, 'monthly': 0}

    last_monthly_retrain = test_start
    cursor = test_start

    while cursor < total_bars - 1:
        chunk_end = min(cursor + EVAL_STEP, total_bars)
        chunk = df.iloc[cursor:chunk_end]
        X_chunk = chunk[feature_cols].values
        y_chunk = chunk['target_return'].values
        dates_chunk = chunk['datetime_utc'].values

        # Predict with all 4 models
        all_preds['none'].extend(model_none.predict(X_chunk))
        all_preds['blind'].extend(model_blind.predict(X_chunk))
        all_preds['shap_guided'].extend(model_shap.predict(X_chunk))
        all_preds['monthly'].extend(model_monthly.predict(X_chunk))
        all_actuals.extend(y_chunk)
        all_dates.extend(dates_chunk)

        # Check for shifts in this chunk
        chunk_start_dt = df['datetime_utc'].iloc[cursor]
        chunk_end_dt = df['datetime_utc'].iloc[chunk_end - 1]
        chunk_shifts = shifts[
            (shifts['datetime_utc'] >= chunk_start_dt) &
            (shifts['datetime_utc'] <= chunk_end_dt)
        ]

        has_shift = len(chunk_shifts) > 0

        if has_shift:
            # Strategy B: Blind — always retrain
            train_start = max(0, chunk_end - TRAIN_WINDOW)
            model_blind = train_model(df, feature_cols, train_start, chunk_end, DEFAULT_PARAMS)
            retrain_counts['blind'] += 1

            # Strategy C: SHAP-guided — retrain only if non-technical
            dominant_groups = chunk_shifts['dominant_group'].value_counts()
            top_group = dominant_groups.index[0] if len(dominant_groups) > 0 else 'unknown'

            if should_retrain_shap(top_group):
                model_shap = train_model(df, feature_cols, train_start, chunk_end, DEFAULT_PARAMS)
                retrain_counts['shap_guided'] += 1

        # Strategy D: Monthly — retrain every MONTHLY_BARS
        if cursor - last_monthly_retrain >= MONTHLY_BARS:
            train_start = max(0, cursor - TRAIN_WINDOW)
            model_monthly = train_model(df, feature_cols, train_start, cursor, DEFAULT_PARAMS)
            retrain_counts['monthly'] += 1
            last_monthly_retrain = cursor

        cursor = chunk_end

    # Compute metrics
    actuals = np.array(all_actuals)
    results = {}

    print(f"\n  {'Strategy':<20s} {'MAE':<12s} {'Dir Acc':<10s} {'Retrains':<10s}")
    print(f"  {'-'*52}")

    for strategy in ['none', 'blind', 'shap_guided', 'monthly']:
        preds = np.array(all_preds[strategy])
        mae = mean_absolute_error(actuals, preds)
        dir_acc = np.mean(np.sign(actuals) == np.sign(preds))
        n_retrains = retrain_counts[strategy]

        print(f"  {strategy:<20s} {mae:<12.6f} {dir_acc:<10.4f} {n_retrains:<10d}")

        # Rolling MAE
        rolling = pd.Series(np.abs(actuals - preds)).rolling(ROLLING_W).mean()

        results[strategy] = {
            'mae': round(mae, 6),
            'dir_acc': round(dir_acc, 4),
            'retrains': n_retrains,
            'rolling_mae_mean': round(rolling.mean(), 6),
            'rolling_mae_max': round(rolling.max(), 6),
        }

    # Efficiency: MAE per retrain cycle
    print(f"\n  Efficiency (lower MAE with fewer retrains = better):")
    for s in ['blind', 'shap_guided', 'monthly']:
        r = results[s]
        improvement = (results['none']['mae'] - r['mae']) / results['none']['mae'] * 100
        cycles = r['retrains']
        print(f"    {s:<20s} {improvement:+.2f}% improvement in {cycles} retrains")
        results[s]['improvement_pct'] = round(improvement, 2)

    # Save
    out_path = os.path.join(RETRAINING_DIR, f'{pair_name}_shap_guided_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")

    # Save rolling MAE for plotting
    rolling_df = pd.DataFrame({'datetime_utc': all_dates})
    for strategy in ['none', 'blind', 'shap_guided', 'monthly']:
        preds = np.array(all_preds[strategy])
        rolling_df[f'rolling_mae_{strategy}'] = pd.Series(np.abs(actuals - preds)).rolling(ROLLING_W).mean().values

    rolling_path = os.path.join(RETRAINING_DIR, f'{pair_name}_shap_guided_rolling.csv')
    rolling_df.to_csv(rolling_path, index=False)

    return results


if __name__ == '__main__':
    all_results = {}
    for pair in ['EURUSD', 'GBPJPY', 'XAUUSD']:
        results = walk_forward_evaluate(pair)
        all_results[pair] = results

    # Overall
    print(f"\n{'='*60}")
    print("SHAP-GUIDED RETRAINING — Overall")
    print(f"{'='*60}")

    for pair, res in all_results.items():
        print(f"\n  {pair}:")
        shap_r = res['shap_guided']
        blind_r = res['blind']
        print(f"    SHAP-guided: MAE={shap_r['mae']}, {shap_r['retrains']} retrains, {shap_r.get('improvement_pct', 0):+.2f}%")
        print(f"    Blind:       MAE={blind_r['mae']}, {blind_r['retrains']} retrains, {blind_r.get('improvement_pct', 0):+.2f}%")
        if blind_r['retrains'] > 0 and shap_r['retrains'] > 0:
            cycle_reduction = (1 - shap_r['retrains'] / blind_r['retrains']) * 100
            print(f"    → SHAP-guided uses {cycle_reduction:.0f}% fewer retraining cycles")

    overall_path = os.path.join(RETRAINING_DIR, 'shap_guided_overall.json')
    with open(overall_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {overall_path}")
