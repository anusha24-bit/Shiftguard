"""
Phase 7: Selective Retraining
Compares 3 retraining strategies after confirmed shifts:
  A) No retraining (baseline — model degrades)
  B) Full retrain (all historical data up to shift)
  C) Window retrain (last N bars only)
  D) Weighted retrain (exponential decay, upweight recent)

Measures recovery: rolling MAE → how many bars until model returns to pre-shift error.

Usage:
    python src/retraining/selective.py
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
RETRAINING_DIR = os.path.join(PROJECT_ROOT, 'results', 'retraining')
os.makedirs(RETRAINING_DIR, exist_ok=True)

EXCLUDE_COLS = ['datetime_utc', 'date', 'session', 'target_return', 'target_direction', 'volume']

# XGBoost default params (from tuning results — use EURUSD best as default)
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

WINDOW_SIZE = 180  # 180 4H bars = 30 trading days
EVAL_HORIZON = 180  # evaluate recovery over 180 bars after retraining
ROLLING_WINDOW = 30  # rolling MAE window


def get_feature_cols(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def get_shift_events(pair_name):
    """Load detected shifts, filter to high-severity + test period only."""
    shifts_path = os.path.join(DETECTION_DIR, f'{pair_name}_shifts.csv')
    shifts = pd.read_csv(shifts_path)
    shifts['datetime_utc'] = pd.to_datetime(shifts['datetime_utc'])

    # Only test period (2021+)
    shifts = shifts[shifts['datetime_utc'] >= '2021-01-01']

    # Only high severity (3+) to keep it manageable
    if 'severity' in shifts.columns:
        shifts = shifts[shifts['severity'] >= 3]

    # Deduplicate: keep one shift per week (avoid overlapping retraining windows)
    shifts = shifts.sort_values('datetime_utc')
    filtered = []
    last_dt = None
    for _, row in shifts.iterrows():
        if last_dt is None or (row['datetime_utc'] - last_dt).days >= 7:
            filtered.append(row)
            last_dt = row['datetime_utc']

    return pd.DataFrame(filtered)


def retrain_no_update(model, X_test_chunk, y_test_chunk):
    """Strategy A: No retraining — use existing model as-is."""
    return model.predict(X_test_chunk)


def retrain_full(df, feature_cols, shift_idx, params):
    """Strategy B: Full retrain on all data up to shift point."""
    train_data = df.iloc[:shift_idx]
    X = train_data[feature_cols].values
    y = train_data['target_return'].values

    model = xgb.XGBRegressor(**params)
    model.fit(X, y, verbose=False)
    return model


def retrain_window(df, feature_cols, shift_idx, window_size, params):
    """Strategy C: Window retrain — only last N bars before shift."""
    start = max(0, shift_idx - window_size)
    train_data = df.iloc[start:shift_idx]
    X = train_data[feature_cols].values
    y = train_data['target_return'].values

    model = xgb.XGBRegressor(**params)
    model.fit(X, y, verbose=False)
    return model


def retrain_weighted(df, feature_cols, shift_idx, params, decay=0.995):
    """Strategy D: Weighted retrain — exponential decay, recent data upweighted."""
    train_data = df.iloc[:shift_idx]
    X = train_data[feature_cols].values
    y = train_data['target_return'].values

    # Exponential weights: most recent = 1.0, decays backwards
    n = len(X)
    weights = np.array([decay ** (n - 1 - i) for i in range(n)])

    model = xgb.XGBRegressor(**params)
    model.fit(X, y, sample_weight=weights, verbose=False)
    return model


def compute_recovery_time(rolling_mae, pre_shift_mae, threshold=1.1):
    """
    How many bars until rolling MAE returns to within threshold × pre-shift level.
    Returns number of bars, or -1 if never recovers.
    """
    target = pre_shift_mae * threshold
    for i, val in enumerate(rolling_mae):
        if pd.notna(val) and val <= target:
            return i
    return -1


def run_retraining_experiment(pair_name):
    """Run all 4 strategies on detected shifts for one pair."""
    print(f"\n{'='*60}")
    print(f"Selective Retraining — {pair_name}")
    print(f"{'='*60}")

    # Load data
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f'{pair_name}_features.csv'))
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    feature_cols = get_feature_cols(df)

    # Load base model
    model_path = os.path.join(RESULTS_DIR, f'xgboost_{pair_name}.json')
    base_model = xgb.XGBRegressor()
    base_model.load_model(model_path)

    # Get shift events
    shifts = get_shift_events(pair_name)
    print(f"  Shifts to process: {len(shifts)}")

    if len(shifts) == 0:
        print("  No high-severity shifts in test period. Skipping.")
        return None

    all_results = []

    for shift_num, (_, shift_row) in enumerate(shifts.iterrows()):
        shift_dt = shift_row['datetime_utc']

        # Find shift index in df
        time_diff = (df['datetime_utc'] - shift_dt).abs()
        shift_idx = time_diff.idxmin()

        # Evaluation window: EVAL_HORIZON bars after shift
        eval_end = min(len(df), shift_idx + EVAL_HORIZON)
        if eval_end - shift_idx < 30:
            continue

        eval_data = df.iloc[shift_idx:eval_end]
        X_eval = eval_data[feature_cols].values
        y_eval = eval_data['target_return'].values

        # Pre-shift baseline MAE (30 bars before shift)
        pre_start = max(0, shift_idx - ROLLING_WINDOW)
        pre_data = df.iloc[pre_start:shift_idx]
        X_pre = pre_data[feature_cols].values
        y_pre = pre_data['target_return'].values
        pre_pred = base_model.predict(X_pre)
        pre_shift_mae = mean_absolute_error(y_pre, pre_pred)

        # --- Strategy A: No retraining ---
        pred_none = base_model.predict(X_eval)
        mae_none = mean_absolute_error(y_eval, pred_none)
        rolling_none = pd.Series(np.abs(y_eval - pred_none)).rolling(ROLLING_WINDOW).mean()
        recovery_none = compute_recovery_time(rolling_none.values, pre_shift_mae)

        # --- Strategy B: Full retrain ---
        model_full = retrain_full(df, feature_cols, shift_idx, DEFAULT_PARAMS)
        pred_full = model_full.predict(X_eval)
        mae_full = mean_absolute_error(y_eval, pred_full)
        rolling_full = pd.Series(np.abs(y_eval - pred_full)).rolling(ROLLING_WINDOW).mean()
        recovery_full = compute_recovery_time(rolling_full.values, pre_shift_mae)

        # --- Strategy C: Window retrain (30 days) ---
        model_window = retrain_window(df, feature_cols, shift_idx, WINDOW_SIZE, DEFAULT_PARAMS)
        pred_window = model_window.predict(X_eval)
        mae_window = mean_absolute_error(y_eval, pred_window)
        rolling_window = pd.Series(np.abs(y_eval - pred_window)).rolling(ROLLING_WINDOW).mean()
        recovery_window = compute_recovery_time(rolling_window.values, pre_shift_mae)

        # --- Strategy D: Weighted retrain ---
        model_weighted = retrain_weighted(df, feature_cols, shift_idx, DEFAULT_PARAMS)
        pred_weighted = model_weighted.predict(X_eval)
        mae_weighted = mean_absolute_error(y_eval, pred_weighted)
        rolling_weighted = pd.Series(np.abs(y_eval - pred_weighted)).rolling(ROLLING_WINDOW).mean()
        recovery_weighted = compute_recovery_time(rolling_weighted.values, pre_shift_mae)

        result = {
            'shift_datetime': str(shift_dt),
            'shift_type': shift_row.get('type', 'unknown'),
            'pre_shift_mae': round(pre_shift_mae, 6),
            # No retrain
            'mae_no_retrain': round(mae_none, 6),
            'recovery_no_retrain': recovery_none,
            # Full
            'mae_full_retrain': round(mae_full, 6),
            'recovery_full_retrain': recovery_full,
            # Window
            'mae_window_retrain': round(mae_window, 6),
            'recovery_window_retrain': recovery_window,
            # Weighted
            'mae_weighted_retrain': round(mae_weighted, 6),
            'recovery_weighted_retrain': recovery_weighted,
        }
        all_results.append(result)

        if (shift_num + 1) % 5 == 0 or shift_num == 0:
            print(f"  [{shift_num+1}/{len(shifts)}] {shift_dt.date()} | "
                  f"No:{mae_none:.5f} Full:{mae_full:.5f} "
                  f"Win:{mae_window:.5f} Wgt:{mae_weighted:.5f}")

    results_df = pd.DataFrame(all_results)

    # Summary
    print(f"\n  STRATEGY COMPARISON (mean across {len(results_df)} shifts):")
    print(f"  {'Strategy':<20s} {'MAE':<12s} {'Avg Recovery (bars)':<20s}")
    print(f"  {'-'*52}")

    strategies = [
        ('No Retrain', 'mae_no_retrain', 'recovery_no_retrain'),
        ('Full Retrain', 'mae_full_retrain', 'recovery_full_retrain'),
        ('Window (30d)', 'mae_window_retrain', 'recovery_window_retrain'),
        ('Weighted', 'mae_weighted_retrain', 'recovery_weighted_retrain'),
    ]

    summary = {}
    for name, mae_col, rec_col in strategies:
        avg_mae = results_df[mae_col].mean()
        valid_rec = results_df[results_df[rec_col] >= 0][rec_col]
        avg_rec = valid_rec.mean() if len(valid_rec) > 0 else -1
        pct_recovered = (results_df[rec_col] >= 0).mean() * 100
        print(f"  {name:<20s} {avg_mae:<12.6f} {avg_rec:<10.1f} ({pct_recovered:.0f}% recovered)")
        summary[name] = {
            'avg_mae': round(avg_mae, 6),
            'avg_recovery_bars': round(avg_rec, 1),
            'pct_recovered': round(pct_recovered, 1),
        }

    # Save
    out_path = os.path.join(RETRAINING_DIR, f'{pair_name}_retraining_results.csv')
    results_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    json_path = os.path.join(RETRAINING_DIR, f'{pair_name}_retraining_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return results_df, summary


if __name__ == '__main__':
    all_summaries = {}
    for pair in ['EURUSD', 'GBPJPY', 'XAUUSD']:
        result = run_retraining_experiment(pair)
        if result:
            _, summary = result
            all_summaries[pair] = summary

    # Overall summary
    print(f"\n{'='*60}")
    print("SELECTIVE RETRAINING — Overall Summary")
    print(f"{'='*60}")

    overall_path = os.path.join(RETRAINING_DIR, 'retraining_overall_summary.json')
    with open(overall_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    print(f"Saved: {overall_path}")
