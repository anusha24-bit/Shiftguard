"""
Selective Retraining v2 — Major Shifts Only
Only retrains at ground truth geopolitical events + highest severity unexpected shifts.
These are the regime changes that actually break models.

Usage:
    python src/retraining/selective_v2.py
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
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
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

EVAL_HORIZON = 360   # 360 bars = 60 trading days (10 weeks) after shift
PRE_WINDOW = 60      # 60 bars before shift for baseline MAE
ROLLING_W = 30       # rolling MAE window
RETRAIN_WINDOW = 180  # 30 trading days for window strategy


def get_feature_cols(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def get_major_events():
    """Load ground truth geopolitical events — the real regime breakers."""
    path = os.path.join(DATA_DIR, 'events', 'geopolitical_events.csv')
    events = pd.read_csv(path)
    events['date'] = pd.to_datetime(events['date'])
    # Only events in test period (2021+) with severity >= 4
    events = events[(events['date'] >= '2020-01-01') & (events['severity'] >= 4)]
    return events


def retrain_full(df, feature_cols, shift_idx, params):
    X = df.iloc[:shift_idx][feature_cols].values
    y = df.iloc[:shift_idx]['target_return'].values
    model = xgb.XGBRegressor(**params)
    model.fit(X, y, verbose=False)
    return model


def retrain_window(df, feature_cols, shift_idx, window, params):
    start = max(0, shift_idx - window)
    X = df.iloc[start:shift_idx][feature_cols].values
    y = df.iloc[start:shift_idx]['target_return'].values
    model = xgb.XGBRegressor(**params)
    model.fit(X, y, verbose=False)
    return model


def retrain_weighted(df, feature_cols, shift_idx, params, decay=0.995):
    X = df.iloc[:shift_idx][feature_cols].values
    y = df.iloc[:shift_idx]['target_return'].values
    n = len(X)
    weights = np.array([decay ** (n - 1 - i) for i in range(n)])
    model = xgb.XGBRegressor(**params)
    model.fit(X, y, sample_weight=weights, verbose=False)
    return model


def compute_rolling_mae(y_true, y_pred, window):
    return pd.Series(np.abs(y_true - y_pred)).rolling(window).mean()


def recovery_time(rolling_mae, baseline_mae, threshold=1.2):
    """Bars until MAE returns to within threshold × baseline. -1 if never."""
    target = baseline_mae * threshold
    for i, val in enumerate(rolling_mae):
        if pd.notna(val) and val <= target:
            return i
    return -1


def run_major_retraining(pair_name):
    print(f"\n{'='*60}")
    print(f"Major Shift Retraining — {pair_name}")
    print(f"{'='*60}")

    df = pd.read_csv(os.path.join(PROCESSED_DIR, f'{pair_name}_features.csv'))
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    feature_cols = get_feature_cols(df)

    # Base model (trained on 2015-2019)
    model_path = os.path.join(RESULTS_DIR, f'xgboost_{pair_name}.json')
    base_model = xgb.XGBRegressor()
    base_model.load_model(model_path)

    events = get_major_events()
    print(f"  Major events: {len(events)}")

    results = []

    for _, event in events.iterrows():
        event_date = event['date']
        event_name = event['event_name']

        # Find closest bar
        time_diff = (df['datetime_utc'] - event_date).abs()
        shift_idx = time_diff.idxmin()

        # Need enough data before and after
        if shift_idx < PRE_WINDOW or shift_idx + EVAL_HORIZON > len(df):
            continue

        # Pre-shift baseline
        pre_data = df.iloc[shift_idx - PRE_WINDOW:shift_idx]
        X_pre = pre_data[feature_cols].values
        y_pre = pre_data['target_return'].values
        pre_pred = base_model.predict(X_pre)
        baseline_mae = mean_absolute_error(y_pre, pre_pred)

        # Post-shift evaluation
        post_data = df.iloc[shift_idx:shift_idx + EVAL_HORIZON]
        X_post = post_data[feature_cols].values
        y_post = post_data['target_return'].values

        # Strategy A: No retrain
        pred_none = base_model.predict(X_post)
        mae_none = mean_absolute_error(y_post, pred_none)
        roll_none = compute_rolling_mae(y_post, pred_none, ROLLING_W)
        rec_none = recovery_time(roll_none.values, baseline_mae)
        # MAE spike: max rolling MAE / baseline
        spike_none = roll_none.max() / baseline_mae if baseline_mae > 0 else 0

        # Strategy B: Full retrain
        model_full = retrain_full(df, feature_cols, shift_idx, DEFAULT_PARAMS)
        pred_full = model_full.predict(X_post)
        mae_full = mean_absolute_error(y_post, pred_full)
        roll_full = compute_rolling_mae(y_post, pred_full, ROLLING_W)
        rec_full = recovery_time(roll_full.values, baseline_mae)
        spike_full = roll_full.max() / baseline_mae if baseline_mae > 0 else 0

        # Strategy C: Window retrain
        model_win = retrain_window(df, feature_cols, shift_idx, RETRAIN_WINDOW, DEFAULT_PARAMS)
        pred_win = model_win.predict(X_post)
        mae_win = mean_absolute_error(y_post, pred_win)
        roll_win = compute_rolling_mae(y_post, pred_win, ROLLING_W)
        rec_win = recovery_time(roll_win.values, baseline_mae)
        spike_win = roll_win.max() / baseline_mae if baseline_mae > 0 else 0

        # Strategy D: Weighted retrain
        model_wgt = retrain_weighted(df, feature_cols, shift_idx, DEFAULT_PARAMS)
        pred_wgt = model_wgt.predict(X_post)
        mae_wgt = mean_absolute_error(y_post, pred_wgt)
        roll_wgt = compute_rolling_mae(y_post, pred_wgt, ROLLING_W)
        rec_wgt = recovery_time(roll_wgt.values, baseline_mae)
        spike_wgt = roll_wgt.max() / baseline_mae if baseline_mae > 0 else 0

        # MAE improvement %
        imp_full = (mae_none - mae_full) / mae_none * 100 if mae_none > 0 else 0
        imp_win = (mae_none - mae_win) / mae_none * 100 if mae_none > 0 else 0
        imp_wgt = (mae_none - mae_wgt) / mae_none * 100 if mae_none > 0 else 0

        result = {
            'event_date': str(event_date.date()),
            'event_name': event_name,
            'severity': event['severity'],
            'baseline_mae': round(baseline_mae, 6),
            # No retrain
            'mae_no_retrain': round(mae_none, 6),
            'recovery_no_retrain': rec_none,
            'spike_no_retrain': round(spike_none, 2),
            # Full
            'mae_full': round(mae_full, 6),
            'recovery_full': rec_full,
            'spike_full': round(spike_full, 2),
            'improvement_full_pct': round(imp_full, 2),
            # Window
            'mae_window': round(mae_win, 6),
            'recovery_window': rec_win,
            'improvement_window_pct': round(imp_win, 2),
            # Weighted
            'mae_weighted': round(mae_wgt, 6),
            'recovery_weighted': rec_wgt,
            'improvement_weighted_pct': round(imp_wgt, 2),
        }
        results.append(result)

        print(f"  {event_date.date()} | {event_name[:40]:<40s} | "
              f"No:{mae_none:.5f} Full:{mae_full:.5f} ({imp_full:+.1f}%) "
              f"Rec: {rec_none}→{rec_full} bars")

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("  No events processed.")
        return None, None

    # Summary
    print(f"\n  {'='*60}")
    print(f"  MAJOR SHIFT RETRAINING SUMMARY — {pair_name}")
    print(f"  {len(results_df)} events analyzed")
    print(f"  {'='*60}")

    print(f"\n  {'Strategy':<18s} {'Avg MAE':<12s} {'Avg Recovery':<14s} {'Avg Spike':<12s} {'Avg Improvement'}")
    print(f"  {'-'*70}")

    summary = {}
    for name, mae_col, rec_col in [
        ('No Retrain', 'mae_no_retrain', 'recovery_no_retrain'),
        ('Full Retrain', 'mae_full', 'recovery_full'),
        ('Window (30d)', 'mae_window', 'recovery_window'),
        ('Weighted', 'mae_weighted', 'recovery_weighted'),
    ]:
        avg_mae = results_df[mae_col].mean()
        valid_rec = results_df[results_df[rec_col] >= 0][rec_col]
        avg_rec = valid_rec.mean() if len(valid_rec) > 0 else -1
        pct_rec = (results_df[rec_col] >= 0).mean() * 100

        imp_col = f'improvement_{name.lower().split()[0]}_pct' if name != 'No Retrain' else None
        avg_imp = results_df[imp_col].mean() if imp_col and imp_col in results_df.columns else 0

        print(f"  {name:<18s} {avg_mae:<12.6f} {avg_rec:<8.1f} bars   {pct_rec:.0f}% rec    {avg_imp:+.2f}%")

        summary[name] = {
            'avg_mae': round(avg_mae, 6),
            'avg_recovery_bars': round(avg_rec, 1),
            'pct_recovered': round(pct_rec, 1),
        }

    # Save
    out_path = os.path.join(RETRAINING_DIR, f'{pair_name}_major_retraining.csv')
    results_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    json_path = os.path.join(RETRAINING_DIR, f'{pair_name}_major_retraining_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return results_df, summary


if __name__ == '__main__':
    all_summaries = {}
    for pair in ['EURUSD', 'GBPJPY', 'XAUUSD']:
        result = run_major_retraining(pair)
        if result:
            _, summary = result
            all_summaries[pair] = summary

    overall_path = os.path.join(RETRAINING_DIR, 'major_retraining_overall.json')
    with open(overall_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Major shift retraining complete.")
    print(f"{'='*60}")
