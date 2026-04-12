"""
Walk-Forward Retraining Experiment
Train on 6-month rolling window. Step forward 1 month at a time.
Compare:
  A) No retrain — train once, never update (model goes stale)
  B) Blind monthly — retrain every month regardless
  C) ShiftGuard — retrain only when detection engine flags a non-technical shift
  D) Oracle — retrain only at ground truth geopolitical events

Shows degradation over time without retraining, and recovery with ShiftGuard.

Usage:
    python src/retraining/walkforward.py
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
DETECTION_DIR = os.path.join(PROJECT_ROOT, 'results', 'detection')
ATTRIBUTION_DIR = os.path.join(PROJECT_ROOT, 'results', 'attribution')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
RETRAINING_DIR = os.path.join(PROJECT_ROOT, 'results', 'retraining')
os.makedirs(RETRAINING_DIR, exist_ok=True)

EXCLUDE_COLS = ['datetime_utc', 'date', 'session', 'target_return', 'target_direction', 'volume', 'month']

PARAMS = {
    'learning_rate': 0.05,
    'max_depth': 4,
    'n_estimators': 200,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'tree_method': 'hist',
    'random_state': 42,
    'verbosity': 0,
}

TRAIN_BARS = 180 * 6    # 6 months (~1080 bars)
STEP_BARS = 180         # 1 month step (~180 bars)
ROLLING_W = 60          # rolling MAE window for smoothing


def get_feature_cols(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def train_xgb(df, feature_cols, start, end):
    """Train XGBoost on df[start:end]."""
    chunk = df.iloc[start:end]
    X = chunk[feature_cols].values
    y = chunk['target_return'].values
    model = xgb.XGBRegressor(**PARAMS)
    model.fit(X, y, verbose=False)
    return model


def load_shift_dates(pair_name):
    """Load detected shifts with SHAP attribution. Return set of dates where non-technical shift occurred."""
    shifts = pd.read_csv(os.path.join(DETECTION_DIR, f'{pair_name}_shifts.csv'))
    shifts['datetime_utc'] = pd.to_datetime(shifts['datetime_utc'])

    attr_path = os.path.join(ATTRIBUTION_DIR, f'{pair_name}_attribution.csv')
    if os.path.exists(attr_path):
        attr = pd.read_csv(attr_path)
        attr['datetime_utc'] = pd.to_datetime(attr['datetime_utc'])
        shifts = shifts.merge(
            attr[['datetime_utc', 'dominant_group']].drop_duplicates(),
            on='datetime_utc', how='left'
        )
    else:
        shifts['dominant_group'] = 'unknown'

    shifts['dominant_group'] = shifts['dominant_group'].fillna('unknown')

    # Non-technical shifts (these warrant retraining)
    non_tech = shifts[shifts['dominant_group'].isin(['macro', 'sentiment', 'volatility', 'other', 'unknown'])]
    # Group by month — if any non-technical shift in a month, flag that month
    non_tech_months = set(non_tech['datetime_utc'].dt.to_period('M').astype(str))

    # ALL shifts (for blind strategy)
    all_months = set(shifts['datetime_utc'].dt.to_period('M').astype(str))

    return all_months, non_tech_months


def load_geopolitical_months():
    """Load ground truth event months."""
    path = os.path.join(DATA_DIR, 'events', 'geopolitical_events.csv')
    events = pd.read_csv(path)
    events['date'] = pd.to_datetime(events['date'])
    return set(events['date'].dt.to_period('M').astype(str))


def run_walkforward(pair_name):
    print(f"\n{'='*60}")
    print(f"Walk-Forward Experiment — {pair_name}")
    print(f"{'='*60}")

    df = pd.read_csv(os.path.join(PROCESSED_DIR, f'{pair_name}_features.csv'))
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    df['month'] = df['datetime_utc'].dt.to_period('M').astype(str)
    feature_cols = get_feature_cols(df)

    # Load shift info
    all_shift_months, shap_shift_months = load_shift_dates(pair_name)
    geo_months = load_geopolitical_months()

    print(f"  Bars: {len(df)}, Features: {len(feature_cols)}")
    print(f"  Months with any shift: {len(all_shift_months)}")
    print(f"  Months with non-tech shift: {len(shap_shift_months)}")
    print(f"  Months with geopolitical event: {len(geo_months)}")

    # Start walk-forward from bar TRAIN_BARS
    cursor = TRAIN_BARS
    total = len(df)

    # Train initial model (same for all strategies)
    model_none = train_xgb(df, feature_cols, 0, TRAIN_BARS)
    model_blind = train_xgb(df, feature_cols, 0, TRAIN_BARS)
    model_shap = train_xgb(df, feature_cols, 0, TRAIN_BARS)
    model_oracle = train_xgb(df, feature_cols, 0, TRAIN_BARS)

    # Storage
    records = []
    retrain_log = {'none': 0, 'blind_monthly': 0, 'shiftguard': 0, 'oracle': 0}
    step_num = 0

    while cursor + STEP_BARS <= total:
        step_end = cursor + STEP_BARS
        chunk = df.iloc[cursor:step_end]
        X = chunk[feature_cols].values
        y = chunk['target_return'].values
        current_month = df['month'].iloc[cursor]

        # Predict with each model
        pred_none = model_none.predict(X)
        pred_blind = model_blind.predict(X)
        pred_shap = model_shap.predict(X)
        pred_oracle = model_oracle.predict(X)

        # Per-bar records
        for i in range(len(y)):
            records.append({
                'datetime_utc': chunk['datetime_utc'].iloc[i],
                'actual': y[i],
                'pred_none': pred_none[i],
                'pred_blind': pred_blind[i],
                'pred_shap': pred_shap[i],
                'pred_oracle': pred_oracle[i],
                'month': current_month,
            })

        # --- Retraining decisions ---
        train_start = max(0, step_end - TRAIN_BARS)

        # B) Blind monthly — always retrain
        model_blind = train_xgb(df, feature_cols, train_start, step_end)
        retrain_log['blind_monthly'] += 1

        # C) ShiftGuard — retrain only if non-technical shift this month
        if current_month in shap_shift_months:
            model_shap = train_xgb(df, feature_cols, train_start, step_end)
            retrain_log['shiftguard'] += 1

        # D) Oracle — retrain only at geopolitical events
        if current_month in geo_months:
            model_oracle = train_xgb(df, feature_cols, train_start, step_end)
            retrain_log['oracle'] += 1

        # A) No retrain — never updates

        step_num += 1
        cursor = step_end

        if step_num % 10 == 0:
            # Quick progress
            mae_n = mean_absolute_error(y, pred_none)
            mae_s = mean_absolute_error(y, pred_shap)
            print(f"  Step {step_num} | {current_month} | No:{mae_n:.5f} ShiftGuard:{mae_s:.5f}")

    # Build results dataframe
    results_df = pd.DataFrame(records)

    # Compute rolling MAE
    for strategy in ['none', 'blind', 'shap', 'oracle']:
        col = f'pred_{strategy}'
        results_df[f'error_{strategy}'] = np.abs(results_df['actual'] - results_df[col])
        results_df[f'rolling_mae_{strategy}'] = results_df[f'error_{strategy}'].rolling(ROLLING_W).mean()
        results_df[f'dir_correct_{strategy}'] = (
            np.sign(results_df['actual']) == np.sign(results_df[col])
        ).astype(int)

    # Monthly aggregation
    monthly = results_df.groupby('month').agg({
        'error_none': 'mean',
        'error_blind': 'mean',
        'error_shap': 'mean',
        'error_oracle': 'mean',
        'dir_correct_none': 'mean',
        'dir_correct_blind': 'mean',
        'dir_correct_shap': 'mean',
        'dir_correct_oracle': 'mean',
    }).reset_index()

    # Overall summary
    print(f"\n{'='*60}")
    print(f"RESULTS — {pair_name}")
    print(f"{'='*60}")

    summary = {}
    retrain_key_map = {'none': 'none', 'blind': 'blind_monthly', 'shap': 'shiftguard', 'oracle': 'oracle'}
    for strategy, col in [('No Retrain', 'none'), ('Blind Monthly', 'blind'),
                           ('ShiftGuard', 'shap'), ('Oracle', 'oracle')]:
        mae = results_df[f'error_{col}'].mean()
        dir_acc = results_df[f'dir_correct_{col}'].mean()
        retrains = retrain_log[retrain_key_map[col]]
        improvement = (results_df['error_none'].mean() - mae) / results_df['error_none'].mean() * 100

        print(f"  {strategy:<18s} MAE: {mae:.6f}  Dir Acc: {dir_acc:.4f}  "
              f"Retrains: {retrains:<4d}  vs No-Retrain: {improvement:+.2f}%")

        summary[strategy] = {
            'mae': round(mae, 6),
            'dir_acc': round(dir_acc, 4),
            'retrains': retrains,
            'improvement_vs_none': round(improvement, 2),
        }

    # Efficiency
    print(f"\n  Efficiency:")
    for s in ['Blind Monthly', 'ShiftGuard', 'Oracle']:
        imp = summary[s]['improvement_vs_none']
        ret = summary[s]['retrains']
        eff = imp / ret if ret > 0 else 0
        print(f"    {s:<18s} {imp:+.2f}% improvement / {ret} retrains = {eff:.3f}% per cycle")
        summary[s]['efficiency'] = round(eff, 4)

    # Save
    results_df.to_csv(os.path.join(RETRAINING_DIR, f'{pair_name}_walkforward_bars.csv'), index=False)
    monthly.to_csv(os.path.join(RETRAINING_DIR, f'{pair_name}_walkforward_monthly.csv'), index=False)

    with open(os.path.join(RETRAINING_DIR, f'{pair_name}_walkforward_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved to results/retraining/")
    return summary


if __name__ == '__main__':
    all_summaries = {}
    for pair in ['EURUSD', 'GBPJPY', 'XAUUSD']:
        summary = run_walkforward(pair)
        all_summaries[pair] = summary

    # Cross-pair summary
    print(f"\n{'='*60}")
    print("WALK-FORWARD — Cross-Pair Summary")
    print(f"{'='*60}")
    print(f"\n{'Pair':<10} {'No Retrain MAE':<16} {'ShiftGuard MAE':<16} {'Improvement':<14} {'Retrains Saved'}")
    print("-" * 70)
    for pair, s in all_summaries.items():
        none_mae = s['No Retrain']['mae']
        sg_mae = s['ShiftGuard']['mae']
        imp = s['ShiftGuard']['improvement_vs_none']
        blind_ret = s['Blind Monthly']['retrains']
        sg_ret = s['ShiftGuard']['retrains']
        saved = blind_ret - sg_ret
        print(f"{pair:<10} {none_mae:<16.6f} {sg_mae:<16.6f} {imp:+.2f}%         {saved} fewer")

    with open(os.path.join(RETRAINING_DIR, 'walkforward_overall.json'), 'w') as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved: results/retraining/walkforward_overall.json")
