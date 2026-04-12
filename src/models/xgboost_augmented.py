"""
Augmented XGBoost — Detection signals fed back as features.
The model now KNOWS when a shift was detected, what type, and severity.

Compares:
  A) Base XGBoost (original features only)
  B) Augmented XGBoost (original features + detection signals)

Walk-forward: 6-month rolling window, monthly steps.

Usage:
    python src/models/xgboost_augmented.py
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.metrics import precision_recall_fscore_support

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
DETECTION_DIR = os.path.join(PROJECT_ROOT, 'results', 'detection')
ATTRIBUTION_DIR = os.path.join(PROJECT_ROOT, 'results', 'attribution')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'predictions')
RETRAINING_DIR = os.path.join(PROJECT_ROOT, 'results', 'retraining')
os.makedirs(RETRAINING_DIR, exist_ok=True)

EXCLUDE_COLS = ['datetime_utc', 'date', 'session', 'target_return', 'target_direction', 'volume', 'month']

PARAMS = {
    'learning_rate': 0.05,
    'max_depth': 5,
    'n_estimators': 300,
    'reg_alpha': 0.1,
    'reg_lambda': 3.0,
    'tree_method': 'hist',
    'random_state': 42,
    'verbosity': 0,
}

TRAIN_BARS = 180 * 6  # 6 months
STEP_BARS = 180       # 1 month
ROLLING_W = 60


def get_feature_cols(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def augment_with_detection(df, pair_name):
    """
    Add detection engine signals as features to the dataframe.
    These tell the model: "a shift was just detected, here's what kind."
    """
    df = df.copy()
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])

    # Load shifts
    shifts_path = os.path.join(DETECTION_DIR, f'{pair_name}_shifts.csv')
    shifts = pd.read_csv(shifts_path)
    shifts['datetime_utc'] = pd.to_datetime(shifts['datetime_utc'])

    # Load attribution
    attr_path = os.path.join(ATTRIBUTION_DIR, f'{pair_name}_attribution.csv')
    if os.path.exists(attr_path):
        attr = pd.read_csv(attr_path)
        attr['datetime_utc'] = pd.to_datetime(attr['datetime_utc'])
        shifts = shifts.merge(
            attr[['datetime_utc', 'dominant_group', 'group_technical',
                  'group_sentiment', 'group_volatility', 'group_macro']].drop_duplicates(),
            on='datetime_utc', how='left'
        )

    # Create daily shift signals
    shifts['date'] = shifts['datetime_utc'].dt.date.astype(str)

    # Aggregate per date: was there a shift? what type? severity?
    daily_signals = shifts.groupby('date').agg({
        'severity': 'max',
        'type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'none',
    }).reset_index()
    daily_signals.columns = ['date', 'shift_severity', 'shift_type']

    # Get SHAP group percentages per date
    if 'group_technical' in shifts.columns:
        group_daily = shifts.groupby('date').agg({
            'group_technical': 'mean',
            'group_sentiment': 'mean',
            'group_volatility': 'mean',
            'group_macro': 'mean',
        }).reset_index()
        group_daily.columns = ['date', 'det_technical_pct', 'det_sentiment_pct',
                               'det_volatility_pct', 'det_macro_pct']
        daily_signals = daily_signals.merge(group_daily, on='date', how='left')

    # Merge onto main df
    if 'date' not in df.columns:
        df['date'] = df['datetime_utc'].dt.date.astype(str)

    df = df.merge(daily_signals, on='date', how='left')

    # Fill non-shift days
    df['shift_severity'] = df['shift_severity'].fillna(0).astype(float)
    df['shift_detected'] = (df['shift_severity'] > 0).astype(int)

    # Encode shift type
    df['shift_is_scheduled'] = (df['shift_type'] == 'scheduled').astype(int)
    df['shift_is_unexpected'] = (df['shift_type'] == 'unexpected').astype(int)

    # Fill group percentages
    for col in ['det_technical_pct', 'det_sentiment_pct', 'det_volatility_pct', 'det_macro_pct']:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Rolling shift count (how many shifts in last 30 bars)
    df['shifts_last_30'] = df['shift_detected'].rolling(30, min_periods=1).sum()
    df['shifts_last_90'] = df['shift_detected'].rolling(90, min_periods=1).sum()

    # Rolling severity (max severity in last 30 bars)
    df['max_severity_30'] = df['shift_severity'].rolling(30, min_periods=1).max()

    # Time since last shift (bars)
    shift_indices = df.index[df['shift_detected'] == 1].tolist()
    bars_since = []
    last_shift = -9999
    for i in range(len(df)):
        if i in shift_indices:
            last_shift = i
        bars_since.append(i - last_shift)
    df['bars_since_last_shift'] = bars_since

    # Drop intermediate columns
    df = df.drop(columns=['shift_type'], errors='ignore')

    return df


def train_xgb(X, y, params):
    model = xgb.XGBRegressor(**params)
    model.fit(X, y, verbose=False)
    return model


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred))
    y_dir_true = (y_true > 0).astype(int)
    y_dir_pred = (y_pred > 0).astype(int)
    acc = accuracy_score(y_dir_true, y_dir_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_dir_true, y_dir_pred, average='binary', zero_division=0)
    return {'mae': mae, 'dir_acc': dir_acc, 'accuracy': acc, 'f1': f1, 'precision': p, 'recall': r}


def run_comparison(pair_name):
    print(f"\n{'='*60}")
    print(f"Augmented XGBoost — {pair_name}")
    print(f"{'='*60}")

    # Load base features
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f'{pair_name}_features.csv'))
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    base_feature_cols = get_feature_cols(df)

    # Augment with detection signals
    df_aug = augment_with_detection(df, pair_name)
    aug_feature_cols = get_feature_cols(df_aug)
    new_cols = [c for c in aug_feature_cols if c not in base_feature_cols]

    print(f"  Base features: {len(base_feature_cols)}")
    print(f"  Augmented features: {len(aug_feature_cols)} (+{len(new_cols)} detection signals)")
    print(f"  New features: {new_cols}")

    # Walk-forward comparison
    cursor = TRAIN_BARS
    total = len(df_aug)

    records = []
    step = 0

    while cursor + STEP_BARS <= total:
        step_end = cursor + STEP_BARS
        train_start = max(0, cursor - TRAIN_BARS)

        # Train data
        train_chunk = df_aug.iloc[train_start:cursor]
        X_train_base = train_chunk[base_feature_cols].values
        X_train_aug = train_chunk[aug_feature_cols].values
        y_train = train_chunk['target_return'].values

        # Test data
        test_chunk = df_aug.iloc[cursor:step_end]
        X_test_base = test_chunk[base_feature_cols].values
        X_test_aug = test_chunk[aug_feature_cols].values
        y_test = test_chunk['target_return'].values

        # Train both models
        model_base = train_xgb(X_train_base, y_train, PARAMS)
        model_aug = train_xgb(X_train_aug, y_train, PARAMS)

        # Predict
        pred_base = model_base.predict(X_test_base)
        pred_aug = model_aug.predict(X_test_aug)

        for i in range(len(y_test)):
            records.append({
                'datetime_utc': test_chunk['datetime_utc'].iloc[i],
                'actual': y_test[i],
                'pred_base': pred_base[i],
                'pred_augmented': pred_aug[i],
                'shift_detected': test_chunk['shift_detected'].iloc[i],
            })

        step += 1
        cursor = step_end

        if step % 10 == 0:
            mae_b = mean_absolute_error(y_test, pred_base)
            mae_a = mean_absolute_error(y_test, pred_aug)
            print(f"  Step {step} | Base:{mae_b:.5f} Aug:{mae_a:.5f}")

    results_df = pd.DataFrame(records)
    actuals = results_df['actual'].values

    # Overall metrics
    print(f"\n{'='*60}")
    print(f"RESULTS — {pair_name}")
    print(f"{'='*60}")

    for name, col in [('Base XGBoost', 'pred_base'), ('Augmented XGBoost', 'pred_augmented')]:
        preds = results_df[col].values
        metrics = evaluate(actuals, preds)
        print(f"\n  {name}:")
        print(f"    MAE:      {metrics['mae']:.6f}")
        print(f"    Dir Acc:  {metrics['dir_acc']:.4f}")
        print(f"    F1:       {metrics['f1']:.4f}")
        print(f"    Prec:     {metrics['precision']:.4f}  Recall: {metrics['recall']:.4f}")

    # Improvement
    base_mae = mean_absolute_error(actuals, results_df['pred_base'].values)
    aug_mae = mean_absolute_error(actuals, results_df['pred_augmented'].values)
    improvement = (base_mae - aug_mae) / base_mae * 100

    base_dir = np.mean(np.sign(actuals) == np.sign(results_df['pred_base'].values))
    aug_dir = np.mean(np.sign(actuals) == np.sign(results_df['pred_augmented'].values))
    dir_improvement = (aug_dir - base_dir) / base_dir * 100

    print(f"\n  MAE Improvement:     {improvement:+.2f}%")
    print(f"  Dir Acc Improvement: {dir_improvement:+.2f}%")

    # Performance during shift periods vs non-shift
    shift_mask = results_df['shift_detected'] == 1
    if shift_mask.sum() > 0:
        print(f"\n  During shifts ({shift_mask.sum()} bars):")
        shift_base_mae = mean_absolute_error(actuals[shift_mask], results_df['pred_base'].values[shift_mask])
        shift_aug_mae = mean_absolute_error(actuals[shift_mask], results_df['pred_augmented'].values[shift_mask])
        shift_imp = (shift_base_mae - shift_aug_mae) / shift_base_mae * 100
        print(f"    Base MAE:      {shift_base_mae:.6f}")
        print(f"    Augmented MAE: {shift_aug_mae:.6f}")
        print(f"    Improvement:   {shift_imp:+.2f}%")

        print(f"\n  During non-shifts ({(~shift_mask).sum()} bars):")
        ns_base_mae = mean_absolute_error(actuals[~shift_mask], results_df['pred_base'].values[~shift_mask])
        ns_aug_mae = mean_absolute_error(actuals[~shift_mask], results_df['pred_augmented'].values[~shift_mask])
        ns_imp = (ns_base_mae - ns_aug_mae) / ns_base_mae * 100
        print(f"    Base MAE:      {ns_base_mae:.6f}")
        print(f"    Augmented MAE: {ns_aug_mae:.6f}")
        print(f"    Improvement:   {ns_imp:+.2f}%")

    # Save
    summary = {
        'base_mae': round(base_mae, 6),
        'augmented_mae': round(aug_mae, 6),
        'mae_improvement_pct': round(improvement, 2),
        'base_dir_acc': round(base_dir, 4),
        'augmented_dir_acc': round(aug_dir, 4),
        'dir_acc_improvement_pct': round(dir_improvement, 2),
        'base_features': len(base_feature_cols),
        'augmented_features': len(aug_feature_cols),
        'detection_signal_features': new_cols,
    }

    results_df.to_csv(os.path.join(RETRAINING_DIR, f'{pair_name}_augmented_predictions.csv'), index=False)

    with open(os.path.join(RETRAINING_DIR, f'{pair_name}_augmented_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved to results/retraining/")
    return summary


if __name__ == '__main__':
    all_summaries = {}
    for pair in ['EURUSD', 'GBPJPY', 'XAUUSD']:
        summary = run_comparison(pair)
        all_summaries[pair] = summary

    print(f"\n{'='*60}")
    print("AUGMENTED XGBOOST — Cross-Pair Summary")
    print(f"{'='*60}")
    print(f"\n{'Pair':<10} {'Base MAE':<12} {'Aug MAE':<12} {'MAE Imp':<10} {'Base Dir':<10} {'Aug Dir':<10} {'Dir Imp'}")
    print("-" * 74)
    for pair, s in all_summaries.items():
        print(f"{pair:<10} {s['base_mae']:<12.6f} {s['augmented_mae']:<12.6f} "
              f"{s['mae_improvement_pct']:+.2f}%     "
              f"{s['base_dir_acc']:<10.4f} {s['augmented_dir_acc']:<10.4f} "
              f"{s['dir_acc_improvement_pct']:+.2f}%")

    with open(os.path.join(RETRAINING_DIR, 'augmented_overall.json'), 'w') as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved: results/retraining/augmented_overall.json")
