"""
Baseline 2: Random Forest
All 4 feature groups, flat features (no sequence), GridSearchCV with TimeSeriesSplit.
Trains on 2015-2019, validates on 2020, tests on 2021-2025.

Usage:
    python3 src/models/baseline_rf.py
"""
import sys
import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'predictions')
os.makedirs(RESULTS_DIR, exist_ok=True)

EXCLUDE_COLS = ['datetime_utc', 'date', 'session', 'target_return', 'target_direction', 'volume']

PARAM_GRID = {
    'n_estimators': [300, 500, 800],
    'max_depth': [8, 12, 18, None],
    'min_samples_leaf': [5, 10, 20],
}


def get_feature_cols(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def split_data(df):
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    train = df[df['datetime_utc'] < '2020-01-01'].copy()
    val = df[(df['datetime_utc'] >= '2020-01-01') & (df['datetime_utc'] < '2021-01-01')].copy()
    test = df[df['datetime_utc'] >= '2021-01-01'].copy()
    return train, val, test


def evaluate(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred))

    y_true_dir = (y_true > 0).astype(int)
    y_pred_dir = (y_pred > 0).astype(int)
    acc = accuracy_score(y_true_dir, y_pred_dir)
    p, r, f1, _ = precision_recall_fscore_support(y_true_dir, y_pred_dir, average='binary', zero_division=0)

    print(f"\n  {label} Results:")
    print(f"    MAE:      {mae:.6f}")
    print(f"    RMSE:     {rmse:.6f}")
    print(f"    Dir Acc:  {dir_acc:.4f}")
    print(f"    Accuracy: {acc:.4f}")
    print(f"    F1:       {f1:.4f}")
    print(f"    Prec:     {p:.4f}  Recall: {r:.4f}")

    return {'mae': mae, 'rmse': rmse, 'dir_acc': dir_acc, 'accuracy': acc, 'f1': f1, 'precision': p, 'recall': r}


def run_pair(pair_name):
    print(f"\n{'='*60}")
    print(f"Random Forest — {pair_name}")
    print(f"{'='*60}")

    # Load
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f'{pair_name}_features.csv'))
    feature_cols = get_feature_cols(df)
    print(f"  Features: {len(feature_cols)}, Rows: {len(df)}")

    # Split
    train_df, val_df, test_df = split_data(df)
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Prepare data — RF handles NaN poorly, fill with median
    X_train = train_df[feature_cols].fillna(train_df[feature_cols].median())
    X_val = val_df[feature_cols].fillna(train_df[feature_cols].median())
    X_test = test_df[feature_cols].fillna(train_df[feature_cols].median())

    y_train = train_df['target_return'].fillna(0).values
    y_val = val_df['target_return'].fillna(0).values
    y_test = test_df['target_return'].fillna(0).values

    # Replace any remaining inf
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_val = X_val.replace([np.inf, -np.inf], 0)
    X_test = X_test.replace([np.inf, -np.inf], 0)

    # Hyperparameter tuning with TimeSeriesSplit on training data
    print(f"  GridSearchCV with TimeSeriesSplit(5)...")
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        PARAM_GRID,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        verbose=0,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    best_params = grid.best_params_
    best_score = -grid.best_score_
    print(f"  Best params: {best_params}")
    print(f"  Best CV MAE: {best_score:.6f}")

    # Train final model on full training set with best params
    model = grid.best_estimator_

    # Evaluate on validation
    val_pred = model.predict(X_val)
    val_results = evaluate(y_val, val_pred, "Validation")

    # Evaluate on test
    test_pred = model.predict(X_test)
    test_results = evaluate(y_test, test_pred, "Test")
    test_results['best_params'] = best_params

    # Feature importances (top 15)
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"\n  Top 15 Feature Importances:")
    for feat, imp in importances.head(15).items():
        print(f"    {feat:<25s} {imp:.4f}")

    # Save predictions
    pred_df = pd.DataFrame({
        'datetime_utc': test_df['datetime_utc'].values,
        'actual': y_test,
        'predicted': test_pred,
    })
    pred_path = os.path.join(RESULTS_DIR, f'baseline2_rf_{pair_name}.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"\n  Predictions saved: {pred_path}")

    # Save feature importances
    imp_path = os.path.join(RESULTS_DIR, f'baseline2_rf_{pair_name}_importances.csv')
    importances.to_csv(imp_path)

    return test_results


if __name__ == '__main__':
    all_results = {}
    for pair in ['EURUSD', 'GBPJPY', 'XAUUSD']:
        results = run_pair(pair)
        all_results[pair] = results

    # Summary table
    print(f"\n{'='*60}")
    print("BASELINE 2: Random Forest — Summary")
    print(f"{'='*60}")
    print(f"{'Pair':<10} {'MAE':<10} {'RMSE':<10} {'Dir Acc':<10} {'F1':<10}")
    print("-" * 50)
    for pair, r in all_results.items():
        print(f"{pair:<10} {r['mae']:<10.6f} {r['rmse']:<10.6f} {r['dir_acc']:<10.4f} {r['f1']:<10.4f}")

    # Save summary
    # Convert best_params for JSON serialization
    for pair in all_results:
        if 'best_params' in all_results[pair]:
            params = all_results[pair]['best_params']
            all_results[pair]['best_params'] = {k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in params.items()}

    summary_path = os.path.join(RESULTS_DIR, 'baseline2_rf_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSummary saved: {summary_path}")
