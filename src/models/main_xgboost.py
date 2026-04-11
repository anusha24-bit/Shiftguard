"""
Main Prediction Model: XGBoost
All 4 feature groups, hyperparameter tuning via TimeSeriesSplit.
Trains on 2015-2019, validates on 2020, tests on 2021-2025.

Usage:
    python src/models/main_xgboost.py
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'predictions')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

EXCLUDE_COLS = ['datetime_utc', 'date', 'session', 'target_return', 'target_direction', 'volume']

# Hyperparameter search space
PARAM_GRID = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'n_estimators': [300, 500, 800],
    'reg_alpha': [0, 0.1, 1.0],
    'reg_lambda': [1.0, 5.0],
}


# ============================================================
# Data helpers
# ============================================================
def get_feature_cols(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def split_data(df):
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    train = df[df['datetime_utc'] < '2020-01-01'].copy()
    val = df[(df['datetime_utc'] >= '2020-01-01') & (df['datetime_utc'] < '2021-01-01')].copy()
    test = df[df['datetime_utc'] >= '2021-01-01'].copy()
    return train, val, test


# ============================================================
# Evaluation
# ============================================================
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


# ============================================================
# Hyperparameter tuning
# ============================================================
def tune_hyperparams(X_train, y_train):
    """Grid search with TimeSeriesSplit on training data."""
    print("  Tuning hyperparameters...")
    tscv = TimeSeriesSplit(n_splits=5)

    best_score = float('inf')
    best_params = None
    all_results = []
    total = 1
    for v in PARAM_GRID.values():
        total *= len(v)
    count = 0

    for lr in PARAM_GRID['learning_rate']:
        for depth in PARAM_GRID['max_depth']:
            for n_est in PARAM_GRID['n_estimators']:
                for alpha in PARAM_GRID['reg_alpha']:
                    for lam in PARAM_GRID['reg_lambda']:
                        count += 1
                        params = {
                            'learning_rate': lr,
                            'max_depth': depth,
                            'n_estimators': n_est,
                            'reg_alpha': alpha,
                            'reg_lambda': lam,
                        }

                        fold_scores = []
                        for train_idx, val_idx in tscv.split(X_train):
                            X_tr, X_vl = X_train[train_idx], X_train[val_idx]
                            y_tr, y_vl = y_train[train_idx], y_train[val_idx]

                            model = xgb.XGBRegressor(
                                **params,
                                tree_method='hist',
                                random_state=42,
                                verbosity=0,
                            )
                            model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
                            pred = model.predict(X_vl)
                            fold_scores.append(mean_absolute_error(y_vl, pred))

                        avg_mae = np.mean(fold_scores)
                        all_results.append({**params, 'cv_mae': avg_mae})

                        if avg_mae < best_score:
                            best_score = avg_mae
                            best_params = params.copy()

                        if count % 20 == 0 or count == total:
                            print(f"    [{count}/{total}] Best MAE so far: {best_score:.6f}")

    print(f"  Best params: {best_params}")
    print(f"  Best CV MAE: {best_score:.6f}")

    return best_params, pd.DataFrame(all_results)


# ============================================================
# Main
# ============================================================
def run_pair(pair_name):
    print(f"\n{'='*60}")
    print(f"XGBoost Main Model — {pair_name}")
    print(f"{'='*60}")

    # Load
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f'{pair_name}_features.csv'))
    feature_cols = get_feature_cols(df)
    print(f"  Features: {len(feature_cols)}, Rows: {len(df)}")

    # Split
    train_df, val_df, test_df = split_data(df)
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    X_train = train_df[feature_cols].values
    y_train = train_df['target_return'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['target_return'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['target_return'].values

    # Tune on training data
    best_params, tuning_results = tune_hyperparams(X_train, y_train)

    # Save tuning results
    tuning_path = os.path.join(RESULTS_DIR, f'xgboost_{pair_name}_tuning.csv')
    tuning_results.to_csv(tuning_path, index=False)

    # Train final model on full training set with best params
    print(f"\n  Training final model with best params...")
    model = xgb.XGBRegressor(
        **best_params,
        tree_method='hist',
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Predict
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    # Evaluate
    val_results = evaluate(y_val, val_pred, "Validation")
    test_results = evaluate(y_test, test_pred, "Test")
    test_results['best_params'] = best_params

    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False)

    imp_path = os.path.join(RESULTS_DIR, f'xgboost_{pair_name}_importances.csv')
    importance.to_csv(imp_path, index=False)
    print(f"\n  Top 10 features:")
    for _, row in importance.head(10).iterrows():
        print(f"    {row['feature']:<30s} {row['importance']:.4f}")

    # Save predictions
    pred_df = pd.DataFrame({
        'datetime_utc': test_df['datetime_utc'].values,
        'actual': y_test,
        'predicted': test_pred,
    })
    pred_path = os.path.join(RESULTS_DIR, f'xgboost_{pair_name}_predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"  Predictions saved: {pred_path}")

    # Save model
    model_path = os.path.join(RESULTS_DIR, f'xgboost_{pair_name}.json')
    model.save_model(model_path)
    print(f"  Model saved: {model_path}")

    # Rolling MAE for adaptation analysis
    rolling_mae = pd.Series(np.abs(y_test - test_pred)).rolling(30).mean()
    rolling_df = pd.DataFrame({
        'datetime_utc': test_df['datetime_utc'].values,
        'rolling_mae_30': rolling_mae.values,
    })
    rolling_path = os.path.join(RESULTS_DIR, f'xgboost_{pair_name}_rolling_mae.csv')
    rolling_df.to_csv(rolling_path, index=False)

    return test_results


if __name__ == '__main__':
    all_results = {}
    for pair in ['EURUSD', 'GBPJPY', 'XAUUSD']:
        results = run_pair(pair)
        all_results[pair] = results

    # Summary
    print(f"\n{'='*60}")
    print("MAIN MODEL: XGBoost — Summary")
    print(f"{'='*60}")
    print(f"{'Pair':<10} {'MAE':<10} {'RMSE':<10} {'Dir Acc':<10} {'F1':<10}")
    print("-" * 50)
    for pair, r in all_results.items():
        print(f"{pair:<10} {r['mae']:<10.6f} {r['rmse']:<10.6f} {r['dir_acc']:<10.4f} {r['f1']:<10.4f}")

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, 'xgboost_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSummary saved: {summary_path}")
