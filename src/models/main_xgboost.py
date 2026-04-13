"""
Canonical monitored model for ShiftGuard.

This model predicts next-bar return and supplies:
1. prediction outputs for performance drift monitoring,
2. a TreeSHAP-compatible model for attribution,
3. a stable baseline for selective retraining experiments.

Usage:
    python src/models/main_xgboost.py
"""
import json
import os
import sys
import argparse

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "predictions")
os.makedirs(RESULTS_DIR, exist_ok=True)

EXCLUDE_COLS = ["datetime_utc", "date", "session", "target_return", "target_direction", "volume"]

PARAM_GRID = {
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "n_estimators": [300, 500],
    "reg_alpha": [0, 0.1, 1.0],
    "reg_lambda": [1.0, 5.0],
}

FAST_PARAM_GRID = {
    "learning_rate": [0.05],
    "max_depth": [3, 5],
    "n_estimators": [200],
    "reg_alpha": [0.1],
    "reg_lambda": [1.0, 5.0],
}


def get_feature_cols(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def split_data(df):
    df = df.copy()
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    train = df[df["datetime_utc"] < "2020-01-01"].copy()
    val = df[(df["datetime_utc"] >= "2020-01-01") & (df["datetime_utc"] < "2021-01-01")].copy()
    test = df[df["datetime_utc"] >= "2021-01-01"].copy()
    return train, val, test


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred))

    y_true_dir = (y_true > 0).astype(int)
    y_pred_dir = (y_pred > 0).astype(int)
    acc = accuracy_score(y_true_dir, y_pred_dir)
    p, r, f1, _ = precision_recall_fscore_support(y_true_dir, y_pred_dir, average="binary", zero_division=0)

    return {
        "mae": round(float(mae), 6),
        "rmse": round(float(rmse), 6),
        "dir_acc": round(float(dir_acc), 4),
        "accuracy": round(float(acc), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
    }


def tune_hyperparams(X_train, y_train, fast=False):
    grid = FAST_PARAM_GRID if fast else PARAM_GRID
    tscv = TimeSeriesSplit(n_splits=5)
    best_score = float("inf")
    best_params = None
    rows = []

    for lr in grid["learning_rate"]:
        for depth in grid["max_depth"]:
            for n_est in grid["n_estimators"]:
                for alpha in grid["reg_alpha"]:
                    for lam in grid["reg_lambda"]:
                        params = {
                            "learning_rate": lr,
                            "max_depth": depth,
                            "n_estimators": n_est,
                            "reg_alpha": alpha,
                            "reg_lambda": lam,
                        }
                        fold_scores = []

                        for train_idx, val_idx in tscv.split(X_train):
                            model = xgb.XGBRegressor(
                                **params,
                                tree_method="hist",
                                random_state=42,
                                verbosity=0,
                            )
                            model.fit(X_train[train_idx], y_train[train_idx], verbose=False)
                            pred = model.predict(X_train[val_idx])
                            fold_scores.append(mean_absolute_error(y_train[val_idx], pred))

                        cv_mae = float(np.mean(fold_scores))
                        rows.append({**params, "cv_mae": round(cv_mae, 6)})
                        if cv_mae < best_score:
                            best_score = cv_mae
                            best_params = params.copy()

    return best_params, pd.DataFrame(rows).sort_values("cv_mae")


def run_pair(pair_name, fast=False):
    print(f"\n{'=' * 60}")
    print(f"ShiftGuard Main XGBoost - {pair_name}")
    print(f"{'=' * 60}")

    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{pair_name}_features.csv"))
    feature_cols = get_feature_cols(df)
    train_df, val_df, test_df = split_data(df)

    X_train = train_df[feature_cols].values
    y_train = train_df["target_return"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["target_return"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["target_return"].values

    best_params, tuning_df = tune_hyperparams(X_train, y_train, fast=fast)
    tuning_df.to_csv(os.path.join(RESULTS_DIR, f"xgboost_{pair_name}_tuning.csv"), index=False)

    model = xgb.XGBRegressor(
        **best_params,
        tree_method="hist",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_metrics = evaluate(y_val, val_pred)
    test_metrics = evaluate(y_test, test_pred)

    pred_df = pd.DataFrame(
        {
            "datetime_utc": test_df["datetime_utc"].values,
            "actual": y_test,
            "predicted": test_pred,
        }
    )
    pred_df.to_csv(os.path.join(RESULTS_DIR, f"xgboost_{pair_name}_predictions.csv"), index=False)

    rolling_df = pd.DataFrame(
        {
            "datetime_utc": test_df["datetime_utc"].values,
            "rolling_mae_30": pd.Series(np.abs(y_test - test_pred)).rolling(30).mean().values,
        }
    )
    rolling_df.to_csv(os.path.join(RESULTS_DIR, f"xgboost_{pair_name}_rolling_mae.csv"), index=False)

    pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).to_csv(
        os.path.join(RESULTS_DIR, f"xgboost_{pair_name}_importances.csv"),
        index=False,
    )

    model.save_model(os.path.join(RESULTS_DIR, f"xgboost_{pair_name}.json"))

    return {
        "pair": pair_name,
        "n_features": len(feature_cols),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "best_params": best_params,
        "validation": val_metrics,
        "test": test_metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", default=["EURUSD", "GBPJPY", "XAUUSD"])
    parser.add_argument("--fast", action="store_true", help="Use a much smaller tuning grid for local validation.")
    args = parser.parse_args()

    summary = {}
    for pair in args.pairs:
        pair_result = run_pair(pair, fast=args.fast)
        summary[pair] = pair_result

    with open(os.path.join(RESULTS_DIR, "xgboost_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved canonical model outputs to {RESULTS_DIR}")
