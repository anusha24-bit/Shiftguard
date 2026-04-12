"""
3-Phase Sequential Validation
Tests model degradation across distinct market regimes.

Phase 1: Train 2015-2018, Test 2018-2019 (pre-COVID, calm)
Phase 2: Train 2019-2022, Test 2022-2023 (COVID, recovery, Fed hikes)
Phase 3: Train 2023-2025, Test 2025-2026 (post-hike, BOJ shift, gold rally)

Compares:
  A) Static: Train on Phase 1 only, never retrain, test across all phases
  B) Phase-retrained: Retrain at each phase boundary
  C) ShiftGuard: Retrain only when detection engine flags non-technical shift

Usage:
    python src/models/phase_experiment.py
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, mean_absolute_error)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'phases')
os.makedirs(RESULTS_DIR, exist_ok=True)

EXCLUDE_COLS = ['datetime_utc', 'date', 'session', 'target_return', 'target_direction', 'volume']

# Tuned params from earlier experiments
PARAMS = {
    'learning_rate': 0.01,
    'max_depth': 5,
    'n_estimators': 500,
    'reg_alpha': 0.1,
    'reg_lambda': 5.0,
    'tree_method': 'hist',
    'random_state': 42,
    'verbosity': 0,
}

# 3 phases with distinct market characters
PHASES = [
    {'name': 'Phase 1', 'train': ('2015-01-01', '2018-01-01'), 'test': ('2018-01-01', '2019-07-01'),
     'character': 'Pre-COVID, low vol, steady rates'},
    {'name': 'Phase 2', 'train': ('2019-01-01', '2022-01-01'), 'test': ('2022-01-01', '2023-07-01'),
     'character': 'COVID crash, recovery, Fed hike cycle'},
    {'name': 'Phase 3', 'train': ('2023-01-01', '2025-01-01'), 'test': ('2025-01-01', '2026-04-01'),
     'character': 'Post-hike, BOJ shift, gold rally, tariffs'},
]


def get_feature_cols(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def train_xgb(X, y):
    model = xgb.XGBRegressor(**PARAMS)
    model.fit(X, y, verbose=False)
    return model


def evaluate(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred))
    y_dir_true = (y_true > 0).astype(int)
    y_dir_pred = (y_pred > 0).astype(int)
    f1 = f1_score(y_dir_true, y_dir_pred, zero_division=0)
    prec = precision_score(y_dir_true, y_dir_pred, zero_division=0)
    rec = recall_score(y_dir_true, y_dir_pred, zero_division=0)
    return {'mae': round(mae, 6), 'dir_acc': round(dir_acc, 4), 'f1': round(f1, 4),
            'precision': round(prec, 4), 'recall': round(rec, 4)}


def get_shap_dominant_group(model, X, feature_cols):
    """Get dominant feature group via SHAP."""
    groups_path = os.path.join(PROCESSED_DIR, 'feature_groups.json')
    with open(groups_path) as f:
        FEATURE_GROUPS = json.load(f)
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X[:100])  # subsample for speed
        mean_abs = np.abs(shap_vals).mean(axis=0)
        group_imp = {}
        for feat, imp in zip(feature_cols, mean_abs):
            for gname, gfeats in FEATURE_GROUPS.items():
                if feat in gfeats:
                    group_imp[gname] = group_imp.get(gname, 0) + imp
                    break
            else:
                group_imp['other'] = group_imp.get('other', 0) + imp
        return max(group_imp, key=group_imp.get), group_imp
    except:
        return 'other', {}


def run_pair(pair_name):
    print(f"\n{'='*60}")
    print(f"3-Phase Experiment — {pair_name}")
    print(f"{'='*60}")

    df = pd.read_csv(os.path.join(PROCESSED_DIR, f'{pair_name}_features.csv'))
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    feature_cols = get_feature_cols(df)
    print(f"  Loaded: {len(df)} bars, {len(feature_cols)} features")

    # --- Strategy A: Static (train on Phase 1 only, test everywhere) ---
    phase1_train = df[(df['datetime_utc'] >= PHASES[0]['train'][0]) &
                      (df['datetime_utc'] < PHASES[0]['train'][1])]
    X_p1 = phase1_train[feature_cols].values
    y_p1 = phase1_train['target_return'].values
    static_model = train_xgb(X_p1, y_p1)
    print(f"  Static model trained on Phase 1: {len(phase1_train)} bars")

    results = {}

    for phase in PHASES:
        pname = phase['name']
        print(f"\n  --- {pname}: {phase['character']} ---")

        # Get test data for this phase
        test_data = df[(df['datetime_utc'] >= phase['test'][0]) &
                       (df['datetime_utc'] < phase['test'][1])]
        if len(test_data) < 50:
            print(f"    Skipping — only {len(test_data)} bars")
            continue

        X_test = test_data[feature_cols].values
        y_test = test_data['target_return'].values
        print(f"    Test: {len(test_data)} bars ({phase['test'][0]} to {phase['test'][1]})")

        # A) Static prediction
        pred_static = static_model.predict(X_test)
        eval_static = evaluate(y_test, pred_static)

        # B) Phase-retrained: train on THIS phase's training data
        train_data = df[(df['datetime_utc'] >= phase['train'][0]) &
                        (df['datetime_utc'] < phase['train'][1])]
        X_train = train_data[feature_cols].values
        y_train = train_data['target_return'].values
        retrained_model = train_xgb(X_train, y_train)
        pred_retrained = retrained_model.predict(X_test)
        eval_retrained = evaluate(y_test, pred_retrained)

        # C) ShiftGuard: retrain only if SHAP says non-technical shift
        # Check what drives predictions in this phase
        dominant, group_imp = get_shap_dominant_group(retrained_model, X_test, feature_cols)

        # If non-technical dominant → retrain (use phase-specific data)
        # If technical dominant → use static model
        if dominant != 'technical':
            shap_model = retrained_model  # retrain was justified
            pred_shap = pred_retrained
            eval_shap = eval_retrained
            shap_action = f"RETRAINED (dominant: {dominant})"
        else:
            shap_model = static_model  # no retrain needed
            pred_shap = pred_static
            eval_shap = eval_static
            shap_action = f"SKIPPED (dominant: {dominant})"

        # Improvement calculations
        mae_imp_retrained = (eval_static['mae'] - eval_retrained['mae']) / eval_static['mae'] * 100
        dir_imp_retrained = (eval_retrained['dir_acc'] - eval_static['dir_acc']) / eval_static['dir_acc'] * 100
        mae_imp_shap = (eval_static['mae'] - eval_shap['mae']) / eval_static['mae'] * 100
        dir_imp_shap = (eval_shap['dir_acc'] - eval_static['dir_acc']) / eval_static['dir_acc'] * 100

        print(f"\n    {'Strategy':<20s} {'MAE':<12s} {'Dir Acc':<10s} {'F1':<8s} {'vs Static'}")
        print(f"    {'-'*60}")
        print(f"    {'Static':<20s} {eval_static['mae']:<12.6f} {eval_static['dir_acc']:<10.4f} {eval_static['f1']:<8.4f} baseline")
        print(f"    {'Phase-Retrained':<20s} {eval_retrained['mae']:<12.6f} {eval_retrained['dir_acc']:<10.4f} {eval_retrained['f1']:<8.4f} MAE:{mae_imp_retrained:+.1f}% Dir:{dir_imp_retrained:+.1f}%")
        print(f"    {'ShiftGuard':<20s} {eval_shap['mae']:<12.6f} {eval_shap['dir_acc']:<10.4f} {eval_shap['f1']:<8.4f} MAE:{mae_imp_shap:+.1f}% Dir:{dir_imp_shap:+.1f}%")
        print(f"    SHAP decision: {shap_action}")

        # Group importance
        if group_imp:
            total = sum(group_imp.values())
            print(f"    Feature attribution: ", end="")
            for g, v in sorted(group_imp.items(), key=lambda x: -x[1]):
                pct = v/total*100 if total > 0 else 0
                if pct > 5:
                    print(f"{g}:{pct:.0f}% ", end="")
            print()

        results[pname] = {
            'character': phase['character'],
            'test_bars': len(test_data),
            'static': eval_static,
            'retrained': eval_retrained,
            'shiftguard': eval_shap,
            'shap_action': shap_action,
            'dominant_group': dominant,
            'mae_improvement_retrained': round(mae_imp_retrained, 2),
            'dir_improvement_retrained': round(dir_imp_retrained, 2),
            'mae_improvement_shap': round(mae_imp_shap, 2),
            'dir_improvement_shap': round(dir_imp_shap, 2),
        }

    # Cross-phase degradation analysis
    print(f"\n  {'='*60}")
    print(f"  DEGRADATION ANALYSIS — {pair_name}")
    print(f"  {'='*60}")
    print(f"  How much does the static model degrade across phases?")
    if 'Phase 1' in results and 'Phase 3' in results:
        p1_dir = results['Phase 1']['static']['dir_acc']
        p3_dir = results['Phase 3']['static']['dir_acc']
        degradation = (p3_dir - p1_dir) / p1_dir * 100
        print(f"    Phase 1 Dir Acc: {p1_dir:.4f}")
        print(f"    Phase 3 Dir Acc: {p3_dir:.4f}")
        print(f"    Degradation: {degradation:+.2f}%")

    # Save
    with open(os.path.join(RESULTS_DIR, f'{pair_name}_phase_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to results/phases/")

    return results


if __name__ == '__main__':
    all_results = {}
    for pair in ['EURUSD', 'GBPJPY', 'XAUUSD']:
        results = run_pair(pair)
        all_results[pair] = results

    # Overall summary
    print(f"\n{'='*60}")
    print("3-PHASE EXPERIMENT — Overall Summary")
    print(f"{'='*60}")
    for pair in all_results:
        print(f"\n  {pair}:")
        for pname, r in all_results[pair].items():
            print(f"    {pname}: Static Dir={r['static']['dir_acc']} → Retrained Dir={r['retrained']['dir_acc']} "
                  f"({r['dir_improvement_retrained']:+.1f}%) | SHAP: {r['shap_action']}")

    with open(os.path.join(RESULTS_DIR, 'phase_overall.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: results/phases/phase_overall.json")
