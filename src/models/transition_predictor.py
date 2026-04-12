"""
Main Model: Regime Transition Predictor
Predicts "will the volatility regime change in the next 6 bars (1 day)?"
Binary classification: 0=no change, 1=transition coming.

Walk-forward with retraining comparison:
  A) Static (train once, never update)
  B) Blind monthly (retrain every month)
  C) ShiftGuard (retrain only at detected non-technical shifts)

Usage:
    python src/models/transition_predictor.py
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, classification_report)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.features.technical import compute_technical_features
from src.features.volatility import compute_volatility_features
from src.features.macro import compute_macro_features
from src.features.sentiment import compute_sentiment_features
from src.features.regime import compute_regime_features, compute_adaptive_regime_labels

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'regime')
os.makedirs(RESULTS_DIR, exist_ok=True)

META_COLS = ['datetime_utc', 'date', 'session', 'target_return', 'target_direction',
             'volume', 'month', 'regime_label', 'target_regime', 'regime_changed',
             'target_transition_6', 'target_transition_12', 'target_transition_18']

TARGET_COL = 'target_transition_6'  # predict 1-day ahead transition
TRAIN_BARS = 180 * 6   # 6 months
STEP_BARS = 180         # 1 month

# Tuning grid
PARAM_GRID = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'n_estimators': [200, 400],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1.0, 5.0],
    'scale_pos_weight': [1.0, 3.0, 5.0],  # handle class imbalance (transitions are rare)
}

BASE_PARAMS = {
    'tree_method': 'hist',
    'random_state': 42,
    'verbosity': 0,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
}


def get_feature_cols(df):
    return [c for c in df.columns if c not in META_COLS]


def train_model(X, y, params):
    """Train with class safety."""
    unique = np.unique(y)
    if len(unique) < 2:
        idx = np.random.randint(0, len(X))
        X = np.vstack([X, X[idx:idx+1]])
        missing = 1 if 0 in unique else 0
        y = np.append(y, missing)
    model = xgb.XGBClassifier(**params)
    model.fit(X, y, verbose=False)
    return model


def tune_params(X_train, y_train, X_val, y_val):
    """Grid search for best F1 (not accuracy — transitions are rare)."""
    best_f1 = 0
    best_params = None
    count = 0

    for lr in PARAM_GRID['learning_rate']:
        for depth in PARAM_GRID['max_depth']:
            for n_est in PARAM_GRID['n_estimators']:
                for alpha in PARAM_GRID['reg_alpha']:
                    for lam in PARAM_GRID['reg_lambda']:
                        for spw in PARAM_GRID['scale_pos_weight']:
                            count += 1
                            params = {
                                **BASE_PARAMS,
                                'learning_rate': lr,
                                'max_depth': depth,
                                'n_estimators': n_est,
                                'reg_alpha': alpha,
                                'reg_lambda': lam,
                                'scale_pos_weight': spw,
                            }
                            model = train_model(X_train, y_train, params)
                            preds = model.predict(X_val)
                            f1 = f1_score(y_val, preds, zero_division=0)
                            if f1 > best_f1:
                                best_f1 = f1
                                best_params = params.copy()

    if best_params is None:
        best_params = {**BASE_PARAMS, 'learning_rate': 0.05, 'max_depth': 5,
                       'n_estimators': 300, 'reg_alpha': 0.1, 'reg_lambda': 3.0, 'scale_pos_weight': 3.0}

    print(f"    Best F1: {best_f1:.4f} | lr={best_params['learning_rate']}, "
          f"depth={best_params['max_depth']}, spw={best_params['scale_pos_weight']}")
    return best_params


def build_dataset(pair_name):
    print(f"  Building dataset for {pair_name}...")
    path = os.path.join(DATA_DIR, 'price', f'{pair_name}_4h.csv')
    df = pd.read_csv(path, parse_dates=['datetime_utc'])
    df = df.sort_values('datetime_utc').reset_index(drop=True)

    df = compute_technical_features(df)
    df = compute_volatility_features(df)
    df = compute_macro_features(df, pair_name, DATA_DIR)
    df = compute_sentiment_features(df, pair_name, DATA_DIR)
    df = compute_regime_features(df)
    df = compute_adaptive_regime_labels(df)

    df = df.iloc[260:].reset_index(drop=True)
    df = df.iloc[:-18].reset_index(drop=True)  # drop last 18 bars (no target)

    transition_rate = df[TARGET_COL].mean()
    print(f"  Rows: {len(df)}, Features: {len(get_feature_cols(df))}")
    print(f"  Transition rate: {transition_rate:.2%} (class 1)")

    return df


def get_shap_dominant_group(explainer, X_chunk, feature_cols):
    """Get dominant SHAP feature group for a chunk."""
    groups_path = os.path.join(PROCESSED_DIR, 'feature_groups.json')
    with open(groups_path) as f:
        FEATURE_GROUPS = json.load(f)

    try:
        shap_vals = explainer.shap_values(X_chunk)
        mean_abs = np.abs(shap_vals).mean(axis=0)
        group_imp = {}
        for feat, imp in zip(feature_cols, mean_abs):
            found = False
            for gname, gfeats in FEATURE_GROUPS.items():
                if feat in gfeats:
                    group_imp[gname] = group_imp.get(gname, 0) + imp
                    found = True
                    break
            if not found:
                group_imp['other'] = group_imp.get('other', 0) + imp
        return max(group_imp, key=group_imp.get) if group_imp else 'other'
    except:
        return 'other'


def run_experiment(pair_name):
    print(f"\n{'='*60}")
    print(f"Transition Predictor — {pair_name}")
    print(f"{'='*60}")

    df = build_dataset(pair_name)
    feature_cols = get_feature_cols(df)

    cursor = TRAIN_BARS
    total = len(df)

    # Tune on initial window
    train_start = max(0, cursor - TRAIN_BARS)
    split = train_start + int((cursor - train_start) * 0.8)

    X_tr = df.iloc[train_start:split][feature_cols].values
    y_tr = df.iloc[train_start:split][TARGET_COL].values
    X_vl = df.iloc[split:cursor][feature_cols].values
    y_vl = df.iloc[split:cursor][TARGET_COL].values

    print(f"  Tuning ({sum(len(v) for v in PARAM_GRID.values())} param options)...")
    best_params = tune_params(X_tr, y_tr, X_vl, y_vl)

    # Initialize 3 strategies
    X_init = df.iloc[train_start:cursor][feature_cols].values
    y_init = df.iloc[train_start:cursor][TARGET_COL].values

    model_static = train_model(X_init, y_init, best_params)
    model_blind = train_model(X_init, y_init, best_params)
    model_shap = train_model(X_init, y_init, best_params)

    explainer = shap.TreeExplainer(model_shap)

    records = []
    retrains = {'static': 0, 'blind': 0, 'shap_guided': 0}
    step = 0

    while cursor + STEP_BARS <= total:
        step_end = cursor + STEP_BARS
        chunk = df.iloc[cursor:step_end]
        X_chunk = chunk[feature_cols].values
        y_chunk = chunk[TARGET_COL].values

        # Predict
        pred_static = model_static.predict(X_chunk)
        pred_blind = model_blind.predict(X_chunk)
        pred_shap = model_shap.predict(X_chunk)

        # Probabilities for AUC
        prob_static = model_static.predict_proba(X_chunk)[:, 1]
        prob_blind = model_blind.predict_proba(X_chunk)[:, 1]
        prob_shap = model_shap.predict_proba(X_chunk)[:, 1]

        for i in range(len(y_chunk)):
            records.append({
                'datetime_utc': chunk['datetime_utc'].iloc[i],
                'actual': y_chunk[i],
                'pred_static': pred_static[i],
                'pred_blind': pred_blind[i],
                'pred_shap': pred_shap[i],
                'prob_static': prob_static[i],
                'prob_blind': prob_blind[i],
                'prob_shap': prob_shap[i],
                'regime_label': chunk['regime_label'].iloc[i],
                'regime_changed': chunk['regime_changed'].iloc[i],
            })

        # Retraining decisions
        train_start = max(0, step_end - TRAIN_BARS)
        X_retrain = df.iloc[train_start:step_end][feature_cols].values
        y_retrain = df.iloc[train_start:step_end][TARGET_COL].values

        # B) Blind monthly
        model_blind = train_model(X_retrain, y_retrain, best_params)
        retrains['blind'] += 1

        # C) SHAP-guided: retrain only if non-technical regime change
        has_change = chunk['regime_changed'].sum() > 0
        if has_change:
            dominant = get_shap_dominant_group(explainer, X_chunk, feature_cols)
            if dominant != 'technical':
                model_shap = train_model(X_retrain, y_retrain, best_params)
                retrains['shap_guided'] += 1
                explainer = shap.TreeExplainer(model_shap)

        step += 1
        cursor = step_end

        if step % 10 == 0:
            recent_actual = [r['actual'] for r in records[-STEP_BARS:]]
            recent_static = [r['pred_static'] for r in records[-STEP_BARS:]]
            recent_shap = [r['pred_shap'] for r in records[-STEP_BARS:]]
            f1_s = f1_score(recent_actual, recent_static, zero_division=0)
            f1_sg = f1_score(recent_actual, recent_shap, zero_division=0)
            print(f"  Step {step} | Static F1:{f1_s:.3f} SHAP-guided F1:{f1_sg:.3f}")

    # Results
    results_df = pd.DataFrame(records)
    actuals = results_df['actual'].values

    print(f"\n{'='*60}")
    print(f"RESULTS — {pair_name}")
    print(f"{'='*60}")

    summary = {}
    for sname, pred_col, prob_col in [
        ('Static', 'pred_static', 'prob_static'),
        ('Blind Monthly', 'pred_blind', 'prob_blind'),
        ('ShiftGuard', 'pred_shap', 'prob_shap'),
    ]:
        preds = results_df[pred_col].values
        probs = results_df[prob_col].values

        acc = accuracy_score(actuals, preds)
        f1 = f1_score(actuals, preds, zero_division=0)
        prec = precision_score(actuals, preds, zero_division=0)
        rec = recall_score(actuals, preds, zero_division=0)
        try:
            auc = roc_auc_score(actuals, probs)
        except:
            auc = 0.5

        rkey = sname.lower().replace(' ', '_')
        n_ret = retrains.get(rkey, retrains.get('shap_guided' if 'shift' in rkey.lower() else rkey, 0))

        print(f"\n  {sname} ({n_ret} retrains):")
        print(f"    Accuracy:  {acc:.4f}")
        print(f"    F1:        {f1:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall:    {rec:.4f}")
        print(f"    AUC-ROC:   {auc:.4f}")

        summary[sname] = {
            'accuracy': round(acc, 4),
            'f1': round(f1, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'auc_roc': round(auc, 4),
            'retrains': n_ret,
        }

    # Improvement
    base_f1 = summary['Static']['f1']
    for sname in ['Blind Monthly', 'ShiftGuard']:
        imp = (summary[sname]['f1'] - base_f1) / base_f1 * 100 if base_f1 > 0 else 0
        summary[sname]['f1_improvement_pct'] = round(imp, 2)
        print(f"\n  {sname} vs Static: F1 {imp:+.2f}%")

    # Performance during regime transitions specifically
    transition_mask = results_df['regime_changed'] == 1
    if transition_mask.sum() > 10:
        print(f"\n  During regime transitions ({transition_mask.sum()} bars):")
        for sname, col in [('Static', 'pred_static'), ('ShiftGuard', 'pred_shap')]:
            t_f1 = f1_score(actuals[transition_mask], results_df[col].values[transition_mask], zero_division=0)
            t_rec = recall_score(actuals[transition_mask], results_df[col].values[transition_mask], zero_division=0)
            print(f"    {sname}: F1={t_f1:.4f} Recall={t_rec:.4f}")

    # Save
    results_df.to_csv(os.path.join(RESULTS_DIR, f'{pair_name}_transition_predictions.csv'), index=False)
    with open(os.path.join(RESULTS_DIR, f'{pair_name}_transition_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved to results/regime/")
    return summary


if __name__ == '__main__':
    all_summaries = {}
    for pair in ['EURUSD', 'GBPJPY', 'XAUUSD']:
        summary = run_experiment(pair)
        all_summaries[pair] = summary

    print(f"\n{'='*60}")
    print("TRANSITION PREDICTOR — Cross-Pair Summary")
    print(f"{'='*60}")
    print(f"\n{'Pair':<10} {'Static F1':<12} {'Blind F1':<12} {'ShiftGuard F1':<15} {'SG Retrains'}")
    print("-" * 60)
    for pair, s in all_summaries.items():
        print(f"{pair:<10} {s['Static']['f1']:<12.4f} {s['Blind Monthly']['f1']:<12.4f} "
              f"{s['ShiftGuard']['f1']:<15.4f} {s['ShiftGuard']['retrains']}")

    with open(os.path.join(RESULTS_DIR, 'transition_overall.json'), 'w') as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved: results/regime/transition_overall.json")
