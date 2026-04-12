"""
Volatility Regime Classifier
Predicts next-bar volatility regime (LOW/MEDIUM/HIGH) using XGBClassifier.
Walk-forward with SHAP-guided regime-matched retraining.

Usage:
    python src/models/regime_classifier.py
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

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

EXCLUDE_COLS = ['datetime_utc', 'date', 'session', 'target_return', 'target_direction',
                'volume', 'month', 'regime_label', 'target_regime', 'regime_changed']

BASE_PARAMS = {
    'tree_method': 'hist',
    'random_state': 42,
    'verbosity': 0,
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
}

PARAM_GRID = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'n_estimators': [200, 400],
    'reg_alpha': [0, 0.1, 1.0],
    'reg_lambda': [1.0, 5.0],
}

TRAIN_BARS = 180 * 6  # 6 months
STEP_BARS = 180       # 1 month
REGIME_NAMES = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}


def get_feature_cols(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def build_regime_dataset(pair_name):
    """Build full feature matrix with regime labels."""
    print(f"  Building regime dataset for {pair_name}...")

    path = os.path.join(DATA_DIR, 'price', f'{pair_name}_4h.csv')
    df = pd.read_csv(path, parse_dates=['datetime_utc'])
    df = df.sort_values('datetime_utc').reset_index(drop=True)

    # Groups 1-4
    df = compute_technical_features(df)
    df = compute_volatility_features(df)
    df = compute_macro_features(df, pair_name, DATA_DIR)
    df = compute_sentiment_features(df, pair_name, DATA_DIR)

    # Group 5: Regime features
    df = compute_regime_features(df)

    # Adaptive regime labels
    df = compute_adaptive_regime_labels(df)

    # Drop warmup
    df = df.iloc[260:].reset_index(drop=True)  # 252 (for long ATR pct) + buffer
    df = df.iloc[:-1].reset_index(drop=True)   # drop last (no target)

    print(f"  Rows: {len(df)}, Features: {len(get_feature_cols(df))}")
    print(f"  Regime distribution: {df['target_regime'].value_counts().to_dict()}")

    return df


def find_similar_regimes(df, shift_idx, dominant_group, feature_cols, n_similar=500):
    """
    Find historical periods with similar SHAP profile for regime-matched retraining.
    Returns indices of similar regime periods.
    """
    # Get the regime features around the shift
    window = min(60, shift_idx)
    recent = df.iloc[shift_idx - window:shift_idx]

    if dominant_group == 'sentiment':
        # Find past periods where sentiment features had similar values
        key_cols = [c for c in ['vix', 'vix_change', 'dxy_change', 'vol_divergence'] if c in feature_cols]
    elif dominant_group == 'macro':
        key_cols = [c for c in ['rate_diff', 'yield_spread', 'event_surprise', 'event_vol_interaction'] if c in feature_cols]
    elif dominant_group == 'volatility':
        key_cols = [c for c in ['atr_pct_short', 'vol_ratio_5_60', 'range_expansion', 'hurst_exponent'] if c in feature_cols]
    else:
        key_cols = [c for c in ['atr_pct_short', 'vol_ratio_5_60'] if c in feature_cols]

    if not key_cols:
        return list(range(max(0, shift_idx - TRAIN_BARS), shift_idx))

    # Compute similarity: euclidean distance of key features
    recent_profile = recent[key_cols].mean().values
    all_data = df.iloc[:shift_idx][key_cols].fillna(0).values

    distances = np.sqrt(np.sum((all_data - recent_profile) ** 2, axis=1))
    similar_indices = np.argsort(distances)[:n_similar]

    return similar_indices.tolist()


def train_classifier(X, y, params=None):
    """Train with safety: ensure all 3 classes present."""
    if params is None:
        params = {**BASE_PARAMS, 'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 300, 'reg_alpha': 0.1, 'reg_lambda': 3.0}
    # Ensure all 3 classes exist in training data
    unique = np.unique(y)
    if len(unique) < 3:
        # Add synthetic samples for missing classes
        for cls in [0, 1, 2]:
            if cls not in unique:
                # Duplicate a random row with the missing label
                idx = np.random.randint(0, len(X))
                X = np.vstack([X, X[idx:idx+1]])
                y = np.append(y, cls)
    model = xgb.XGBClassifier(**params)
    model.fit(X, y, verbose=False)
    return model


def tune_and_train(X_train, y_train, X_val, y_val):
    """Grid search for best params, return best model."""
    best_acc = 0
    best_params = None
    
    for lr in PARAM_GRID['learning_rate']:
        for depth in PARAM_GRID['max_depth']:
            for n_est in PARAM_GRID['n_estimators']:
                for alpha in PARAM_GRID['reg_alpha']:
                    for lam in PARAM_GRID['reg_lambda']:
                        params = {
                            **BASE_PARAMS,
                            'learning_rate': lr,
                            'max_depth': depth,
                            'n_estimators': n_est,
                            'reg_alpha': alpha,
                            'reg_lambda': lam,
                        }
                        model = train_classifier(X_train, y_train, params)
                        preds = model.predict(X_val)
                        acc = accuracy_score(y_val, preds)
                        if acc > best_acc:
                            best_acc = acc
                            best_params = params.copy()
    
    # Retrain on full data with best params
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    best_model = train_classifier(X_full, y_full, best_params)
    return best_model, best_params, best_acc


def run_regime_experiment(pair_name):
    print(f"\n{'='*60}")
    print(f"Volatility Regime Classifier — {pair_name}")
    print(f"{'='*60}")

    df = build_regime_dataset(pair_name)
    feature_cols = get_feature_cols(df)

    # Walk-forward
    cursor = TRAIN_BARS
    total = len(df)

    # 4 strategies
    strategies = {
        'no_retrain': {'model': None, 'retrains': 0, 'preds': [], 'method': 'static'},
        'blind_monthly': {'model': None, 'retrains': 0, 'preds': [], 'method': 'blind'},
        'shap_guided': {'model': None, 'retrains': 0, 'preds': [], 'method': 'shap'},
        'regime_matched': {'model': None, 'retrains': 0, 'preds': [], 'method': 'matched'},
    }

    all_actuals = []
    all_dates = []
    step = 0

    # Initial training with tuning
    train_start = max(0, cursor - TRAIN_BARS)
    split_point = train_start + int((cursor - train_start) * 0.8)
    X_train_init = df.iloc[train_start:split_point][feature_cols].values
    y_train_init = df.iloc[train_start:split_point]['target_regime'].values
    X_val_init = df.iloc[split_point:cursor][feature_cols].values
    y_val_init = df.iloc[split_point:cursor]['target_regime'].values
    
    print(f"  Tuning hyperparameters ({len(PARAM_GRID['learning_rate'])*len(PARAM_GRID['max_depth'])*len(PARAM_GRID['n_estimators'])*len(PARAM_GRID['reg_alpha'])*len(PARAM_GRID['reg_lambda'])} combos)...")
    best_model, best_params, best_val_acc = tune_and_train(X_train_init, y_train_init, X_val_init, y_val_init)
    print(f"  Best params: lr={best_params['learning_rate']}, depth={best_params['max_depth']}, n_est={best_params['n_estimators']}, alpha={best_params['reg_alpha']}, lambda={best_params['reg_lambda']}")
    print(f"  Best val accuracy: {best_val_acc:.4f}")
    
    TUNED_PARAMS = best_params
    for s in strategies.values():
        s['model'] = train_classifier(df.iloc[train_start:cursor][feature_cols].values, df.iloc[train_start:cursor]['target_regime'].values, TUNED_PARAMS)

    # SHAP explainer for guided strategies
    explainer = shap.TreeExplainer(strategies['shap_guided']['model'])

    # Feature group mapping
    groups_path = os.path.join(PROCESSED_DIR, 'feature_groups.json')
    with open(groups_path) as f:
        FEATURE_GROUPS = json.load(f)

    def get_dominant_group(shap_vals, feat_cols):
        mean_abs = np.abs(shap_vals).mean(axis=0)
        # Handle multi-class: average across classes
        if len(mean_abs.shape) > 1:
            mean_abs = mean_abs.mean(axis=0)
        group_imp = {}
        for feat, imp in zip(feat_cols, mean_abs):
            found = False
            for gname, gfeats in FEATURE_GROUPS.items():
                if feat in gfeats:
                    group_imp[gname] = group_imp.get(gname, 0) + imp
                    found = True
                    break
            if not found:
                group_imp['other'] = group_imp.get('other', 0) + imp
        return max(group_imp, key=group_imp.get) if group_imp else 'other'

    while cursor + STEP_BARS <= total:
        step_end = cursor + STEP_BARS
        chunk = df.iloc[cursor:step_end]
        X_chunk = chunk[feature_cols].values
        y_chunk = chunk['target_regime'].values

        # Predict with all strategies
        for s in strategies.values():
            preds = s['model'].predict(X_chunk)
            s['preds'].extend(preds)

        all_actuals.extend(y_chunk)
        all_dates.extend(chunk['datetime_utc'].values)

        # Check for regime change in this chunk
        regime_changes = chunk['regime_changed'].sum()
        has_regime_change = regime_changes > 0

        train_start = max(0, step_end - TRAIN_BARS)
        X_train_full = df.iloc[train_start:step_end][feature_cols].values
        y_train_full = df.iloc[train_start:step_end]['target_regime'].values

        # Strategy B: Blind monthly — always retrain
        strategies['blind_monthly']['model'] = train_classifier(X_train_full, y_train_full, TUNED_PARAMS)
        strategies['blind_monthly']['retrains'] += 1

        if has_regime_change:
            # Get SHAP attribution for this chunk
            try:
                shap_vals = explainer.shap_values(X_chunk)
                dominant = get_dominant_group(shap_vals, feature_cols)
            except:
                dominant = 'other'

            # Strategy C: SHAP-guided — retrain only on non-technical shifts
            if dominant != 'technical':
                strategies['shap_guided']['model'] = train_classifier(X_train_full, y_train_full, TUNED_PARAMS)
                strategies['shap_guided']['retrains'] += 1
                # Update explainer
                explainer = shap.TreeExplainer(strategies['shap_guided']['model'])

            # Strategy D: Regime-matched — retrain on similar historical regimes
            similar_idx = find_similar_regimes(df, step_end, dominant, feature_cols)
            if len(similar_idx) > 50:
                X_matched = df.iloc[similar_idx][feature_cols].values
                y_matched = df.iloc[similar_idx]['target_regime'].values
                strategies['regime_matched']['model'] = train_classifier(X_matched, y_matched, TUNED_PARAMS)
                strategies['regime_matched']['retrains'] += 1

        step += 1
        cursor = step_end

        if step % 10 == 0:
            acc_none = accuracy_score(all_actuals[-STEP_BARS:], strategies['no_retrain']['preds'][-STEP_BARS:])
            acc_matched = accuracy_score(all_actuals[-STEP_BARS:], strategies['regime_matched']['preds'][-STEP_BARS:])
            print(f"  Step {step} | Static:{acc_none:.3f} Matched:{acc_matched:.3f}")

    # Results
    actuals = np.array(all_actuals)

    print(f"\n{'='*60}")
    print(f"RESULTS — {pair_name}")
    print(f"{'='*60}")

    summary = {}
    for sname, s in strategies.items():
        preds = np.array(s['preds'])
        acc = accuracy_score(actuals, preds)
        f1 = f1_score(actuals, preds, average='macro')
        retrains = s['retrains']

        print(f"\n  {sname}:")
        print(f"    Accuracy: {acc:.4f}")
        print(f"    Macro F1: {f1:.4f}")
        print(f"    Retrains: {retrains}")
        print(f"    Classification Report:")
        print(classification_report(actuals, preds, target_names=['LOW', 'MEDIUM', 'HIGH'], zero_division=0))

        summary[sname] = {
            'accuracy': round(acc, 4),
            'macro_f1': round(f1, 4),
            'retrains': retrains,
        }

    # Improvement table
    base_acc = summary['no_retrain']['accuracy']
    print(f"\n  Improvement over no-retrain (acc={base_acc:.4f}):")
    for sname in ['blind_monthly', 'shap_guided', 'regime_matched']:
        s = summary[sname]
        imp = (s['accuracy'] - base_acc) / base_acc * 100
        print(f"    {sname:<20s} acc={s['accuracy']:.4f} ({imp:+.2f}%) | {s['retrains']} retrains")
        summary[sname]['improvement_pct'] = round(imp, 2)

    # Save
    with open(os.path.join(RESULTS_DIR, f'{pair_name}_regime_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Save predictions for plotting
    pred_df = pd.DataFrame({
        'datetime_utc': all_dates,
        'actual': actuals,
        **{f'pred_{s}': strategies[s]['preds'] for s in strategies},
    })
    pred_df.to_csv(os.path.join(RESULTS_DIR, f'{pair_name}_regime_predictions.csv'), index=False)

    print(f"\n  Saved to results/regime/")
    return summary


if __name__ == '__main__':
    all_summaries = {}
    for pair in ['EURUSD', 'GBPJPY', 'XAUUSD']:
        summary = run_regime_experiment(pair)
        all_summaries[pair] = summary

    print(f"\n{'='*60}")
    print("REGIME CLASSIFIER — Cross-Pair Summary")
    print(f"{'='*60}")
    print(f"\n{'Pair':<10} {'Static':<10} {'Blind':<10} {'SHAP':<10} {'Matched':<10}")
    print("-" * 50)
    for pair, s in all_summaries.items():
        print(f"{pair:<10} {s['no_retrain']['accuracy']:<10.4f} "
              f"{s['blind_monthly']['accuracy']:<10.4f} "
              f"{s['shap_guided']['accuracy']:<10.4f} "
              f"{s['regime_matched']['accuracy']:<10.4f}")

    with open(os.path.join(RESULTS_DIR, 'regime_overall.json'), 'w') as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved: results/regime/regime_overall.json")
