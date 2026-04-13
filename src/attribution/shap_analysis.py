"""
Phase 5: SHAP Attribution Layer
Runs TreeSHAP on XGBoost to explain which feature group drove each detected shift.

For each shift:
1. Load pre-shift trained XGBoost model
2. Compute SHAP values on post-shift data
3. Aggregate by feature group (Technical, Volatility, Macro, Sentiment)
4. Output: "This shift was driven by Volatility (55%) + Macro (28%)"

Usage:
    python src/attribution/shap_analysis.py
"""
import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import shap

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'predictions')
DETECTION_DIR = os.path.join(PROJECT_ROOT, 'results', 'detection')
ATTRIBUTION_DIR = os.path.join(PROJECT_ROOT, 'results', 'attribution')
os.makedirs(ATTRIBUTION_DIR, exist_ok=True)

EXCLUDE_COLS = ['datetime_utc', 'date', 'session', 'target_return', 'target_direction', 'volume']

# Load feature groups mapping
GROUPS_PATH = os.path.join(PROCESSED_DIR, 'feature_groups.json')
with open(GROUPS_PATH) as f:
    FEATURE_GROUPS = json.load(f)


def get_feature_cols(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def map_feature_to_group(feature_name):
    """Map a feature name to its group. Returns 'other' if not found."""
    for group, features in FEATURE_GROUPS.items():
        if feature_name in features:
            return group
    # Check session dummies
    if feature_name.startswith('sess_'):
        return 'technical'
    return 'other'


def compute_group_attribution(shap_values, feature_names):
    """
    Aggregate SHAP values by feature group.
    Returns dict: {group_name: mean_abs_shap_contribution_pct}
    """
    # Mean absolute SHAP per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Map to groups
    group_shap = {}
    for feat, shap_val in zip(feature_names, mean_abs_shap):
        group = map_feature_to_group(feat)
        group_shap[group] = group_shap.get(group, 0) + shap_val

    # Convert to percentages
    total = sum(group_shap.values())
    if total > 0:
        group_pct = {k: round(v / total * 100, 1) for k, v in group_shap.items()}
    else:
        group_pct = {k: 0 for k in group_shap}

    # Sort by contribution
    group_pct = dict(sorted(group_pct.items(), key=lambda x: -x[1]))

    return group_pct


def analyze_shift(model, df, feature_cols, shift_idx, window=60):
    """
    Run SHAP analysis around a single shift point.

    Args:
        model: Trained XGBoost model
        df: Full feature dataframe
        feature_cols: List of feature columns
        shift_idx: Index in df where shift was detected
        window: Number of bars before/after shift to analyze

    Returns:
        Dict with group attribution and top features
    """
    post_start = shift_idx
    post_end = min(len(df), shift_idx + window)

    if post_end - post_start < 10:
        return None

    post_data = df.iloc[post_start:post_end][feature_cols].values

    # TreeSHAP (exact, fast for tree models)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(post_data)

    # Group attribution
    group_attr = compute_group_attribution(shap_values, feature_cols)

    # Top 10 individual features
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:10]
    top_features = [(feature_cols[i], round(float(mean_abs[i]), 6)) for i in top_indices]

    return {
        'group_attribution': group_attr,
        'top_features': top_features,
        'n_samples': post_end - post_start,
    }


def run_attribution(pair_name):
    """Run SHAP attribution for all detected shifts in one pair."""
    print(f"\n{'='*60}")
    print(f"SHAP Attribution — {pair_name}")
    print(f"{'='*60}")

    # Load feature data
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f'{pair_name}_features.csv'))
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    feature_cols = get_feature_cols(df)

    # Load XGBoost model
    model_path = os.path.join(RESULTS_DIR, f'xgboost_{pair_name}.json')
    if not os.path.exists(model_path):
        print(f"  Skipping {pair_name} - monitored model not found at {model_path}")
        return pd.DataFrame()
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    print(f"  Model loaded: {model_path}")

    # Load detected shifts
    shifts_path = os.path.join(DETECTION_DIR, f'{pair_name}_shifts.csv')
    shifts_df = pd.read_csv(shifts_path)
    print(f"  Shifts loaded: {len(shifts_df)}")

    # Sample shifts for analysis (too many to SHAP all of them)
    # Take top severity + random sample
    if len(shifts_df) > 50:
        # Take all high severity + random sample of the rest
        high_sev = shifts_df[shifts_df.get('severity', 0) >= 3] if 'severity' in shifts_df.columns else shifts_df.head(0)
        remaining = shifts_df.drop(high_sev.index)
        sample_size = min(30, len(remaining))
        sampled = pd.concat([high_sev, remaining.sample(sample_size, random_state=42)])
        sampled = sampled.drop_duplicates().sort_values('datetime_utc')
    else:
        sampled = shifts_df

    print(f"  Analyzing {len(sampled)} shifts (sampled from {len(shifts_df)})...")

    # Run SHAP for each shift
    results = []
    for idx, shift_row in sampled.iterrows():
        shift_dt = pd.to_datetime(shift_row['datetime_utc'])

        # Find closest index in df
        time_diff = (df['datetime_utc'] - shift_dt).abs()
        closest_idx = time_diff.idxmin()

        attr = analyze_shift(model, df, feature_cols, closest_idx, window=60)

        if attr is None:
            continue

        result = {
            'datetime_utc': str(shift_dt),
            'shift_type': shift_row.get('type', 'unknown'),
            'severity': shift_row.get('severity', 0),
            **{f'group_{k}': v for k, v in attr['group_attribution'].items()},
            'dominant_group': max(attr['group_attribution'], key=attr['group_attribution'].get),
            'top_feature_1': attr['top_features'][0][0] if attr['top_features'] else '',
            'top_feature_1_shap': attr['top_features'][0][1] if attr['top_features'] else 0,
            'top_feature_2': attr['top_features'][1][0] if len(attr['top_features']) > 1 else '',
            'top_feature_3': attr['top_features'][2][0] if len(attr['top_features']) > 2 else '',
        }
        results.append(result)

    results_df = pd.DataFrame(results)

    # Summary stats
    if not results_df.empty and 'dominant_group' in results_df.columns:
        print(f"\n  Dominant group distribution:")
        for group, count in results_df['dominant_group'].value_counts().items():
            pct = count / len(results_df) * 100
            print(f"    {group:<15s} {count:>4d} shifts ({pct:.1f}%)")

    # Save
    out_path = os.path.join(ATTRIBUTION_DIR, f'{pair_name}_attribution.csv')
    results_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    # Save summary
    summary = {
        'pair': pair_name,
        'total_shifts_analyzed': len(results_df),
        'dominant_group_counts': results_df['dominant_group'].value_counts().to_dict() if not results_df.empty else {},
    }

    # Mean group percentages across all shifts
    group_cols = [c for c in results_df.columns if c.startswith('group_')]
    if group_cols:
        for col in group_cols:
            summary[f'mean_{col}'] = round(results_df[col].mean(), 1)

    json_path = os.path.join(ATTRIBUTION_DIR, f'{pair_name}_attribution_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    return results_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", default=['EURUSD', 'GBPJPY', 'XAUUSD'])
    args = parser.parse_args()

    for pair in args.pairs:
        run_attribution(pair)

    print(f"\n{'='*60}")
    print("SHAP Attribution complete. Results in results/attribution/")
    print(f"{'='*60}")
