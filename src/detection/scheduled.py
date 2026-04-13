"""
Scheduled Shift Detector
Detects distribution shifts around known economic calendar events.
Uses KS test + MMD on pre-event vs post-event feature windows.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def compute_mmd(X, Y, gamma=1.0):
    """
    Maximum Mean Discrepancy (MMD) with RBF kernel.
    Measures distance between two multivariate distributions.
    """
    from scipy.spatial.distance import cdist
    XX = cdist(X, X, 'sqeuclidean')
    YY = cdist(Y, Y, 'sqeuclidean')
    XY = cdist(X, Y, 'sqeuclidean')

    K_XX = np.exp(-gamma * XX)
    K_YY = np.exp(-gamma * YY)
    K_XY = np.exp(-gamma * XY)

    n = len(X)
    m = len(Y)

    mmd = K_XX.sum() / (n * n) + K_YY.sum() / (m * m) - 2 * K_XY.sum() / (n * m)
    return mmd


def detect_scheduled_shifts(df, calendar_df, feature_cols, window_size=60, alpha=0.05):
    """
    Detect shifts around scheduled economic events.

    For each high-impact event date:
    1. Take pre-event window (window_size 4H bars before)
    2. Take post-event window (window_size 4H bars after)
    3. Run KS test per feature + MMD on full feature vector
    4. If significant → scheduled shift

    Args:
        df: Feature dataframe with datetime_utc column
        calendar_df: Economic calendar with date, impact_level columns
        feature_cols: List of feature column names to test
        window_size: Number of 4H bars before/after event (60 bars = 10 trading days)
        alpha: Significance threshold for KS test

    Returns:
        List of detected shift dicts
    """
    df = df.copy()
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])

    # Get high-impact event dates
    high_events = calendar_df[calendar_df['impact_level'] == 'High'].copy()
    high_events['date'] = pd.to_datetime(high_events['date'])
    event_dates = sorted(high_events['date'].unique())

    candidates = []

    for event_date in event_dates:
        # Find the bar index closest to this event date
        event_mask = df['datetime_utc'].dt.date == event_date.date()
        event_indices = df.index[event_mask]

        if len(event_indices) == 0:
            continue

        event_idx = event_indices[0]

        # Pre and post windows
        pre_start = max(0, event_idx - window_size)
        post_end = min(len(df), event_idx + window_size)

        if event_idx - pre_start < 20 or post_end - event_idx < 20:
            continue  # not enough data

        pre_window = df.iloc[pre_start:event_idx][feature_cols]
        post_window = df.iloc[event_idx:post_end][feature_cols]

        # Compare only shared fully-observed features so each column matches across windows.
        valid_cols = [
            col for col in feature_cols
            if pre_window[col].notna().all() and post_window[col].notna().all()
        ]

        if not valid_cols:
            continue

        pre_data = pre_window[valid_cols].values
        post_data = post_window[valid_cols].values

        if pre_data.shape[0] < 20 or post_data.shape[0] < 20:
            continue

        # KS test per feature
        ks_pvalues = []
        ks_stats = []

        for i in range(len(valid_cols)):
            pre_col = pre_data[:, i]
            post_col = post_data[:, i]

            # Skip if constant
            if np.std(pre_col) == 0 and np.std(post_col) == 0:
                continue

            stat, pval = ks_2samp(pre_col, post_col, method='asymp')
            ks_pvalues.append(pval)
            ks_stats.append(stat)

        if len(ks_pvalues) == 0:
            continue

        # Count significant features
        significant_count = sum(1 for p in ks_pvalues if p < alpha)
        significant_ratio = significant_count / len(ks_pvalues)

        # MMD on full feature vector (subsample for speed)
        max_samples = min(100, pre_data.shape[0], post_data.shape[0])
        mmd_score = compute_mmd(
            pre_data[:max_samples],
            post_data[:max_samples],
            gamma=1.0 / pre_data.shape[1]
        )

        # Get event names for this date
        event_names = high_events[high_events['date'] == event_date]['event_name'].tolist()

        candidates.append({
            'datetime_utc': pd.Timestamp(event_date),
            'type': 'scheduled',
            'significant_features_ratio': float(significant_ratio),
            'mean_ks_stat': float(np.mean(ks_stats)),
            'mmd_score': float(mmd_score),
            'event_names': '; '.join(event_names[:3]),
            'n_significant': int(significant_count),
            'n_tested': int(len(ks_pvalues)),
        })

    if not candidates:
        return []

    cand_df = pd.DataFrame(candidates).sort_values('datetime_utc').reset_index(drop=True)

    # Use pair-specific empirical thresholds so scheduled alerts represent the strongest event-driven shifts,
    # rather than nearly every high-impact calendar day.
    ratio_thr = max(0.45, cand_df['significant_features_ratio'].quantile(0.75))
    ks_thr = max(0.30, cand_df['mean_ks_stat'].quantile(0.75))
    mmd_thr = cand_df['mmd_score'].quantile(0.75)

    selected = cand_df[
        (cand_df['significant_features_ratio'] >= ratio_thr)
        & (
            (cand_df['mean_ks_stat'] >= ks_thr)
            | (cand_df['mmd_score'] >= mmd_thr)
        )
    ].copy()

    if selected.empty:
        return []

    # Combined score for severity and short-window deduping.
    selected['score'] = (
        selected['significant_features_ratio'].rank(pct=True)
        + selected['mean_ks_stat'].rank(pct=True)
        + selected['mmd_score'].rank(pct=True)
    ) / 3.0
    selected['severity'] = np.clip(np.ceil(selected['score'] * 5), 1, 5).astype(int)

    deduped = []
    cooldown_days = 3
    for _, row in selected.sort_values(['datetime_utc', 'score'], ascending=[True, False]).iterrows():
        if deduped:
            last_dt = deduped[-1]['datetime_utc']
            if (row['datetime_utc'] - last_dt).days <= cooldown_days:
                if row['score'] > deduped[-1]['score']:
                    deduped[-1] = row.to_dict()
                continue
        deduped.append(row.to_dict())

    shifts = []
    for row in deduped:
        shifts.append({
            'datetime_utc': str(pd.Timestamp(row['datetime_utc']).date()),
            'type': 'scheduled',
            'severity': int(row['severity']),
            'significant_features_ratio': round(row['significant_features_ratio'], 3),
            'mean_ks_stat': round(row['mean_ks_stat'], 4),
            'mmd_score': round(row['mmd_score'], 6),
            'event_names': row['event_names'],
            'n_significant': int(row['n_significant']),
            'n_tested': int(row['n_tested']),
        })

    return shifts
