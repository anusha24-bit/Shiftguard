"""
Scheduled Shift Detector
Detects distribution shifts around known economic calendar events.
Uses KS test + MMD on pre-event vs post-event feature windows.
"""
from __future__ import annotations

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


def parse_event_timestamp(date_value, time_value) -> pd.Timestamp | pd.NaT:
    """Parse calendar date/time into a UTC timestamp."""
    event_date = pd.to_datetime(date_value, errors='coerce')
    if pd.isna(event_date):
        return pd.NaT

    if pd.isna(time_value) or str(time_value).strip() == '':
        return event_date

    timestamp = pd.to_datetime(
        f"{event_date.strftime('%Y-%m-%d')} {str(time_value).strip()}",
        errors='coerce',
    )
    return timestamp


def align_event_timestamp_to_bar(
    bar_times: pd.Series,
    event_timestamp: pd.Timestamp,
) -> tuple[pd.Timestamp, int] | None:
    """Align a calendar event to the nearest available feature bar."""
    if pd.isna(event_timestamp) or bar_times.empty:
        return None

    time_diffs = (bar_times - event_timestamp).abs()
    event_idx = int(time_diffs.idxmin())
    return bar_times.iloc[event_idx], event_idx


def build_aligned_high_impact_events(df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
    """Attach high-impact calendar events to the nearest feature bar."""
    high_events = calendar_df[calendar_df['impact_level'] == 'High'].copy()
    if high_events.empty:
        return pd.DataFrame(columns=['event_timestamp', 'aligned_timestamp', 'aligned_index', 'event_names'])

    high_events['event_timestamp'] = high_events.apply(
        lambda row: parse_event_timestamp(row.get('date'), row.get('time_utc')),
        axis=1,
    )
    high_events = high_events.dropna(subset=['event_timestamp']).sort_values('event_timestamp')
    if high_events.empty:
        return pd.DataFrame(columns=['event_timestamp', 'aligned_timestamp', 'aligned_index', 'event_names'])

    aligned_rows = []
    bar_times = df['datetime_utc']
    for _, event_row in high_events.iterrows():
        aligned = align_event_timestamp_to_bar(bar_times, event_row['event_timestamp'])
        if aligned is None:
            continue
        aligned_timestamp, aligned_index = aligned
        aligned_rows.append({
            'event_timestamp': event_row['event_timestamp'],
            'aligned_timestamp': aligned_timestamp,
            'aligned_index': aligned_index,
            'event_name': event_row.get('event_name', ''),
        })

    if not aligned_rows:
        return pd.DataFrame(columns=['event_timestamp', 'aligned_timestamp', 'aligned_index', 'event_names'])

    aligned_df = pd.DataFrame(aligned_rows)
    return aligned_df.groupby('aligned_timestamp', as_index=False).agg(
        event_timestamp=('event_timestamp', 'min'),
        aligned_index=('aligned_index', 'min'),
        event_names=(
            'event_name',
            lambda values: '; '.join([
                name for name in pd.unique(values)
                if isinstance(name, str) and name.strip()
            ][:3]),
        ),
    )


def detect_scheduled_shifts(df, calendar_df, feature_cols, window_size=60, alpha=0.05):
    """
    Detect shifts around scheduled economic events.

    For each high-impact event time:
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
    df = df.sort_values('datetime_utc').reset_index(drop=True)

    aligned_events = build_aligned_high_impact_events(df, calendar_df)

    candidates = []

    for _, event_row in aligned_events.iterrows():
        event_timestamp = pd.Timestamp(event_row['aligned_timestamp'])
        event_idx = int(event_row['aligned_index'])

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

        candidates.append({
            'datetime_utc': event_timestamp,
            'type': 'scheduled',
            'significant_features_ratio': float(significant_ratio),
            'mean_ks_stat': float(np.mean(ks_stats)),
            'mmd_score': float(mmd_score),
            'event_names': event_row['event_names'],
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
            'datetime_utc': str(pd.Timestamp(row['datetime_utc'])),
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
