"""
Dual-Mode Shift Detection Engine
Orchestrates scheduled, unexpected, and performance drift detectors.

Usage:
    python src/detection/engine.py
"""
from __future__ import annotations

import sys
import os
import json
import argparse
from typing import Any
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.detection.scheduled import detect_scheduled_shifts
from src.detection.performance import detect_performance_drift

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'predictions')
DETECTION_DIR = os.path.join(PROJECT_ROOT, 'results', 'detection')
os.makedirs(DETECTION_DIR, exist_ok=True)

EXCLUDE_COLS = ['datetime_utc', 'date', 'session', 'target_return', 'target_direction', 'volume']

PAIR_CURRENCIES = {
    'EURUSD': ['USD', 'EUR'],
    'GBPJPY': ['GBP', 'JPY'],
    'XAUUSD': ['USD'],
}


def load_unexpected_shift_detector():
    """Import ADWIN-based detection only when the full detector is needed."""
    try:
        from src.detection.unexpected import detect_unexpected_shifts
    except ModuleNotFoundError as exc:
        if exc.name == 'river':
            raise ModuleNotFoundError(
                "Unexpected shift detection requires the optional 'river' dependency. "
                "Install requirements.txt before running the full detection engine."
            ) from exc
        raise
    return detect_unexpected_shifts


def load_prediction_window(pair_name: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Infer the active evaluation window from the monitored-model predictions."""
    pred_path = os.path.join(RESULTS_DIR, f'xgboost_{pair_name}_predictions.csv')
    if not os.path.exists(pred_path):
        return None
    pred_df = pd.read_csv(pred_path)
    if pred_df.empty or 'datetime_utc' not in pred_df.columns:
        return None
    pred_df['datetime_utc'] = pd.to_datetime(pred_df['datetime_utc'], errors='coerce')
    pred_df = pred_df.dropna(subset=['datetime_utc'])
    if pred_df.empty:
        return None
    return pred_df['datetime_utc'].min(), pred_df['datetime_utc'].max()


def filter_shifts_to_prediction_window(
    shifts: list[dict[str, Any]],
    prediction_window: tuple[pd.Timestamp, pd.Timestamp] | None,
) -> list[dict[str, Any]]:
    """Keep only shifts that fall inside the active monitored-model prediction window."""
    if prediction_window is None:
        return shifts
    window_start, window_end = prediction_window
    filtered: list[dict[str, Any]] = []
    for shift in shifts:
        shift_dt = pd.to_datetime(shift.get('datetime_utc'), errors='coerce')
        if pd.notna(shift_dt) and window_start <= shift_dt <= window_end:
            filtered.append(shift)
    return filtered


def load_calendar(pair_name: str) -> pd.DataFrame:
    """Load combined calendar for the pair's currencies."""
    currencies = PAIR_CURRENCIES.get(pair_name, ['USD'])
    dfs = []
    for curr in currencies:
        path = os.path.join(DATA_DIR, 'calendar', f'{curr}_economic_calendar.csv')
        if os.path.exists(path):
            cal = pd.read_csv(path)
            cal['date'] = pd.to_datetime(cal['date'])
            dfs.append(cal)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def run_detection(pair_name: str) -> pd.DataFrame:
    """Run all 3 detectors for one pair."""
    print(f"\n{'='*60}")
    print(f"Detection Engine — {pair_name}")
    print(f"{'='*60}")

    # Load feature data
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f'{pair_name}_features.csv'))
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    print(f"  Loaded: {len(df)} bars, {len(feature_cols)} features")

    # Load calendar
    calendar_df = load_calendar(pair_name)
    print(f"  Calendar: {len(calendar_df)} events")

    # Get set of event dates for filtering unexpected shifts
    high_events = calendar_df[calendar_df['impact_level'] == 'High']
    calendar_dates = set(high_events['date'].dt.date.astype(str).values)
    prediction_window = load_prediction_window(pair_name)

    # --- 1. Scheduled shifts ---
    print("\n  [1/3] Detecting scheduled shifts (KS + MMD)...")
    scheduled = detect_scheduled_shifts(df, calendar_df, feature_cols, window_size=60)
    scheduled = filter_shifts_to_prediction_window(scheduled, prediction_window)
    print(f"    Found: {len(scheduled)} scheduled shifts")

    # --- 2. Unexpected shifts ---
    print("\n  [2/3] Detecting unexpected shifts (ADWIN)...")
    detect_unexpected_shifts = load_unexpected_shift_detector()
    unexpected = detect_unexpected_shifts(df, feature_cols, calendar_dates, delta=0.002)
    unexpected = filter_shifts_to_prediction_window(unexpected, prediction_window)
    print(f"    Found: {len(unexpected)} unexpected shifts")

    # --- 3. Performance drift ---
    print("\n  [3/3] Detecting performance drift (DDM)...")
    pred_path = os.path.join(RESULTS_DIR, f'xgboost_{pair_name}_predictions.csv')
    perf_drifts = []
    perf_warnings = []
    if os.path.exists(pred_path):
        pred_df = pd.read_csv(pred_path)
        perf_drifts, perf_warnings = detect_performance_drift(pred_df)
        perf_drifts = filter_shifts_to_prediction_window(perf_drifts, prediction_window)
        perf_warnings = filter_shifts_to_prediction_window(perf_warnings, prediction_window)
        print(f"    Found: {len(perf_drifts)} drifts, {len(perf_warnings)} warnings")
    else:
        print(f"    Skipped — no predictions file found at {pred_path}")

    # --- Combine all shifts ---
    all_shifts = scheduled + unexpected + perf_drifts
    all_shifts_df = pd.DataFrame(all_shifts)

    if not all_shifts_df.empty:
        all_shifts_df = all_shifts_df.sort_values('datetime_utc').reset_index(drop=True)

    # Summary
    print(f"\n  SUMMARY:")
    print(f"    Scheduled shifts:   {len(scheduled)}")
    print(f"    Unexpected shifts:  {len(unexpected)}")
    print(f"    Performance drifts: {len(perf_drifts)}")
    print(f"    Total:              {len(all_shifts)}")

    # Save
    out_path = os.path.join(DETECTION_DIR, f'{pair_name}_shifts.csv')
    all_shifts_df.to_csv(out_path, index=False)
    print(f"    Saved: {out_path}")

    # Save detailed JSON
    detail = {
        'pair': pair_name,
        'scheduled': scheduled,
        'unexpected': unexpected,
        'performance_drifts': perf_drifts,
        'performance_warnings': perf_warnings[:20],  # cap for readability
        'total_shifts': len(all_shifts),
    }
    json_path = os.path.join(DETECTION_DIR, f'{pair_name}_detection_detail.json')
    with open(json_path, 'w') as f:
        json.dump(detail, f, indent=2, default=str)

    return all_shifts_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", default=['EURUSD', 'GBPJPY', 'XAUUSD'])
    args = parser.parse_args()

    for pair in args.pairs:
        shifts = run_detection(pair)

    # Cross-reference with geopolitical events (ground truth)
    print(f"\n{'='*60}")
    print("Ground Truth Comparison")
    print(f"{'='*60}")

    events_path = os.path.join(DATA_DIR, 'events', 'geopolitical_events.csv')
    if os.path.exists(events_path):
        geo = pd.read_csv(events_path)
        print(f"  Known events: {len(geo)}")
        for _, row in geo.iterrows():
            print(f"    {row['date']} | {row['event_name']} | Severity: {row['severity']}")
    else:
        print("  No geopolitical events file found")

    print(f"\n{'='*60}")
    print("Detection complete. Results in results/detection/")
    print(f"{'='*60}")
