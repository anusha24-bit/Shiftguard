from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def normalize_datetime_key(value: object) -> str:
    parsed = pd.to_datetime(value, format='mixed', errors='coerce')
    if pd.notna(parsed):
        return str(parsed)
    return str(value)


def decisions_dir(project_root: str | os.PathLike[str] | None = None) -> Path:
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    path = root / 'results' / 'decisions'
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_decisions(pair: str, project_root: str | os.PathLike[str] | None = None) -> pd.DataFrame:
    path = decisions_dir(project_root) / f'{pair}_decisions.csv'
    if not path.exists():
        return pd.DataFrame(columns=['datetime_utc', 'decision', 'notes'])

    decisions = pd.read_csv(path)
    if decisions.empty:
        return pd.DataFrame(columns=['datetime_utc', 'decision', 'notes'])

    decisions['datetime_utc'] = decisions['datetime_utc'].map(normalize_datetime_key)
    return decisions


def save_decision(
    pair: str,
    dt: object,
    decision: str,
    notes: str,
    project_root: str | os.PathLike[str] | None = None,
) -> None:
    path = decisions_dir(project_root) / f'{pair}_decisions.csv'
    old = load_decisions(pair, project_root)
    dt_key = normalize_datetime_key(dt)
    old = old[old['datetime_utc'] != dt_key]
    new_row = pd.DataFrame([{'datetime_utc': dt_key, 'decision': decision, 'notes': notes}])
    pd.concat([old, new_row], ignore_index=True).to_csv(path, index=False)


def queue_retrain(
    pair: str,
    dt: object,
    severity: int,
    shift_type: str,
    event_name: str,
    project_root: str | os.PathLike[str] | None = None,
) -> None:
    path = decisions_dir(project_root) / f'{pair}_retrain_queue.csv'
    dt_key = normalize_datetime_key(dt)
    new_row = pd.DataFrame([{
        'datetime_utc': dt_key,
        'severity': severity,
        'type': shift_type,
        'event': event_name,
        'status': 'queued',
    }])
    if path.exists():
        existing = pd.read_csv(path)
        if not existing.empty and 'datetime_utc' in existing.columns:
            existing['datetime_utc'] = existing['datetime_utc'].map(normalize_datetime_key)
            existing = existing[existing['datetime_utc'] != dt_key]
            new_row = pd.concat([existing, new_row], ignore_index=True)
    new_row.to_csv(path, index=False)


def auto_confirm_shifts(
    pair: str,
    shifts_df: pd.DataFrame,
    project_root: str | os.PathLike[str] | None = None,
) -> int:
    if shifts_df.empty or 'datetime_utc' not in shifts_df.columns:
        return 0

    shifts = shifts_df.copy()
    shifts['datetime_utc'] = shifts['datetime_utc'].map(normalize_datetime_key)
    decisions = load_decisions(pair, project_root)
    decided = set(decisions['datetime_utc'].astype(str)) if not decisions.empty else set()
    created = 0

    for _, row in shifts.iterrows():
        dt_key = row['datetime_utc']
        if dt_key in decided:
            continue

        shift_type = row.get('type', 'unknown')
        event_name = row.get('trigger_event', row.get('event_names', row.get('event', 'auto-confirmed shift')))
        severity = int(row['severity']) if 'severity' in row and pd.notna(row['severity']) else 0
        save_decision(
            pair,
            dt_key,
            'auto_confirm',
            f"type={shift_type} | auto-confirmed by pipeline",
            project_root=project_root,
        )
        queue_retrain(pair, dt_key, severity, shift_type, str(event_name), project_root=project_root)
        created += 1

    return created


def auto_confirm_from_detection(
    pair: str,
    project_root: str | os.PathLike[str] | None = None,
) -> int:
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    shifts_path = root / 'results' / 'detection' / f'{pair}_shifts.csv'
    if not shifts_path.exists():
        return 0
    shifts = pd.read_csv(shifts_path)
    return auto_confirm_shifts(pair, shifts, project_root=project_root)
