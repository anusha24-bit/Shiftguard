"""
Canonical ShiftGuard pipeline runner.

This script runs the submission-facing pipeline in order:
1. train the monitored XGBoost model,
2. detect scheduled/unexpected/performance shifts,
3. explain detected shifts with SHAP,
4. optionally run selective retraining when approved decisions already exist.

Usage:
    python src/run_pipeline.py
"""
from __future__ import annotations

import os
import runpy
import sys
import argparse
from typing import Sequence
import pandas as pd

from src.dashboard.decision_utils import auto_confirm_from_detection

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DECISIONS_DIR = os.path.join(PROJECT_ROOT, "results", "decisions")
APPROVED_DECISIONS = {
    "confirm",
    "auto_confirm",
    "reclassify_to_scheduled",
    "reclassify_to_unexpected",
}


def run_script(relative_path: str, argv: Sequence[str] | None = None) -> None:
    script_path = os.path.join(PROJECT_ROOT, relative_path)
    print(f"\n{'=' * 70}")
    print(f"Running {relative_path}")
    print(f"{'=' * 70}")
    old_argv = sys.argv[:]
    sys.argv = [script_path] + (argv or [])
    runpy.run_path(script_path, run_name="__main__")
    sys.argv = old_argv


def has_review_decisions(pairs: Sequence[str]) -> bool:
    if not os.path.exists(DECISIONS_DIR):
        return False
    for pair in pairs:
        decisions_path = os.path.join(DECISIONS_DIR, f"{pair}_decisions.csv")
        if not os.path.exists(decisions_path):
            return False
        decisions = pd.read_csv(decisions_path)
        if decisions.empty or "decision" not in decisions.columns:
            return False
        if not decisions["decision"].isin(APPROVED_DECISIONS).any():
            return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", default=["EURUSD", "GBPJPY", "XAUUSD"])
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "--use-existing-decisions",
        action="store_true",
        help="Run retraining immediately when approved decision files already exist for the selected pairs.",
    )
    parser.add_argument(
        "--auto-confirm-decisions",
        action="store_true",
        help="Materialize the dashboard's default auto-confirm decisions from detection outputs and continue into retraining.",
    )
    args = parser.parse_args()

    pair_args = ["--pairs", *args.pairs]
    model_args = pair_args + (["--fast"] if args.fast else [])

    run_script(os.path.join("src", "models", "main_xgboost.py"), model_args)
    run_script(os.path.join("src", "detection", "engine.py"), pair_args)
    run_script(os.path.join("src", "attribution", "shap_analysis.py"), pair_args)

    if args.auto_confirm_decisions:
        total_created = sum(auto_confirm_from_detection(pair, project_root=PROJECT_ROOT) for pair in args.pairs)
        print(f"Auto-confirmed {total_created} shifts across {len(args.pairs)} pair(s).")

    if (args.use_existing_decisions or args.auto_confirm_decisions) and has_review_decisions(args.pairs):
        run_script(os.path.join("src", "retraining", "selective.py"), pair_args)
    else:
        print(f"\n{'=' * 70}")
        print("Pipeline paused for human review")
        print(f"{'=' * 70}")
        print("Next step: streamlit run src/dashboard/app.py")
        print("After reviewing shifts, rerun this script with --use-existing-decisions, use --auto-confirm-decisions, or run python src/retraining/selective.py --pairs ...")
