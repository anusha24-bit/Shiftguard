"""
Canonical ShiftGuard pipeline runner.

This script runs the submission-facing pipeline in order:
1. train the monitored XGBoost model,
2. detect scheduled/unexpected/performance shifts,
3. explain detected shifts with SHAP,
4. run selective retraining if reviewed decisions already exist.

Usage:
    python src/run_pipeline.py
"""
import os
import runpy
import sys
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DECISIONS_DIR = os.path.join(PROJECT_ROOT, "results", "decisions")


def run_script(relative_path, argv=None):
    script_path = os.path.join(PROJECT_ROOT, relative_path)
    print(f"\n{'=' * 70}")
    print(f"Running {relative_path}")
    print(f"{'=' * 70}")
    old_argv = sys.argv[:]
    sys.argv = [script_path] + (argv or [])
    runpy.run_path(script_path, run_name="__main__")
    sys.argv = old_argv


def has_review_decisions():
    if not os.path.exists(DECISIONS_DIR):
        return False
    return any(name.endswith("_decisions.csv") for name in os.listdir(DECISIONS_DIR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", default=["EURUSD", "GBPJPY", "XAUUSD"])
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    pair_args = ["--pairs", *args.pairs]
    model_args = pair_args + (["--fast"] if args.fast else [])

    run_script(os.path.join("src", "models", "main_xgboost.py"), model_args)
    run_script(os.path.join("src", "detection", "engine.py"), pair_args)
    run_script(os.path.join("src", "attribution", "shap_analysis.py"), pair_args)

    if has_review_decisions():
        run_script(os.path.join("src", "retraining", "selective.py"))
    else:
        print(f"\n{'=' * 70}")
        print("Pipeline paused for human review")
        print(f"{'=' * 70}")
        print("Next step: streamlit run src/dashboard/app.py")
        print("After reviewing shifts, rerun this script or run python src/retraining/selective.py")
