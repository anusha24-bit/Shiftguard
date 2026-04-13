"""Rebuild the reviewer-facing ShiftGuard artifacts end to end."""
from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEAN_PATTERNS = [
    'data/processed/feature_groups.json',
    'data/processed/EURUSD_features.csv',
    'data/processed/GBPJPY_features.csv',
    'data/processed/XAUUSD_features.csv',
    'results/predictions/xgboost_*.csv',
    'results/predictions/xgboost_*.json',
    'results/detection/*',
    'results/attribution/*',
    'results/decisions/*_decisions.csv',
    'results/decisions/*_retrain_queue.csv',
    'results/retraining/*_retraining_results.csv',
    'results/retraining/*_retraining_summary.json',
    'results/retraining/retraining_overall_summary.json',
    'results/winrate/*_winrate_trades.csv',
    'results/winrate/*_winrate_summary.json',
    'results/winrate/winrate_overall.json',
    'results/figures/statistical_tests.json',
    'results/figures/equity_curves.png',
    'results/figures/profit_factor_comparison.png',
    'results/figures/regime_confusion_matrix.png',
    'results/figures/winrate_comparison.png',
]


def run_script(relative_path: str, argv: Sequence[str] | None = None) -> None:
    script_path = PROJECT_ROOT / relative_path
    print(f"\n{'=' * 72}")
    print(f"Running {relative_path}")
    print(f"{'=' * 72}")
    old_argv = sys.argv[:]
    sys.argv = [str(script_path)] + list(argv or [])
    runpy.run_path(str(script_path), run_name='__main__')
    sys.argv = old_argv


def clean_outputs() -> None:
    for pattern in CLEAN_PATTERNS:
        for path in PROJECT_ROOT.glob(pattern):
            if path.is_file():
                path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", default=['EURUSD', 'GBPJPY', 'XAUUSD'])
    parser.add_argument("--fast", action="store_true", help="Use the smaller hyperparameter grid in main_xgboost.")
    parser.add_argument("--clean", action="store_true", help="Delete generated reviewer-facing outputs before rerunning.")
    parser.add_argument(
        "--skip-build-dataset",
        action="store_true",
        help="Reuse the existing processed feature matrices instead of rebuilding them.",
    )
    parser.add_argument("--skip-winrate", action="store_true", help="Skip the supplementary win-rate experiment.")
    parser.add_argument("--skip-figures", action="store_true", help="Skip statistical/figure generation.")
    args = parser.parse_args()

    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('MPLCONFIGDIR', str(PROJECT_ROOT / '.mpl-cache'))

    if args.clean:
        clean_outputs()

    pair_args = ["--pairs", *args.pairs]
    if not args.skip_build_dataset:
        run_script(os.path.join('src', 'features', 'build_dataset.py'))

    pipeline_args = pair_args + (['--fast'] if args.fast else []) + ['--auto-confirm-decisions']
    run_script(os.path.join('src', 'run_pipeline.py'), pipeline_args)

    if not args.skip_winrate:
        run_script(os.path.join('src', 'models', 'winrate_experiment.py'), pair_args)

    if not args.skip_figures:
        run_script(os.path.join('src', 'analysis', 'generate_figures.py'), pair_args)


if __name__ == '__main__':
    main()
