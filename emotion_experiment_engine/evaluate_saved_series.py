"""Batch evaluator for experiment series reports.

This thin wrapper reads a series report JSON, identifies experiment output
directories that still require deferred evaluation, and runs the standard
``evaluate_saved_run`` helper on each when not in dry-run mode.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from .evaluate_saved import evaluate_saved_run


LOGGER = logging.getLogger(__name__)


@dataclass
class SeriesProcessResult:
    """Captured outcome of processing a series report."""

    report_path: Path
    pending_dirs: List[Path]


_DEFERRED_MARKER = "# Evaluation Deferred"


def _load_report(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Series report does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_run_dirs(report_payload: dict) -> Iterable[Path]:
    experiments = report_payload.get("experiments", {})
    for exp in experiments.values():
        output_dir = exp.get("output_dir")
        if not output_dir:
            continue
        run_dir = Path(output_dir).expanduser().resolve()
        if run_dir.exists():
            yield run_dir


def _get_summary_path(run_dir: Path) -> Path | None:
    csv_path = run_dir / "summary_results.csv"
    if csv_path.exists():
        return csv_path
    json_path = run_dir / "summary_results.json"
    if json_path.exists():
        return json_path
    return None


def _has_evaluation_summary(run_dir: Path) -> bool:
    if _get_summary_path(run_dir):
        return True

    readme = run_dir / "README.md"
    if not readme.exists():
        return False
    content = readme.read_text(encoding="utf-8", errors="ignore")
    return _DEFERRED_MARKER not in content


def _check_summary_results(run_dir: Path) -> bool:
    return _get_summary_path(run_dir) is not None


def process_report(
    report_path: Path | str,
    *,
    dry_run: bool,
    max_workers: int = 8,
    continue_completed: bool = True,
) -> SeriesProcessResult:
    report = Path(report_path).expanduser().resolve()
    payload = _load_report(report)

    pending: List[Path] = []
    for run_dir in _iter_run_dirs(payload):
        completed = _has_evaluation_summary(run_dir)
        if completed and continue_completed:
            continue
        pending.append(run_dir)
        if dry_run:
            LOGGER.info("Pending deferred run: %s", run_dir)
            continue
        LOGGER.info("Evaluating deferred run: %s", run_dir)
        evaluate_saved_run(run_dir, max_workers=max_workers)
        LOGGER.info("Completed deferred run: %s", run_dir)
        assert _check_summary_results(run_dir), f"Summary results not found: {run_dir}"
                
    return SeriesProcessResult(report_path=report, pending_dirs=pending)


def _main(argv: List[str] | None = None) -> SeriesProcessResult:
    parser = argparse.ArgumentParser(description="Evaluate all deferred runs in a series report")
    parser.add_argument("--report", required=True, help="Path to experiment_series report JSON")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List pending run directories without executing evaluation",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Worker count forwarded to evaluate_saved",
    )
    parser.add_argument(
        "--continue",
        dest="continue_completed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip already evaluated runs (default); use --no-continue to re-score all runs.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    result = process_report(
        args.report,
        dry_run=args.dry_run,
        max_workers=args.max_workers,
        continue_completed=args.continue_completed,
    )

    if args.dry_run:
        for run_dir in result.pending_dirs:
            print(run_dir)
    return result


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
