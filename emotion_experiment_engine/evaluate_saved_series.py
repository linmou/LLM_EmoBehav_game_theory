"""Batch evaluator for experiment series reports.

This thin wrapper reads a series report JSON, identifies experiment output
directories that still require deferred evaluation, and runs the standard
``evaluate_saved_run`` helper on each when not in dry-run mode.

It also supports a simple watch mode (polling the report file) so you can run
evaluation alongside an ongoing experiment series and automatically score runs
as soon as they transition to ``status == "completed"``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping


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


def _iter_experiments(report_payload: Mapping[str, object]) -> Iterable[MutableMapping[str, object]]:
    experiments = report_payload.get("experiments", {})
    if not isinstance(experiments, dict):
        return
    for exp in experiments.values():
        if isinstance(exp, dict):
            yield exp


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


def _evaluate_saved_run(run_dir: Path, *, max_workers: int) -> None:
    from .evaluate_saved import evaluate_saved_run

    evaluate_saved_run(run_dir, max_workers=max_workers)


def _is_terminal(report_payload: Mapping[str, object]) -> bool:
    experiments = report_payload.get("experiments", {})
    if not isinstance(experiments, dict):
        return True
    for exp in experiments.values():
        if not isinstance(exp, dict):
            continue
        status = str(exp.get("status", "")).strip().lower()
        if status in {"pending", "running"}:
            return False
    return True


def _all_completed_evaluated(report_payload: Mapping[str, object]) -> bool:
    for exp in _iter_experiments(report_payload):
        status = str(exp.get("status", "")).strip().lower()
        if status != "completed":
            continue
        output_dir = exp.get("output_dir")
        if not output_dir:
            continue
        run_dir = Path(str(output_dir)).expanduser().resolve()
        if not run_dir.exists():
            continue
        if not _has_evaluation_summary(run_dir):
            return False
    return True


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
    for exp in _iter_experiments(payload):
        status = str(exp.get("status", "")).strip().lower()
        if status != "completed":
            continue
        output_dir = exp.get("output_dir")
        if not output_dir:
            continue
        run_dir = Path(str(output_dir)).expanduser().resolve()
        if not run_dir.exists():
            continue
        completed = _has_evaluation_summary(run_dir)
        if completed and continue_completed:
            continue
        pending.append(run_dir)
        if dry_run:
            LOGGER.info("Pending deferred run: %s", run_dir)
            continue
        LOGGER.info("Evaluating deferred run: %s", run_dir)
        _evaluate_saved_run(run_dir, max_workers=max_workers)
        LOGGER.info("Completed deferred run: %s", run_dir)
        assert _check_summary_results(run_dir), f"Summary results not found: {run_dir}"
                
    return SeriesProcessResult(report_path=report, pending_dirs=pending)


def watch_report(
    report_path: Path | str,
    *,
    poll_interval_seconds: float = 30.0,
    max_workers: int = 8,
    continue_completed: bool = True,
) -> None:
    report = Path(report_path).expanduser().resolve()

    while True:
        payload = _load_report(report)

        try:
            process_report(
                report,
                dry_run=False,
                max_workers=max_workers,
                continue_completed=continue_completed,
            )
        except Exception:  # keep watching on evaluation errors
            LOGGER.exception("Deferred evaluation attempt failed; will retry")
        else:
            if _is_terminal(payload) and _all_completed_evaluated(payload):
                return

        time.sleep(max(0.0, poll_interval_seconds))


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
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Poll report until all experiments are terminal (no pending/running); evaluate completed runs as they finish.",
    )
    parser.add_argument(
        "--poll-interval-secs",
        type=float,
        default=30.0,
        help="Seconds to wait between report polls when --watch is enabled.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    if args.watch:
        watch_report(
            args.report,
            poll_interval_seconds=args.poll_interval_secs,
            max_workers=args.max_workers,
            continue_completed=args.continue_completed,
        )
        return SeriesProcessResult(
            report_path=Path(args.report).expanduser().resolve(),
            pending_dirs=[],
        )

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
