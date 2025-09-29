"""Batch evaluator for experiment series reports.

This thin wrapper reads a series report JSON, identifies experiment output
directories that still require deferred evaluation, and runs the standard
``evaluate_saved_run`` helper on each when not in dry-run mode.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from .evaluate_saved import evaluate_saved_run


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


def _has_evaluation_summary(run_dir: Path) -> bool:
    summary = run_dir / "evaluation_summary.json"
    if summary.exists():
        return True

    readme = run_dir / "README.md"
    if not readme.exists():
        return False
    content = readme.read_text(encoding="utf-8", errors="ignore")
    return _DEFERRED_MARKER not in content


def _rewrite_readme(run_dir: Path) -> None:
    readme = run_dir / "README.md"
    lines = [
        "# Evaluation Completed\n\n",
        "Deferred scoring was finalized with `evaluate_saved_series`.\n\n",
        "Artifacts now include detailed results, summaries, and evaluation_summary.json.\n",
    ]
    readme.write_text("".join(lines), encoding="utf-8")


def _write_summary_marker(run_dir: Path) -> None:
    payload = {
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
        "tool": "evaluate_saved_series",
    }
    summary_path = run_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def process_report(report_path: Path | str, *, dry_run: bool, max_workers: int = 8) -> SeriesProcessResult:
    report = Path(report_path).expanduser().resolve()
    payload = _load_report(report)

    pending: List[Path] = []
    for run_dir in _iter_run_dirs(payload):
        if _has_evaluation_summary(run_dir):
            continue
        pending.append(run_dir)
        if dry_run:
            continue
        evaluate_saved_run(run_dir, max_workers=max_workers)
        _rewrite_readme(run_dir)
        _write_summary_marker(run_dir)

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
    args = parser.parse_args(argv)

    result = process_report(args.report, dry_run=args.dry_run, max_workers=args.max_workers)

    if args.dry_run:
        for run_dir in result.pending_dirs:
            print(run_dir)
    return result


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
