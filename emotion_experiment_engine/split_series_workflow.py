"""Helpers for split experiment-series evaluation and merge workflows."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from .evaluate_saved_series import _is_terminal, _load_report, process_report


LOGGER = logging.getLogger(__name__)


def _load_payloads(report_paths: Sequence[Path | str]) -> list[dict]:
    return [_load_report(Path(path).expanduser().resolve()) for path in report_paths]


def _normalize_paths(report_paths: Sequence[Path | str]) -> list[Path]:
    return [Path(path).expanduser().resolve() for path in report_paths]


def launch_eval_watchers_tmux(
    *,
    report_paths: Sequence[Path | str],
    env_name: str,
    poll_interval_seconds: float,
    max_workers: int,
    session_name_prefix: str,
) -> list[str]:
    session_names: list[str] = []
    conda_sh = "/home/jjl7137/anaconda3/etc/profile.d/conda.sh"

    for idx, report_path in enumerate(_normalize_paths(report_paths)):
        session_name = f"{session_name_prefix}_{idx}"
        quoted_report = shlex.quote(str(report_path))
        inner_command = (
            f"while [ ! -f {quoted_report} ]; do sleep 5; done; "
            f"source {shlex.quote(conda_sh)} && "
            f"conda activate {shlex.quote(env_name)} && "
            "python -m emotion_experiment_engine.evaluate_saved_series "
            f"--report {quoted_report} --watch "
            f"--poll-interval-secs {poll_interval_seconds} "
            f"--max-workers {max_workers}"
        )
        command = f"bash -lc {shlex.quote(inner_command)}"
        subprocess.run(
            ["tmux", "new-session", "-d", "-c", os.getcwd(), "-s", session_name, command],
            check=True,
        )
        session_names.append(session_name)
    return session_names


def wait_for_reports_evaluated(
    *,
    report_paths: Sequence[Path | str],
    poll_interval_seconds: float,
    max_workers: int,
) -> list[Path]:
    reports = _normalize_paths(report_paths)

    while True:
        all_terminal = True
        all_evaluated = True

        for report_path in reports:
            if not report_path.exists():
                all_terminal = False
                all_evaluated = False
                continue

            payload = _load_report(report_path)
            if not _is_terminal(payload):
                all_terminal = False

            result = process_report(
                report_path,
                dry_run=False,
                max_workers=max_workers,
                continue_completed=True,
            )
            if result.pending_dirs or result.failed_dirs:
                all_evaluated = False

        if all_terminal and all_evaluated:
            return reports

        time.sleep(max(0.0, poll_interval_seconds))


def merge_series_reports(
    report_paths: Sequence[Path | str],
    *,
    merged_output_dir: Path | str,
    merged_series_name: str,
) -> Path:
    reports = _normalize_paths(report_paths)
    payloads = _load_payloads(reports)
    destination = Path(merged_output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    merged_experiments: dict[str, dict] = {}
    merged_sessions: list[dict] = []
    source_output_dirs: list[str] = []
    merged_configs: list[dict] = []

    for report_path, payload in zip(reports, payloads):
        source_output_dirs.append(str(report_path.parent.resolve()))
        merged_sessions.extend(copy.deepcopy(payload.get("sessions", [])))
        merged_configs.append(copy.deepcopy(payload.get("series_config", {})))
        experiments = payload.get("experiments", {})
        if not isinstance(experiments, dict):
            continue
        for exp_id, exp in experiments.items():
            if exp_id in merged_experiments:
                raise ValueError(f"Duplicate experiment id across split reports: {exp_id}")
            if not isinstance(exp, dict):
                raise ValueError(f"Experiment payload must be a mapping: {exp_id}")
            merged_exp = copy.deepcopy(exp)
            output_dir = merged_exp.get("output_dir")
            if output_dir:
                source_run_dir = Path(str(output_dir)).expanduser().resolve()
                link_path = destination / source_run_dir.name
                if link_path.exists() or link_path.is_symlink():
                    if link_path.resolve() != source_run_dir:
                        raise ValueError(f"Merged output collision for {link_path}")
                else:
                    os.symlink(source_run_dir, link_path, target_is_directory=True)
                merged_exp["output_dir"] = str(link_path)
            merged_experiments[exp_id] = merged_exp

    start_times = [
        datetime.fromisoformat(payload["series_start_time"])
        for payload in payloads
        if payload.get("series_start_time")
    ]
    series_start = min(start_times) if start_times else datetime.now()
    now = datetime.now()

    merged_payload = {
        "last_updated": now.isoformat(),
        "series_start_time": series_start.isoformat(),
        "series_duration_seconds": (now - series_start).total_seconds(),
        "series_name": merged_series_name,
        "series_config": {
            "source_reports": [str(path) for path in reports],
            "source_output_dirs": source_output_dirs,
            "merged_from_series_configs": merged_configs,
        },
        "sessions": merged_sessions,
        "experiments": merged_experiments,
    }

    report_name = f"{merged_series_name}_{now.strftime('%Y%m%d_%H')}_memory_experiment_report.json"
    report_path = destination / report_name
    report_path.write_text(json.dumps(merged_payload, indent=2), encoding="utf-8")
    return report_path


def wait_and_merge_reports(
    *,
    report_paths: Sequence[Path | str],
    merged_output_dir: Path | str,
    merged_series_name: str,
    poll_interval_seconds: float,
    max_workers: int,
) -> Path:
    wait_for_reports_evaluated(
        report_paths=report_paths,
        poll_interval_seconds=poll_interval_seconds,
        max_workers=max_workers,
    )
    return merge_series_reports(
        report_paths,
        merged_output_dir=merged_output_dir,
        merged_series_name=merged_series_name,
    )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split series evaluation and merge workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch_parser = subparsers.add_parser("launch-eval-watchers")
    launch_parser.add_argument("--report", action="append", required=True)
    launch_parser.add_argument("--env-name", required=True)
    launch_parser.add_argument("--poll-interval-secs", type=float, default=30.0)
    launch_parser.add_argument("--max-workers", type=int, default=8)
    launch_parser.add_argument("--session-name-prefix", default="split_eval")

    merge_parser = subparsers.add_parser("wait-and-merge")
    merge_parser.add_argument("--report", action="append", required=True)
    merge_parser.add_argument("--merged-output-dir", required=True)
    merge_parser.add_argument("--merged-series-name", required=True)
    merge_parser.add_argument("--poll-interval-secs", type=float, default=30.0)
    merge_parser.add_argument("--max-workers", type=int, default=8)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    if args.command == "launch-eval-watchers":
        sessions = launch_eval_watchers_tmux(
            report_paths=args.report,
            env_name=args.env_name,
            poll_interval_seconds=args.poll_interval_secs,
            max_workers=args.max_workers,
            session_name_prefix=args.session_name_prefix,
        )
        for session in sessions:
            print(session)
        return 0

    report_path = wait_and_merge_reports(
        report_paths=args.report,
        merged_output_dir=args.merged_output_dir,
        merged_series_name=args.merged_series_name,
        poll_interval_seconds=args.poll_interval_secs,
        max_workers=args.max_workers,
    )
    print(report_path)
    return 0


def main() -> None:
    raise SystemExit(_main())


if __name__ == "__main__":
    main()
