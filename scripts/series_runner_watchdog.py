#!/usr/bin/env python3
# Purpose: watch any shard report for stalled progress on assigned GPUs and relaunch the runner from the report when it stops making progress.

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch a shard report and resume the shard when progress stalls."
    )
    parser.add_argument("--report", required=True, help="Path to the shard report JSON")
    parser.add_argument("--series-name", required=True, help="Series name to pass to --name on resume")
    parser.add_argument("--gpus", required=True, help="Comma-separated CUDA device list, e.g. 2 or 2,3")
    parser.add_argument("--run-log", required=True, help="Path to append resumed runner logs")
    parser.add_argument("--monitor-log", required=True, help="Path to write watchdog GPU monitor logs")
    parser.add_argument(
        "--initial-pid",
        type=int,
        required=False,
        help="Existing shard runner PID to adopt before watchdog-owned restarts",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=30.0,
        help="Polling interval for report/GPU checks",
    )
    parser.add_argument(
        "--stall-seconds",
        type=float,
        default=600.0,
        help="Restart when no report progress and idle GPUs persist this long",
    )
    parser.add_argument(
        "--idle-util-threshold",
        type=float,
        default=5.0,
        help="Consider GPUs idle when utilization stays at or below this percentage",
    )
    return parser.parse_args()


def _load_report(report_path: Path) -> dict:
    return json.loads(report_path.read_text(encoding="utf-8"))


def _count_statuses(payload: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for exp in payload.get("experiments", {}).values():
        if not isinstance(exp, dict):
            continue
        status = str(exp.get("status", "")).strip().lower()
        counts[status] = counts.get(status, 0) + 1
    return counts


def _running_ids(payload: dict) -> tuple[str, ...]:
    running = []
    for exp_id, exp in payload.get("experiments", {}).items():
        if isinstance(exp, dict) and str(exp.get("status", "")).strip().lower() == "running":
            running.append(str(exp_id))
    return tuple(sorted(running))


def _report_signature(payload: dict) -> tuple[int, int, tuple[str, ...]]:
    counts = _count_statuses(payload)
    return counts.get("completed", 0), counts.get("running", 0), _running_ids(payload)


def _parse_iso8601(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _report_last_updated(payload: dict, report_path: Path) -> float:
    parsed = _parse_iso8601(payload.get("last_updated"))
    if parsed is not None:
        return parsed.timestamp()
    return report_path.stat().st_mtime


def _is_terminal(payload: dict) -> bool:
    for exp in payload.get("experiments", {}).values():
        if not isinstance(exp, dict):
            continue
        status = str(exp.get("status", "")).strip().lower()
        if status not in {"completed", "failed"}:
            return False
    return True


def _query_gpu_utils(gpu_ids: str) -> list[float]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            gpu_ids,
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    utils = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        utils.append(float(line))
    return utils


def _iter_descendants(root_pid: int) -> list[int]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid="],
        check=True,
        capture_output=True,
        text=True,
    )
    children: dict[int, list[int]] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        pid = int(parts[0])
        ppid = int(parts[1])
        children.setdefault(ppid, []).append(pid)

    ordered: list[int] = []
    stack = [root_pid]
    while stack:
        current = stack.pop()
        for child in children.get(current, []):
            ordered.append(child)
            stack.append(child)
    return ordered


def _terminate_pid_tree(pid: int) -> None:
    descendants = _iter_descendants(pid)
    for target in reversed(descendants):
        try:
            os.kill(target, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    time.sleep(3)

    for target in reversed(descendants):
        try:
            os.kill(target, 0)
        except ProcessLookupError:
            continue
        os.kill(target, signal.SIGKILL)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    os.kill(pid, signal.SIGKILL)


def _start_gpu_monitor(gpu_ids: str, monitor_log: Path) -> subprocess.Popen[str]:
    monitor_log.parent.mkdir(parents=True, exist_ok=True)
    log_file = monitor_log.open("a", encoding="utf-8")
    shell = (
        "while true; do "
        "echo \"=== $(date --iso-8601=seconds) ===\"; "
        f"nvidia-smi -i {gpu_ids} "
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu "
        "--format=csv,noheader; "
        "echo; sleep 30; done"
    )
    return subprocess.Popen(
        ["bash", "-lc", shell],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def _start_runner(report_path: Path, series_name: str, gpu_ids: str, run_log: Path) -> subprocess.Popen[str]:
    run_log.parent.mkdir(parents=True, exist_ok=True)
    log_file = run_log.open("a", encoding="utf-8")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "emotion_experiment_engine.emotion_experiment_series_runner",
            "--resume",
            str(report_path),
            "--name",
            series_name,
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )


def _should_restart(
    *,
    previous_signature: tuple[int, int, tuple[str, ...]],
    current_signature: tuple[int, int, tuple[str, ...]],
    last_progress_ts: float,
    now_ts: float,
    gpu_utils: Iterable[float],
    idle_util_threshold: float,
    stall_seconds: float,
) -> bool:
    if current_signature[1] == 0:
        return False
    if current_signature != previous_signature:
        return False
    stalled_for = now_ts - last_progress_ts
    if stalled_for < stall_seconds:
        return False
    utils = list(gpu_utils)
    if max(utils, default=0.0) <= idle_util_threshold:
        return True
    return min(utils, default=0.0) <= idle_util_threshold and stalled_for >= (2 * stall_seconds)


def main() -> int:
    args = _parse_args()
    report_path = Path(args.report).expanduser().resolve()
    run_log = Path(args.run_log).expanduser().resolve()
    monitor_log = Path(args.monitor_log).expanduser().resolve()

    payload = _load_report(report_path)
    previous_signature = _report_signature(payload)
    last_progress_ts = _report_last_updated(payload, report_path)

    runner_pid = args.initial_pid
    runner_proc: subprocess.Popen[str] | None = None
    monitor_proc = _start_gpu_monitor(args.gpus, monitor_log)
    if runner_pid is None:
        runner_proc = _start_runner(report_path, args.series_name, args.gpus, run_log)
        runner_pid = runner_proc.pid

    try:
        while True:
            payload = _load_report(report_path)
            if _is_terminal(payload):
                return 0

            current_signature = _report_signature(payload)
            report_ts = _report_last_updated(payload, report_path)
            if current_signature != previous_signature or report_ts > last_progress_ts:
                previous_signature = current_signature
                last_progress_ts = max(last_progress_ts, report_ts)

            gpu_utils = _query_gpu_utils(args.gpus)
            now_ts = datetime.now(timezone.utc).timestamp()

            if _should_restart(
                previous_signature=previous_signature,
                current_signature=current_signature,
                last_progress_ts=last_progress_ts,
                now_ts=now_ts,
                gpu_utils=gpu_utils,
                idle_util_threshold=args.idle_util_threshold,
                stall_seconds=args.stall_seconds,
            ):
                if runner_proc is not None and runner_proc.poll() is None:
                    os.killpg(runner_proc.pid, signal.SIGTERM)
                    time.sleep(3)
                    if runner_proc.poll() is None:
                        os.killpg(runner_proc.pid, signal.SIGKILL)
                elif runner_pid is not None:
                    _terminate_pid_tree(runner_pid)

                runner_proc = _start_runner(report_path, args.series_name, args.gpus, run_log)
                runner_pid = runner_proc.pid
                previous_signature = _report_signature(_load_report(report_path))
                last_progress_ts = datetime.now(timezone.utc).timestamp()

            time.sleep(args.poll_seconds)
    finally:
        try:
            os.killpg(monitor_proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
