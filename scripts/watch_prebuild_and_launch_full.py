#!/usr/bin/env python3
# Wait for reader-prebuild runs to finish, then launch the full experiment once.

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path
from typing import Sequence


def build_full_run_command(config_path: Path, gpu_ids: str) -> list[str]:
    return [
        "python",
        "-m",
        "emotion_experiment_engine.emotion_experiment_series_runner",
        "--config",
        str(config_path),
    ]


def should_wait_for_prebuild(process_lines: Sequence[str], config_path: Path) -> bool:
    config_text = str(config_path)
    for line in process_lines:
        if "scripts/prebuild_vlm_readers.py" not in line:
            continue
        if config_text not in line:
            continue
        return True
    return False


def _list_process_lines() -> list[str]:
    result = subprocess.run(
        ["pgrep", "-af", "scripts/prebuild_vlm_readers.py"],
        check=False,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def wait_for_prebuild(config_path: Path, poll_seconds: float) -> None:
    while should_wait_for_prebuild(_list_process_lines(), config_path):
        time.sleep(poll_seconds)


def launch_full_run(config_path: Path, gpu_ids: str) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    result = subprocess.run(
        build_full_run_command(config_path=config_path, gpu_ids=gpu_ids),
        env=env,
        check=False,
    )
    return int(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wait for prebuild_vlm_readers.py to finish, then launch the full experiment."
    )
    parser.add_argument("--config", required=True, help="Full experiment config path")
    parser.add_argument(
        "--gpus",
        default="0,1,2,3",
        help="CUDA_VISIBLE_DEVICES value for the full experiment launch",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=10.0,
        help="Polling interval while waiting for prebuild runs to finish",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    wait_for_prebuild(config_path=config_path, poll_seconds=args.poll_seconds)
    return launch_full_run(config_path=config_path, gpu_ids=args.gpus)


if __name__ == "__main__":
    raise SystemExit(main())
