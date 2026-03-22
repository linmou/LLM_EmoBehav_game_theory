#!/usr/bin/env python3
# Build one-model sanity configs and optionally launch parallel reader prebuild runs for VLMs.

from __future__ import annotations

import argparse
import copy
import dataclasses
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


@dataclasses.dataclass(frozen=True)
class PrebuildJob:
    model_path: str


@dataclasses.dataclass(frozen=True)
class RunningJob:
    process: subprocess.Popen[str]
    model_path: str
    gpu_id: int
    config_path: Path
    log_path: Path
    started_at: float
    last_progress_at: float


def build_single_model_sanity_config(
    base_config: dict[str, Any], model_path: str
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["models"] = [model_path]
    config["sanity_check"] = True
    config["sanity_check_limit"] = 2
    config["defer_evaluation"] = True
    config["output_dir"] = f"{base_config['output_dir']}_reader_prebuild"
    return config


def build_single_model_jobs(models: list[str], gpu_ids: list[int]) -> list[PrebuildJob]:
    if not gpu_ids:
        raise ValueError("gpu_ids must not be empty")
    del gpu_ids
    return [PrebuildJob(model_path=model_path) for model_path in models]


def _sanitize_model_name(model_path: str) -> str:
    return Path(model_path.rstrip("/")).name.replace(".", "_")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _is_completed_log(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    return "Memory experiment series completed." in log_path.read_text(encoding="utf-8")


def _launch_job(job: PrebuildJob, config_path: Path, log_path: Path) -> subprocess.Popen[str]:
    env = os.environ.copy()
    gpu_id = int(config_path.stem.rsplit("_gpu", 1)[1])
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    command = [
        "python",
        "-m",
        "emotion_experiment_engine.emotion_experiment_series_runner",
        "--config",
        str(config_path),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(
        command,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def _gpu_memory_used_mib(gpu_id: int) -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return 0
    line = result.stdout.strip().splitlines()
    if not line:
        return 0
    try:
        return int(line[0].strip())
    except ValueError:
        return 0


def _terminate_running_job(job: RunningJob) -> None:
    try:
        os.killpg(job.process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def _is_startup_stalled(
    job: RunningJob,
    startup_timeout_seconds: float,
    startup_memory_threshold_mib: int,
    now: float,
) -> bool:
    if now - job.started_at < startup_timeout_seconds:
        return False
    return _gpu_memory_used_mib(job.gpu_id) <= startup_memory_threshold_mib


def _log_mtime_seconds(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_mtime


def _has_progress_stalled(job: RunningJob, progress_timeout_seconds: float, now: float) -> bool:
    if progress_timeout_seconds <= 0:
        return False
    log_mtime = _log_mtime_seconds(job.log_path)
    latest_progress = max(job.last_progress_at, log_mtime) if log_mtime > 0 else job.last_progress_at
    return now - latest_progress >= progress_timeout_seconds


def run_parallel_prebuilds(
    config_path: Path,
    gpu_ids: list[int],
    work_dir: Path,
    launch: bool,
    startup_timeout_seconds: float = 120.0,
    startup_memory_threshold_mib: int = 128,
    progress_timeout_seconds: float = 180.0,
) -> list[Path]:
    base_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    jobs = build_single_model_jobs(base_config["models"], gpu_ids)

    generated_configs: list[Path] = []
    generated_dir = work_dir / "tmp" / "reader_prebuild_configs"
    logs_dir = work_dir / "logs" / "reader_prebuild"
    pending_models: list[str] = []
    completed_models: set[str] = set()

    for model_path in base_config["models"]:
        model_name = _sanitize_model_name(model_path)
        is_completed = False
        for gpu_id in gpu_ids:
            if _is_completed_log(logs_dir / f"{model_name}_gpu{gpu_id}.log"):
                is_completed = True
                break
        if is_completed:
            completed_models.add(model_path)
            continue
        pending_models.append(model_path)

    if not launch:
        for index, model_path in enumerate(pending_models):
            gpu_id = gpu_ids[index % len(gpu_ids)]
            model_config = build_single_model_sanity_config(base_config, model_path)
            generated_config = generated_dir / f"{_sanitize_model_name(model_path)}_gpu{gpu_id}.yaml"
            generated_configs.append(generated_config)
            _write_yaml(generated_config, model_config)
        return generated_configs

    running_by_gpu: dict[int, RunningJob] = {}
    while pending_models or running_by_gpu:
        progress_made = False

        for gpu_id in list(gpu_ids):
            running_job = running_by_gpu.get(gpu_id)
            if running_job is not None and running_job.process.poll() is not None:
                del running_by_gpu[gpu_id]
                progress_made = True
                continue

            if running_job is not None and _is_startup_stalled(
                running_job,
                startup_timeout_seconds=startup_timeout_seconds,
                startup_memory_threshold_mib=startup_memory_threshold_mib,
                now=time.time(),
            ):
                _terminate_running_job(running_job)
                pending_models.append(running_job.model_path)
                del running_by_gpu[gpu_id]
                progress_made = True
                continue

            if running_job is not None and _has_progress_stalled(
                running_job,
                progress_timeout_seconds=progress_timeout_seconds,
                now=time.time(),
            ):
                _terminate_running_job(running_job)
                pending_models.append(running_job.model_path)
                del running_by_gpu[gpu_id]
                progress_made = True
                continue

            if gpu_id in running_by_gpu:
                continue
            if not pending_models:
                continue

            model_path = pending_models.pop(0)
            job = PrebuildJob(model_path=model_path)
            model_config = build_single_model_sanity_config(base_config, model_path)
            generated_config = generated_dir / f"{_sanitize_model_name(model_path)}_gpu{gpu_id}.yaml"
            log_path = logs_dir / f"{generated_config.stem}.log"
            generated_configs.append(generated_config)
            _write_yaml(generated_config, model_config)
            running_by_gpu[gpu_id] = RunningJob(
                process=_launch_job(job, generated_config, log_path),
                model_path=model_path,
                gpu_id=gpu_id,
                config_path=generated_config,
                log_path=log_path,
                started_at=time.time(),
                last_progress_at=time.time(),
            )
            progress_made = True

        if not progress_made and running_by_gpu:
            time.sleep(1)

    return generated_configs


def _parse_gpu_ids(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prebuild emotion-reader caches by launching one-model sanity runs per GPU."
    )
    parser.add_argument("--config", required=True, help="Base experiment config path")
    parser.add_argument(
        "--gpus",
        default="0,1,2,3",
        help="Comma-separated GPU ids to use, one model per GPU at a time",
    )
    parser.add_argument(
        "--write-only",
        action="store_true",
        help="Only write derived one-model configs, do not launch runs",
    )
    parser.add_argument(
        "--startup-timeout-seconds",
        type=float,
        default=120.0,
        help="Kill and requeue jobs that stay alive too long without meaningful GPU allocation",
    )
    parser.add_argument(
        "--progress-timeout-seconds",
        type=float,
        default=180.0,
        help="Kill and requeue jobs whose log stops advancing for too long",
    )
    args = parser.parse_args()

    generated = run_parallel_prebuilds(
        config_path=Path(args.config),
        gpu_ids=_parse_gpu_ids(args.gpus),
        work_dir=Path.cwd(),
        launch=not args.write_only,
        startup_timeout_seconds=args.startup_timeout_seconds,
        progress_timeout_seconds=args.progress_timeout_seconds,
    )
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
