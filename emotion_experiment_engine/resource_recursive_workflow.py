"""Workflow helpers for report splitting, merging, and recursive GPU escalation."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from .emotion_experiment_series_runner import MemoryExperimentSeriesRunner
from .evaluate_saved_series import _is_terminal, _load_report, process_report


LOGGER = logging.getLogger(__name__)


def _load_payloads(report_paths: Sequence[Path | str]) -> list[dict]:
    return [_load_report(Path(path).expanduser().resolve()) for path in report_paths]


def _normalize_paths(report_paths: Sequence[Path | str]) -> list[Path]:
    return [Path(path).expanduser().resolve() for path in report_paths]


def _reset_experiment_for_resume(experiment: dict) -> dict:
    normalized = copy.deepcopy(experiment)
    if normalized.get("status") != "completed":
        normalized["status"] = "pending"
        normalized["output_dir"] = None
        normalized["start_time"] = None
        normalized["end_time"] = None
        normalized["time_cost_seconds"] = None
        normalized["error"] = None
    return normalized


def _resolve_runner_series_config(payload: dict[str, Any]) -> dict[str, Any]:
    series_config = copy.deepcopy(payload.get("series_config", {}))
    if not isinstance(series_config, dict):
        return {}
    if "benchmarks" in series_config:
        return series_config

    source_report = series_config.get("source_report")
    if source_report:
        source_payload = _load_report(Path(str(source_report)).expanduser().resolve())
        return _resolve_runner_series_config(source_payload)

    merged_configs = series_config.get("merged_from_series_configs", [])
    if isinstance(merged_configs, list):
        for candidate in merged_configs:
            if isinstance(candidate, dict):
                resolved = _resolve_runner_series_config({"series_config": candidate})
                if "benchmarks" in resolved:
                    return resolved

    source_reports = series_config.get("source_reports", [])
    if isinstance(source_reports, list):
        for candidate in source_reports:
            candidate_path = Path(str(candidate)).expanduser().resolve()
            if candidate_path.exists():
                resolved = _resolve_runner_series_config(_load_report(candidate_path))
                if "benchmarks" in resolved:
                    return resolved

    return series_config


def _copy_report_shell(
    payload: dict[str, Any],
    *,
    series_name: str,
    source_report: Path,
    experiments: dict[str, dict[str, Any]],
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = datetime.now().isoformat()
    series_config = _resolve_runner_series_config(payload)
    series_config["source_report"] = str(source_report)
    if extra_config:
        series_config.update(extra_config)
    return {
        "last_updated": now,
        "series_start_time": payload.get("series_start_time", now),
        "series_duration_seconds": payload.get("series_duration_seconds", 0.0),
        "series_name": series_name,
        "series_config": series_config,
        "sessions": [],
        "experiments": experiments,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def partition_gpu_pool(
    gpu_pool: Sequence[str | int],
    *,
    resource_gpus: int,
) -> tuple[list[list[str]], list[str]]:
    if resource_gpus <= 0:
        raise ValueError("resource_gpus must be positive")

    normalized_gpu_pool = [str(gpu_id) for gpu_id in gpu_pool]
    usable_count = len(normalized_gpu_pool) - (len(normalized_gpu_pool) % resource_gpus)
    gpu_groups = [
        normalized_gpu_pool[idx : idx + resource_gpus]
        for idx in range(0, usable_count, resource_gpus)
    ]
    ignored_gpu_ids = normalized_gpu_pool[usable_count:]
    return gpu_groups, ignored_gpu_ids


def _resource_round_experiment(experiment: dict[str, Any]) -> dict[str, Any]:
    normalized = _reset_experiment_for_resume(experiment)
    normalized.setdefault("required_gpu_count", 1)
    normalized.setdefault("last_attempt_gpu_count", experiment.get("last_attempt_gpu_count"))
    normalized.setdefault("resource_failure_blocked", False)
    return normalized


def build_resource_round_reports(
    report_path: Path | str,
    *,
    round_output_dir: Path | str,
    shard_series_prefix: str,
    resource_gpus: int,
    gpu_pool: Sequence[str | int],
    carry_forward_series_name: str,
) -> dict[str, Any]:
    source_report = Path(report_path).expanduser().resolve()
    destination = Path(round_output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    payload = _load_report(source_report)
    experiments = payload.get("experiments", {})
    if not isinstance(experiments, dict):
        raise ValueError("Report experiments payload must be a mapping")

    gpu_groups, ignored_gpu_ids = partition_gpu_pool(gpu_pool, resource_gpus=resource_gpus)
    if not gpu_groups:
        raise ValueError("gpu_pool does not contain enough GPUs for one resource group")

    carry_forward_experiments: dict[str, dict[str, Any]] = {}
    model_to_experiments: dict[str, dict[str, dict[str, Any]]] = {}

    for exp_id, experiment in experiments.items():
        if not isinstance(experiment, dict):
            raise ValueError(f"Experiment payload must be a mapping: {exp_id}")

        status = str(experiment.get("status", "")).strip().lower()
        if status == "completed":
            carry_forward_experiments[exp_id] = copy.deepcopy(experiment)
            continue

        required_gpu_count = int(experiment.get("required_gpu_count", 1) or 1)
        if required_gpu_count != resource_gpus:
            continue

        model_name = str(experiment.get("model_name", ""))
        model_to_experiments.setdefault(model_name, {})[exp_id] = _resource_round_experiment(experiment)

    carry_forward_report = _write_json(
        destination / f"{carry_forward_series_name}_resume_report.json",
        _copy_report_shell(
            payload,
            series_name=carry_forward_series_name,
            source_report=source_report,
            experiments=carry_forward_experiments,
            extra_config={
                "workflow_role": "resource_carry_forward",
                "current_round_gpu_count": resource_gpus,
            },
        ),
    )

    shard_reports: list[Path] = []
    shard_assignments: dict[str, list[str]] = {}
    scheduled_models = sorted(model_to_experiments.keys())
    shard_experiments: list[dict[str, dict[str, Any]]] = [dict() for _ in gpu_groups]

    for idx, model_name in enumerate(scheduled_models):
        shard_idx = idx % len(gpu_groups)
        shard_experiments[shard_idx].update(model_to_experiments[model_name])

    for shard_idx, gpu_group in enumerate(gpu_groups):
        shard_name = f"{shard_series_prefix}{shard_idx:02d}"
        shard_assignments[shard_name] = sorted(shard_experiments[shard_idx].keys())
        shard_report = _write_json(
            destination / f"{shard_name}_resume_report.json",
            _copy_report_shell(
                payload,
                series_name=shard_name,
                source_report=source_report,
                experiments=shard_experiments[shard_idx],
                extra_config={
                    "workflow_role": "resource_round_shard",
                    "current_round_gpu_count": resource_gpus,
                    "assigned_gpu_ids": list(gpu_group),
                },
            ),
        )
        shard_reports.append(shard_report)

    manifest_path = _write_json(
        destination / "resource_round_manifest.json",
        {
            "source_report": str(source_report),
            "resource_gpus": resource_gpus,
            "gpu_groups": gpu_groups,
            "ignored_gpu_ids": ignored_gpu_ids,
            "carry_forward_report": str(carry_forward_report),
            "shard_reports": [str(path) for path in shard_reports],
            "scheduled_models": scheduled_models,
            "shard_assignments": shard_assignments,
        },
    )
    return {
        "carry_forward_report": carry_forward_report,
        "shard_reports": shard_reports,
        "manifest_path": manifest_path,
        "gpu_groups": gpu_groups,
        "ignored_gpu_ids": ignored_gpu_ids,
    }


def advance_resource_round_state(
    report_path: Path | str,
    *,
    output_dir: Path | str,
    merged_series_name: str,
    current_round_gpu_count: int,
    max_resource_gpus: int,
) -> Path:
    source_report = Path(report_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    payload = _load_report(source_report)
    experiments = payload.get("experiments", {})
    if not isinstance(experiments, dict):
        raise ValueError("Report experiments payload must be a mapping")

    next_round_gpu_count = min(current_round_gpu_count * 2, max_resource_gpus)
    failed_models = {
        str(exp.get("model_name", ""))
        for exp in experiments.values()
        if isinstance(exp, dict)
        and str(exp.get("status", "")).strip().lower() == "failed"
        and int(exp.get("required_gpu_count", 1) or 1) == current_round_gpu_count
    }

    merged_experiments: dict[str, dict[str, Any]] = {}
    for exp_id, experiment in experiments.items():
        if not isinstance(experiment, dict):
            raise ValueError(f"Experiment payload must be a mapping: {exp_id}")

        normalized = copy.deepcopy(experiment)
        model_name = str(normalized.get("model_name", ""))
        status = str(normalized.get("status", "")).strip().lower()
        required_gpu_count = int(normalized.get("required_gpu_count", 1) or 1)

        if model_name in failed_models and status != "completed":
            if current_round_gpu_count < max_resource_gpus:
                normalized = _reset_experiment_for_resume(normalized)
                normalized["required_gpu_count"] = next_round_gpu_count
                normalized["last_attempt_gpu_count"] = current_round_gpu_count
                normalized["resource_failure_blocked"] = False
            else:
                normalized["last_attempt_gpu_count"] = current_round_gpu_count
                if status != "failed":
                    normalized["required_gpu_count"] = current_round_gpu_count * 2
                    normalized["resource_failure_blocked"] = True
        merged_experiments[exp_id] = normalized

    series_config = copy.deepcopy(payload.get("series_config", {}))
    if current_round_gpu_count < max_resource_gpus and failed_models:
        series_config["current_round_gpu_count"] = next_round_gpu_count
        series_config["resource_round_index"] = int(series_config.get("resource_round_index", 1)) + 1
    else:
        series_config["current_round_gpu_count"] = current_round_gpu_count
        series_config["resource_round_index"] = int(series_config.get("resource_round_index", 1))
    series_config["source_report"] = str(source_report)

    next_payload = {
        "last_updated": datetime.now().isoformat(),
        "series_start_time": payload.get("series_start_time", datetime.now().isoformat()),
        "series_duration_seconds": payload.get("series_duration_seconds", 0.0),
        "series_name": merged_series_name,
        "series_config": series_config,
        "sessions": copy.deepcopy(payload.get("sessions", [])),
        "experiments": merged_experiments,
    }
    return _write_json(destination / f"{merged_series_name}_resume_report.json", next_payload)


def merge_round_reports_for_state(
    source_report_path: Path | str,
    *,
    carry_forward_report: Path | str,
    shard_reports: Sequence[Path | str],
    merged_output_dir: Path | str,
    merged_series_name: str,
) -> Path:
    source_report = Path(source_report_path).expanduser().resolve()
    source_payload = _load_report(source_report)
    destination = Path(merged_output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    merged_payload = copy.deepcopy(source_payload)
    merged_payload["series_name"] = merged_series_name
    merged_payload["last_updated"] = datetime.now().isoformat()
    merged_payload.setdefault("series_config", {})
    merged_payload["series_config"]["source_report"] = str(source_report)

    merged_sessions: list[dict[str, Any]] = []
    merged_experiments = copy.deepcopy(source_payload.get("experiments", {}))

    for report_path in [carry_forward_report, *shard_reports]:
        payload = _load_report(Path(report_path).expanduser().resolve())
        merged_sessions.extend(copy.deepcopy(payload.get("sessions", [])))
        experiments = payload.get("experiments", {})
        if not isinstance(experiments, dict):
            raise ValueError("Report experiments payload must be a mapping")
        for exp_id, experiment in experiments.items():
            if not isinstance(experiment, dict):
                raise ValueError(f"Experiment payload must be a mapping: {exp_id}")
            merged_experiments[exp_id] = copy.deepcopy(experiment)

    merged_payload["sessions"] = merged_sessions
    merged_payload["experiments"] = merged_experiments
    return _write_json(destination / f"{merged_series_name}_resume_report.json", merged_payload)


def split_resume_report(
    report_path: Path | str,
    *,
    split_output_dir: Path | str,
    shard_series_prefix: str,
    shard_labels: Sequence[str],
) -> list[Path]:
    source_report = Path(report_path).expanduser().resolve()
    destination = Path(split_output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    if not shard_labels:
        raise ValueError("At least one shard label is required")

    payload = _load_report(source_report)
    experiments = payload.get("experiments", {})
    if not isinstance(experiments, dict):
        raise ValueError("Report experiments payload must be a mapping")

    shard_experiments: list[dict[str, dict]] = [dict() for _ in shard_labels]
    shard_paths: list[Path] = []
    next_incomplete_shard = 0

    for exp_id, experiment in experiments.items():
        if not isinstance(experiment, dict):
            raise ValueError(f"Experiment payload must be a mapping: {exp_id}")
        if experiment.get("status") == "completed":
            shard_idx = 0
        else:
            shard_idx = next_incomplete_shard % len(shard_labels)
            next_incomplete_shard += 1
        shard_experiments[shard_idx][exp_id] = _reset_experiment_for_resume(experiment)

    now = datetime.now()
    for shard_idx, shard_label in enumerate(shard_labels):
        shard_name = f"{shard_series_prefix}{shard_label}"
        shard_payload = {
            "last_updated": now.isoformat(),
            "series_start_time": payload.get("series_start_time", now.isoformat()),
            "series_duration_seconds": payload.get("series_duration_seconds", 0.0),
            "series_name": shard_name,
            "series_config": {
                **copy.deepcopy(payload.get("series_config", {})),
                "source_report": str(source_report),
                "split_shard_label": str(shard_label),
            },
            "sessions": [],
            "experiments": shard_experiments[shard_idx],
        }
        shard_report = destination / f"{shard_name}_resume_report.json"
        shard_report.write_text(json.dumps(shard_payload, indent=2), encoding="utf-8")
        shard_paths.append(shard_report)

    return shard_paths


def split_filtered_resume_report(
    report_path: Path | str,
    *,
    split_output_dir: Path | str,
    shard_series_prefix: str,
    shard_labels: Sequence[str],
    deferred_models: Sequence[str],
    carry_forward_series_name: str,
) -> dict[str, Any]:
    source_report = Path(report_path).expanduser().resolve()
    destination = Path(split_output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    if not shard_labels:
        raise ValueError("At least one shard label is required")

    payload = _load_report(source_report)
    experiments = payload.get("experiments", {})
    if not isinstance(experiments, dict):
        raise ValueError("Report experiments payload must be a mapping")

    deferred_model_set = {str(model) for model in deferred_models}
    carry_forward_experiments: dict[str, dict[str, Any]] = {}
    shard_experiments: list[dict[str, dict[str, Any]]] = [dict() for _ in shard_labels]
    deferred_exp_ids: list[str] = []
    next_incomplete_shard = 0

    for exp_id, experiment in experiments.items():
        if not isinstance(experiment, dict):
            raise ValueError(f"Experiment payload must be a mapping: {exp_id}")

        model_name = str(experiment.get("model_name", ""))
        status = str(experiment.get("status", "")).strip().lower()

        if status == "completed":
            carry_forward_experiments[exp_id] = copy.deepcopy(experiment)
            continue

        if model_name in deferred_model_set:
            deferred_exp_ids.append(exp_id)
            continue

        shard_idx = next_incomplete_shard % len(shard_labels)
        next_incomplete_shard += 1
        shard_experiments[shard_idx][exp_id] = _reset_experiment_for_resume(experiment)

    carry_forward_report = _write_json(
        destination / f"{carry_forward_series_name}_resume_report.json",
        _copy_report_shell(
            payload,
            series_name=carry_forward_series_name,
            source_report=source_report,
            experiments=carry_forward_experiments,
            extra_config={"workflow_role": "carry_forward"},
        ),
    )

    shard_reports: list[Path] = []
    shard_assignments: dict[str, list[str]] = {}
    for shard_idx, shard_label in enumerate(shard_labels):
        shard_name = f"{shard_series_prefix}{shard_label}"
        shard_assignments[shard_name] = sorted(shard_experiments[shard_idx].keys())
        shard_report = _write_json(
            destination / f"{shard_name}_resume_report.json",
            _copy_report_shell(
                payload,
                series_name=shard_name,
                source_report=source_report,
                experiments=shard_experiments[shard_idx],
                extra_config={
                    "workflow_role": "single_gpu_shard",
                    "split_shard_label": str(shard_label),
                    "deferred_models": list(deferred_models),
                },
            ),
        )
        shard_reports.append(shard_report)

    manifest_payload = {
        "source_report": str(source_report),
        "carry_forward_report": str(carry_forward_report),
        "shard_reports": [str(path) for path in shard_reports],
        "deferred_models": list(deferred_models),
        "carry_forward_experiment_ids": sorted(carry_forward_experiments.keys()),
        "deferred_experiment_ids": sorted(deferred_exp_ids),
        "shard_assignments": shard_assignments,
    }
    manifest_path = _write_json(destination / "split_plan_manifest.json", manifest_payload)
    return {
        "carry_forward_report": carry_forward_report,
        "shard_reports": shard_reports,
        "manifest_path": manifest_path,
    }


def build_recovery_resume_report(
    report_path: Path | str,
    *,
    output_dir: Path | str,
    series_name: str,
    deferred_models: Sequence[str],
) -> Path:
    source_report = Path(report_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    payload = _load_report(source_report)
    experiments = payload.get("experiments", {})
    if not isinstance(experiments, dict):
        raise ValueError("Report experiments payload must be a mapping")

    deferred_model_set = {str(model) for model in deferred_models}
    recovery_experiments: dict[str, dict[str, Any]] = {}

    for exp_id, experiment in experiments.items():
        if not isinstance(experiment, dict):
            raise ValueError(f"Experiment payload must be a mapping: {exp_id}")
        model_name = str(experiment.get("model_name", ""))
        status = str(experiment.get("status", "")).strip().lower()
        if model_name not in deferred_model_set or status == "completed":
            continue
        recovery_experiments[exp_id] = _reset_experiment_for_resume(experiment)

    recovery_payload = _copy_report_shell(
        payload,
        series_name=series_name,
        source_report=source_report,
        experiments=recovery_experiments,
        extra_config={
            "workflow_role": "deferred_model_recovery",
            "deferred_models": list(deferred_models),
        },
    )
    return _write_json(destination / f"{series_name}_resume_report.json", recovery_payload)


def merge_reports_for_resume(
    report_paths: Sequence[Path | str],
    *,
    resume_source_report: Path | str,
    merged_output_dir: Path | str,
    merged_series_name: str,
    extra_config: dict[str, Any] | None = None,
) -> Path:
    reports = _normalize_paths(report_paths)
    payloads = _load_payloads(reports)
    resume_source = Path(resume_source_report).expanduser().resolve()
    source_payload = _load_report(resume_source)
    destination = Path(merged_output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    merged_experiments: dict[str, dict] = {}
    merged_sessions: list[dict] = []

    for payload in payloads:
        merged_sessions.extend(copy.deepcopy(payload.get("sessions", [])))
        experiments = payload.get("experiments", {})
        if not isinstance(experiments, dict):
            continue
        for exp_id, exp in experiments.items():
            if exp_id in merged_experiments:
                raise ValueError(f"Duplicate experiment id across split reports: {exp_id}")
            if not isinstance(exp, dict):
                raise ValueError(f"Experiment payload must be a mapping: {exp_id}")
            merged_experiments[exp_id] = copy.deepcopy(exp)

    merged_payload = _copy_report_shell(
        source_payload,
        series_name=merged_series_name,
        source_report=resume_source,
        experiments=merged_experiments,
        extra_config=extra_config,
    )
    merged_payload["sessions"] = merged_sessions
    return _write_json(
        destination / f"{merged_series_name}_resume_report.json",
        merged_payload,
    )


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


def _python_executable(conda_env: str) -> str:
    return str(Path("/home/jjl7137/anaconda3/envs") / conda_env / "bin" / "python")


def _parse_gpu_pool(raw_gpu_pool: str | Sequence[str]) -> list[str]:
    if isinstance(raw_gpu_pool, str):
        return [gpu.strip() for gpu in raw_gpu_pool.split(",") if gpu.strip()]
    return [str(gpu).strip() for gpu in raw_gpu_pool if str(gpu).strip()]


def _has_schedulable_work(
    report_path: Path,
    *,
    current_round_gpu_count: int,
    max_resource_gpus: int,
) -> bool:
    payload = _load_report(report_path)
    experiments = payload.get("experiments", {})
    if not isinstance(experiments, dict):
        return False

    for experiment in experiments.values():
        if not isinstance(experiment, dict):
            continue
        status = str(experiment.get("status", "")).strip().lower()
        if status == "completed":
            continue
        required_gpu_count = int(experiment.get("required_gpu_count", 1) or 1)
        if required_gpu_count == current_round_gpu_count and required_gpu_count <= max_resource_gpus:
            return True
    return False


def _copy_source_report(source_report: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_report, destination)
    return destination


def _bootstrap_source_report_from_config(
    *,
    config_path: Path,
    destination: Path,
    series_name: str,
    gpu_pool: Sequence[str],
    min_resource_gpus: int,
    max_resource_gpus: int,
) -> Path:
    runner = MemoryExperimentSeriesRunner(
        config_path=str(config_path),
        series_name=series_name,
        dry_run=False,
    )
    original_benchmarks = runner.base_config["benchmarks"]
    models = runner.base_config["models"]
    benchmarks = runner.expand_benchmark_configs(original_benchmarks)

    augmented_config = dict(runner.base_config)
    augmented_config["resource_pipeline"] = True
    augmented_config["gpu_pool"] = list(gpu_pool)
    augmented_config["min_resource_gpus"] = min_resource_gpus
    augmented_config["max_resource_gpus"] = max_resource_gpus
    augmented_config["current_round_gpu_count"] = min_resource_gpus
    augmented_config["resource_round_index"] = 1
    runner.report.attach_series_config(augmented_config, series_name)

    for benchmark_config in benchmarks:
        benchmark_name = benchmark_config["name"]
        task_type = benchmark_config["task_type"]
        for model_name in models:
            model_folder_name = runner._format_model_name_for_folder(model_name)
            exp_id = f"{benchmark_name}_{task_type}_{model_folder_name.replace('/', '_')}"
            runner.report.add_experiment(
                f"{benchmark_name}_{task_type}",
                model_name,
                exp_id,
                resolved_model_path=None,
            )
            runner.report.update_experiment(
                exp_id,
                required_gpu_count=min_resource_gpus,
                last_attempt_gpu_count=None,
                resource_failure_blocked=False,
            )

    try:
        runner.report.end_session("completed")
    except Exception:
        pass
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(runner.report.report_file), destination)
    return destination


def _start_watchdog(
    *,
    conda_env: str,
    report_path: Path,
    series_name: str,
    gpu_ids: Sequence[str],
    run_log: Path,
    monitor_log: Path,
    poll_seconds: float,
    stall_seconds: float,
    idle_util_threshold: float,
) -> subprocess.Popen[str]:
    cmd = [
        _python_executable(conda_env),
        str(REPO_ROOT / "scripts" / "series_runner_watchdog.py"),
        "--report",
        str(report_path),
        "--series-name",
        series_name,
        "--gpus",
        ",".join(gpu_ids),
        "--run-log",
        str(run_log),
        "--monitor-log",
        str(monitor_log),
        "--poll-seconds",
        str(poll_seconds),
        "--stall-seconds",
        str(stall_seconds),
        "--idle-util-threshold",
        str(idle_util_threshold),
    ]
    return subprocess.Popen(cmd, cwd=REPO_ROOT, start_new_session=True, text=True)


def _stop_process_group(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)


def _run_round_watchdogs(
    *,
    round_dir: Path,
    shard_reports: list[Path],
    conda_env: str,
    poll_seconds: float,
    stall_seconds: float,
    idle_util_threshold: float,
    logs_dir: Path,
) -> None:
    watchdogs: list[subprocess.Popen[str]] = []
    try:
        for idx, shard_report in enumerate(shard_reports):
            payload = _load_report(shard_report)
            series_name = str(payload.get("series_name", shard_report.stem))
            assigned_gpu_ids = payload.get("series_config", {}).get("assigned_gpu_ids", [])
            if not assigned_gpu_ids:
                raise ValueError(f"Shard report is missing assigned_gpu_ids: {shard_report}")
            watchdogs.append(
                _start_watchdog(
                    conda_env=conda_env,
                    report_path=shard_report,
                    series_name=series_name,
                    gpu_ids=[str(gpu_id) for gpu_id in assigned_gpu_ids],
                    run_log=logs_dir / f"{round_dir.name}_group_{idx:02d}.watchdog.log",
                    monitor_log=logs_dir / f"{round_dir.name}_group_{idx:02d}.gpu.log",
                    poll_seconds=poll_seconds,
                    stall_seconds=stall_seconds,
                    idle_util_threshold=idle_util_threshold,
                )
            )

        for watchdog in watchdogs:
            return_code = watchdog.wait()
            if return_code != 0:
                raise RuntimeError(f"Watchdog exited with code {return_code}")
    finally:
        for watchdog in watchdogs:
            _stop_process_group(watchdog)


def _materialize_final_report(report_path: Path, final_dir: Path) -> Path:
    payload = _load_report(report_path)

    experiments = payload.get("experiments", {})
    unresolved_models: set[str] = set()
    if not isinstance(experiments, dict):
        raise ValueError("Report experiments payload must be a mapping")

    for exp in experiments.values():
        if not isinstance(exp, dict):
            continue
        status = str(exp.get("status", "")).strip().lower()
        if status != "completed":
            unresolved_models.add(str(exp.get("model_name", "")))
            continue

    final_report = _write_json(final_dir / "final_report.json", payload)
    _write_json(
        final_dir / "final_manifest.json",
        {
            "final_report": str(final_report),
            "completed": sum(
                1
                for exp in experiments.values()
                if isinstance(exp, dict) and str(exp.get("status", "")).strip().lower() == "completed"
            ),
            "failed": sum(
                1
                for exp in experiments.values()
                if isinstance(exp, dict) and str(exp.get("status", "")).strip().lower() == "failed"
            ),
            "pending": sum(
                1
                for exp in experiments.values()
                if isinstance(exp, dict) and str(exp.get("status", "")).strip().lower() == "pending"
            ),
            "unresolved_models": sorted(model for model in unresolved_models if model),
        },
    )
    _write_json(
        final_dir / "unresolved_models.json",
        {"models": sorted(model for model in unresolved_models if model)},
    )
    return final_report


def _resolve_pipeline_root(
    *,
    report_path: Path | str | None,
    config_path: Path | str | None,
) -> tuple[Path, Path | None]:
    if report_path:
        source_report = Path(report_path).expanduser().resolve()
        payload = _load_report(source_report)
        series_config = payload.get("series_config", {})
        if not isinstance(series_config, dict):
            raise ValueError("Report series_config must be a mapping")
        output_dir = series_config.get("output_dir")
        if not output_dir:
            raise ValueError("Report series_config.output_dir is required for recursive workflow")
        return Path(str(output_dir)).expanduser().resolve() / "resource_pipeline", None

    if not config_path:
        raise ValueError("Provide exactly one of report_path or config_path")

    cfg_path = Path(str(config_path)).expanduser().resolve()
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config payload must be a mapping")
    output_dir = cfg.get("output_dir")
    if not output_dir:
        raise ValueError("Config output_dir is required for recursive workflow")
    return Path(str(output_dir)).expanduser().resolve() / "resource_pipeline", cfg_path


def _derive_pipeline_series_base(
    *,
    report_path: Path | str | None,
    config_path: Path | str | None,
) -> str:
    if report_path:
        source_report = Path(report_path).expanduser().resolve()
        payload = _load_report(source_report)
        series_name = str(payload.get("series_name", "")).strip()
        if series_name:
            return series_name
        return source_report.stem

    if not config_path:
        raise ValueError("Provide exactly one of report_path or config_path")

    cfg_path = Path(str(config_path)).expanduser().resolve()
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config payload must be a mapping")
    experiment_name = str(cfg.get("experiment_name", "")).strip()
    if experiment_name:
        return experiment_name
    return cfg_path.stem


def orchestrate_resource_pipeline(
    *,
    report_path: Path | str | None = None,
    config_path: Path | str | None = None,
    gpu_pool: Sequence[str] | str,
    min_resource_gpus: int,
    max_resource_gpus: int,
    conda_env: str,
    poll_seconds: float,
    stall_seconds: float,
    idle_util_threshold: float,
    max_workers: int,
) -> Path:
    gpu_pool_list = _parse_gpu_pool(gpu_pool)
    if not gpu_pool_list:
        raise ValueError("gpu_pool must not be empty")
    if bool(report_path) == bool(config_path):
        raise ValueError("Provide exactly one of report_path or config_path")

    pipeline_root, resolved_config_path = _resolve_pipeline_root(
        report_path=report_path,
        config_path=config_path,
    )
    pipeline_series_base = _derive_pipeline_series_base(
        report_path=report_path,
        config_path=config_path,
    )
    source_dir = pipeline_root / "source"
    meta_dir = pipeline_root / "meta"
    logs_dir = meta_dir / "logs"
    rounds_dir = pipeline_root / "rounds"
    final_dir = pipeline_root / "final"
    for directory in [source_dir, meta_dir, logs_dir, rounds_dir, final_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    source_report_path = source_dir / "source_report.json"
    if report_path:
        source_report = Path(report_path).expanduser().resolve()
        planning_report = _copy_source_report(source_report, source_report_path)
    else:
        assert resolved_config_path is not None
        with open(resolved_config_path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        derived_series_name = str(cfg.get("experiment_name") or resolved_config_path.stem)
        source_report = resolved_config_path
        planning_report = _bootstrap_source_report_from_config(
            config_path=resolved_config_path,
            destination=source_report_path,
            series_name=derived_series_name,
            gpu_pool=gpu_pool_list,
            min_resource_gpus=min_resource_gpus,
            max_resource_gpus=max_resource_gpus,
        )

    _write_json(
        meta_dir / "pipeline_config.json",
        {
            "source_report": str(source_report),
            "config_path": str(config_path) if config_path else None,
            "pipeline_root": str(pipeline_root),
            "pipeline_series_base": pipeline_series_base,
            "gpu_pool": gpu_pool_list,
            "min_resource_gpus": min_resource_gpus,
            "max_resource_gpus": max_resource_gpus,
            "conda_env": conda_env,
            "poll_seconds": poll_seconds,
            "stall_seconds": stall_seconds,
            "idle_util_threshold": idle_util_threshold,
            "max_workers": max_workers,
        },
    )

    top_manifest: dict[str, Any] = {
        "source_report": str(source_report),
        "planning_report": str(planning_report),
        "gpu_pool": gpu_pool_list,
        "rounds": [],
    }

    current_round_gpu_count = min_resource_gpus
    round_index = 1

    while _has_schedulable_work(
        planning_report,
        current_round_gpu_count=current_round_gpu_count,
        max_resource_gpus=max_resource_gpus,
    ):
        round_dir = rounds_dir / f"round_{round_index:02d}_g{current_round_gpu_count}"
        round_dir.mkdir(parents=True, exist_ok=True)

        artifacts = build_resource_round_reports(
            planning_report,
            round_output_dir=round_dir,
            shard_series_prefix=f"{pipeline_series_base}_g{current_round_gpu_count}_",
            resource_gpus=current_round_gpu_count,
            gpu_pool=gpu_pool_list,
            carry_forward_series_name=f"{pipeline_series_base}_carry_g{current_round_gpu_count}",
        )

        _run_round_watchdogs(
            round_dir=round_dir,
            shard_reports=[Path(path) for path in artifacts["shard_reports"]],
            conda_env=conda_env,
            poll_seconds=poll_seconds,
            stall_seconds=stall_seconds,
            idle_util_threshold=idle_util_threshold,
            logs_dir=logs_dir,
        )

        merged_state_report = merge_round_reports_for_state(
            planning_report,
            carry_forward_report=artifacts["carry_forward_report"],
            shard_reports=artifacts["shard_reports"],
            merged_output_dir=round_dir,
            merged_series_name=f"{pipeline_series_base}_state_r{round_index:02d}_g{current_round_gpu_count}",
        )

        next_planning_report = advance_resource_round_state(
            merged_state_report,
            output_dir=round_dir,
            merged_series_name=f"{pipeline_series_base}_planning_r{round_index + 1:02d}",
            current_round_gpu_count=current_round_gpu_count,
            max_resource_gpus=max_resource_gpus,
        )
        planning_report = next_planning_report

        round_manifest = _load_report(artifacts["manifest_path"])
        round_manifest["merged_state_report"] = str(merged_state_report)
        round_manifest["next_planning_report"] = str(next_planning_report)
        round_manifest["round_index"] = round_index
        _write_json(round_dir / "round_manifest.json", round_manifest)
        top_manifest["rounds"].append(
            {
                "round_index": round_index,
                "resource_gpus": current_round_gpu_count,
                "round_dir": str(round_dir),
                "manifest_path": str(round_dir / "round_manifest.json"),
                "merged_state_report": str(merged_state_report),
                "next_planning_report": str(next_planning_report),
            }
        )

        payload = _load_report(next_planning_report)
        current_round_gpu_count = int(
            payload.get("series_config", {}).get("current_round_gpu_count", current_round_gpu_count * 2)
        )
        round_index += 1

    final_payload = _load_report(planning_report)
    final_payload["series_name"] = f"{pipeline_series_base}_final"
    planning_report = _write_json(final_dir / "final_planning_report.json", final_payload)

    final_report = _materialize_final_report(planning_report, final_dir)
    top_manifest["final_report"] = str(final_report)
    top_manifest["planning_report"] = str(planning_report)
    _write_json(meta_dir / "summary.json", top_manifest)
    _write_json(pipeline_root / "manifest.json", top_manifest)
    return final_report


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Workflow helpers for report splitting, merging, and recursive GPU escalation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    recursive_parser = subparsers.add_parser("run-recursive")
    recursive_parser.add_argument("--report")
    recursive_parser.add_argument("--config")
    recursive_parser.add_argument("--gpu-pool", required=True, help="Comma-separated GPU ids, e.g. 0,1,2,3")
    recursive_parser.add_argument("--min-resource-gpus", type=int, default=1)
    recursive_parser.add_argument("--max-resource-gpus", type=int, required=True)
    recursive_parser.add_argument("--conda-env", default="llm")
    recursive_parser.add_argument("--poll-seconds", type=float, default=30.0)
    recursive_parser.add_argument("--stall-seconds", type=float, default=600.0)
    recursive_parser.add_argument("--idle-util-threshold", type=float, default=5.0)
    recursive_parser.add_argument("--max-workers", type=int, default=8)

    split_parser = subparsers.add_parser("split-report")
    split_parser.add_argument("--report", required=True)
    split_parser.add_argument("--split-output-dir", required=True)
    split_parser.add_argument("--shard-series-prefix", required=True)
    split_parser.add_argument("--shard-label", action="append", required=True)

    filtered_split_parser = subparsers.add_parser("split-filtered-resume")
    filtered_split_parser.add_argument("--report", required=True)
    filtered_split_parser.add_argument("--split-output-dir", required=True)
    filtered_split_parser.add_argument("--shard-series-prefix", required=True)
    filtered_split_parser.add_argument("--shard-label", action="append", required=True)
    filtered_split_parser.add_argument("--defer-model", action="append", default=[])
    filtered_split_parser.add_argument("--carry-forward-series-name", required=True)

    recovery_parser = subparsers.add_parser("build-recovery-report")
    recovery_parser.add_argument("--report", required=True)
    recovery_parser.add_argument("--output-dir", required=True)
    recovery_parser.add_argument("--series-name", required=True)
    recovery_parser.add_argument("--defer-model", action="append", default=[])

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

    if args.command == "run-recursive":
        final_report = orchestrate_resource_pipeline(
            report_path=args.report,
            config_path=args.config,
            gpu_pool=args.gpu_pool,
            min_resource_gpus=args.min_resource_gpus,
            max_resource_gpus=args.max_resource_gpus,
            conda_env=args.conda_env,
            poll_seconds=args.poll_seconds,
            stall_seconds=args.stall_seconds,
            idle_util_threshold=args.idle_util_threshold,
            max_workers=args.max_workers,
        )
        print(final_report)
        return 0

    if args.command == "split-report":
        report_paths = split_resume_report(
            args.report,
            split_output_dir=args.split_output_dir,
            shard_series_prefix=args.shard_series_prefix,
            shard_labels=args.shard_label,
        )
        for report_path in report_paths:
            print(report_path)
        return 0

    if args.command == "split-filtered-resume":
        artifact_paths = split_filtered_resume_report(
            args.report,
            split_output_dir=args.split_output_dir,
            shard_series_prefix=args.shard_series_prefix,
            shard_labels=args.shard_label,
            deferred_models=args.defer_model,
            carry_forward_series_name=args.carry_forward_series_name,
        )
        print(artifact_paths["carry_forward_report"])
        for report_path in artifact_paths["shard_reports"]:
            print(report_path)
        print(artifact_paths["manifest_path"])
        return 0

    if args.command == "build-recovery-report":
        report_path = build_recovery_resume_report(
            args.report,
            output_dir=args.output_dir,
            series_name=args.series_name,
            deferred_models=args.defer_model,
        )
        print(report_path)
        return 0

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
