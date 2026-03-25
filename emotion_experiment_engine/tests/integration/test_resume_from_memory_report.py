#!/usr/bin/env python3
"""
Tests for continuing an experiment series from a saved MemoryExperimentReport
and for persisting the experiment series config inside the report file.

Responsible file: emotion_experiment_engine/emotion_experiment_series_runner.py

This suite verifies two behaviors:
1) The report JSON persists a `series_config` snapshot of the original config.
2) A new runner can resume from a saved report JSON by executing only the
   experiments that are still not completed in the report.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest
import yaml

from emotion_experiment_engine.emotion_experiment_series_runner import (
    MemoryExperimentSeriesRunner,
)


def _write_yaml(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f)


def _basic_config(tmpdir: str, num_benchmarks: int = 3) -> dict:
    benches: List[dict] = []
    for i in range(num_benchmarks):
        benches.append(
            {
                "name": f"bench_{i}",
                "task_type": f"task_{i}",
                "sample_limit": 5,
                "enable_auto_truncation": False,
                "truncation_strategy": "right",
                "preserve_ratio": 0.8,
            }
        )

    return {
        "models": ["dummy/model"],
        "emotions": ["anger", "happiness"],
        "intensities": [0.5, 1.0],
        "benchmarks": benches,
        "output_dir": str(Path(tmpdir) / "results"),
        "loading_config": {
            "model_path": "dummy/model",
            "gpu_memory_utilization": 0.8,
            "tensor_parallel_size": 1,
            "max_model_len": 1024,
            "enforce_eager": True,
            "quantization": None,
            "trust_remote_code": True,
            "dtype": "float16",
            "seed": 42,
            "disable_custom_all_reduce": False,
            "additional_vllm_kwargs": {},
        },
    }


@pytest.mark.integration
def test_report_includes_series_config_and_resume_from_report_executes_pendings():
    tmpdir = tempfile.mkdtemp()
    cfg_path = Path(tmpdir) / "series.yaml"
    config = _basic_config(tmpdir, num_benchmarks=3)
    _write_yaml(cfg_path, config)

    # Phase 1: run once to generate a report and all experiments
    with patch(
        "emotion_experiment_engine.emotion_experiment_series_runner.MemoryExperimentSeriesRunner._check_model_existence",
        return_value="/resolved/model",
    ):
        calls_phase1 = []

        def _run_first(benchmark_config, model_name, exp_id):
            calls_phase1.append(exp_id)
            return True

        with patch(
            "emotion_experiment_engine.emotion_experiment_series_runner.MemoryExperimentSeriesRunner.run_single_experiment",
            side_effect=_run_first,
        ):
            runner = MemoryExperimentSeriesRunner(str(cfg_path), series_name="t_series", dry_run=False)
            runner.run_experiment_series()

    # Verify report exists and contains a series_config snapshot
    report_path = runner.report.report_file
    assert report_path.exists(), "Report file should exist after run"

    with open(report_path, "r") as f:
        data = json.load(f)

    assert "series_config" in data, "Report should persist the series_config"
    series_cfg = data["series_config"]
    # Minimal sanity checks on captured config snapshot
    for k in ["models", "emotions", "intensities", "benchmarks", "loading_config", "output_dir"]:
        assert k in series_cfg, f"series_config should include '{k}'"

    # Mutate the report to mark some experiments as pending again
    # Choose the first two experiments to resume
    all_exps = list(data["experiments"].values())
    assert len(all_exps) >= 3, "Expected at least 3 experiments from config"
    for exp in all_exps[:2]:
        exp["status"] = "pending"
        exp["end_time"] = None
        exp["time_cost_seconds"] = None
        exp["error"] = None

    with open(report_path, "w") as f:
        json.dump(data, f, indent=2)

    # Phase 2: resume from the saved report and ensure only pendings run
    with patch(
        "emotion_experiment_engine.emotion_experiment_series_runner.MemoryExperimentSeriesRunner._check_model_existence",
        return_value="/resolved/model",
    ):
        calls_phase2 = []

        def _run_pending_only(benchmark_config, model_name, exp_id):
            calls_phase2.append(exp_id)
            return True

        with patch(
            "emotion_experiment_engine.emotion_experiment_series_runner.MemoryExperimentSeriesRunner.run_single_experiment",
            side_effect=_run_pending_only,
        ):
            # Create a new runner that resumes purely from the report path
            resumed = MemoryExperimentSeriesRunner(
                config_path=None,
                series_name="t_series",
                resume=str(report_path),
                dry_run=False,
            )
            resumed.run_experiment_series()

    # We should have executed exactly the 2 pending experiments
    assert len(calls_phase2) == 2, f"Expected 2 pending runs, got {len(calls_phase2)}"

    # Final summary should show no pending experiments
    summary = resumed.report.get_summary()
    assert summary["pending"] == 0


@pytest.mark.integration
def test_resume_from_split_report_does_not_reintroduce_missing_experiments():
    tmpdir = tempfile.mkdtemp()
    report_path = Path(tmpdir) / "split_resume_report.json"

    payload = {
        "last_updated": "2026-03-23T16:00:00",
        "series_start_time": "2026-03-23T15:00:00",
        "series_duration_seconds": 3600.0,
        "series_name": "split_series_gpu0",
        "series_config": _basic_config(tmpdir, num_benchmarks=2),
        "sessions": [],
        "experiments": {
            "bench_0_task_0_dummy_model": {
                "benchmark_name": "bench_0_task_0",
                "model_name": "dummy/model",
                "resolved_model_path": "/resolved/model",
                "status": "pending",
                "start_time": None,
                "end_time": None,
                "time_cost_seconds": None,
                "error": None,
                "output_dir": None,
                "exp_id": "bench_0_task_0_dummy_model",
            }
        },
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with patch(
        "emotion_experiment_engine.emotion_experiment_series_runner.MemoryExperimentSeriesRunner._check_model_existence",
        return_value="/resolved/model",
    ):
        calls = []

        def _run_pending_only(benchmark_config, model_name, exp_id):
            calls.append(exp_id)
            return True

        with patch(
            "emotion_experiment_engine.emotion_experiment_series_runner.MemoryExperimentSeriesRunner.run_single_experiment",
            side_effect=_run_pending_only,
        ):
            resumed = MemoryExperimentSeriesRunner(
                config_path=None,
                series_name="split_series_gpu0",
                resume=str(report_path),
                dry_run=False,
            )
            resumed.run_experiment_series()

    assert calls == ["bench_0_task_0_dummy_model"]
    assert set(resumed.report.experiments.keys()) == {"bench_0_task_0_dummy_model"}


@pytest.mark.integration
def test_resume_from_report_reruns_failed_experiments():
    tmpdir = tempfile.mkdtemp()
    report_path = Path(tmpdir) / "resume_failed_report.json"

    payload = {
        "last_updated": "2026-03-23T16:00:00",
        "series_start_time": "2026-03-23T15:00:00",
        "series_duration_seconds": 3600.0,
        "series_name": "resume_failed_series",
        "series_config": _basic_config(tmpdir, num_benchmarks=2),
        "sessions": [],
        "experiments": {
            "bench_0_task_0_dummy_model": {
                "benchmark_name": "bench_0_task_0",
                "model_name": "dummy/model",
                "resolved_model_path": "/resolved/model",
                "status": "failed",
                "start_time": "2026-03-23T15:00:00",
                "end_time": "2026-03-23T15:01:00",
                "time_cost_seconds": 60.0,
                "error": "oom",
                "output_dir": "/tmp/old_failed_run",
                "exp_id": "bench_0_task_0_dummy_model",
            },
            "bench_1_task_1_dummy_model": {
                "benchmark_name": "bench_1_task_1",
                "model_name": "dummy/model",
                "resolved_model_path": "/resolved/model",
                "status": "completed",
                "start_time": "2026-03-23T15:02:00",
                "end_time": "2026-03-23T15:03:00",
                "time_cost_seconds": 60.0,
                "error": None,
                "output_dir": "/tmp/old_completed_run",
                "exp_id": "bench_1_task_1_dummy_model",
            },
        },
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with patch(
        "emotion_experiment_engine.emotion_experiment_series_runner.MemoryExperimentSeriesRunner._check_model_existence",
        return_value="/resolved/model",
    ):
        calls = []

        def _run_failed_again(benchmark_config, model_name, exp_id):
            calls.append(exp_id)
            return True

        with patch(
            "emotion_experiment_engine.emotion_experiment_series_runner.MemoryExperimentSeriesRunner.run_single_experiment",
            side_effect=_run_failed_again,
        ):
            resumed = MemoryExperimentSeriesRunner(
                config_path=None,
                series_name="resume_failed_series",
                resume=str(report_path),
                dry_run=False,
            )
            resumed.run_experiment_series()

    assert calls == ["bench_0_task_0_dummy_model"]
    assert resumed.report.experiments["bench_0_task_0_dummy_model"]["status"] == "completed"
    assert resumed.report.experiments["bench_1_task_1_dummy_model"]["status"] == "completed"


@pytest.mark.integration
def test_resume_from_report_stops_scheduling_same_model_after_first_failure():
    tmpdir = tempfile.mkdtemp()
    report_path = Path(tmpdir) / "resume_stop_model_report.json"
    payload = {
        "last_updated": "2026-03-25T10:00:00",
        "series_start_time": "2026-03-25T09:00:00",
        "series_duration_seconds": 3600.0,
        "series_name": "resume_stop_model_series",
        "series_config": {
            **_basic_config(tmpdir, num_benchmarks=3),
            "stop_model_on_failure": True,
        },
        "sessions": [],
        "experiments": {
            "bench_0_task_0_dummy_model": {
                "benchmark_name": "bench_0_task_0",
                "model_name": "dummy/model",
                "resolved_model_path": "/resolved/model",
                "status": "pending",
                "start_time": None,
                "end_time": None,
                "time_cost_seconds": None,
                "error": None,
                "output_dir": None,
                "exp_id": "bench_0_task_0_dummy_model",
            },
            "bench_1_task_1_dummy_model": {
                "benchmark_name": "bench_1_task_1",
                "model_name": "dummy/model",
                "resolved_model_path": "/resolved/model",
                "status": "pending",
                "start_time": None,
                "end_time": None,
                "time_cost_seconds": None,
                "error": None,
                "output_dir": None,
                "exp_id": "bench_1_task_1_dummy_model",
            },
            "bench_2_task_2_other_model": {
                "benchmark_name": "bench_2_task_2",
                "model_name": "other/model",
                "resolved_model_path": "/resolved/other-model",
                "status": "pending",
                "start_time": None,
                "end_time": None,
                "time_cost_seconds": None,
                "error": None,
                "output_dir": None,
                "exp_id": "bench_2_task_2_other_model",
            },
        },
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    calls: list[str] = []

    def _run_with_first_failure(_benchmark_config, _model_name, exp_id):
        calls.append(exp_id)
        return exp_id != "bench_0_task_0_dummy_model"

    with patch(
        "emotion_experiment_engine.emotion_experiment_series_runner.MemoryExperimentSeriesRunner._check_model_existence",
        side_effect=lambda model_name: f"/resolved/{model_name.replace('/', '-')}",
    ):
        with patch(
            "emotion_experiment_engine.emotion_experiment_series_runner.MemoryExperimentSeriesRunner.run_single_experiment",
            side_effect=_run_with_first_failure,
        ):
            resumed = MemoryExperimentSeriesRunner(
                config_path=None,
                series_name="resume_stop_model_series",
                resume=str(report_path),
                dry_run=False,
            )
            resumed.run_experiment_series()

    assert calls == [
        "bench_0_task_0_dummy_model",
        "bench_2_task_2_other_model",
    ]
    assert resumed.report.experiments["bench_0_task_0_dummy_model"]["status"] == "failed"
    assert resumed.report.experiments["bench_1_task_1_dummy_model"]["status"] == "pending"
    assert resumed.report.experiments["bench_2_task_2_other_model"]["status"] == "completed"


@pytest.mark.integration
def test_resume_from_merged_report_uses_embedded_source_benchmarks():
    """Responsible for emotion_experiment_series_runner.py merged-report resume behavior."""
    tmpdir = tempfile.mkdtemp()
    report_path = Path(tmpdir) / "merged_resume_report.json"
    source_cfg = _basic_config(tmpdir, num_benchmarks=2)

    payload = {
        "last_updated": "2026-03-25T00:00:00",
        "series_start_time": "2026-03-25T00:00:00",
        "series_duration_seconds": 0.0,
        "series_name": "merged_resume_series",
        "series_config": {
            "source_reports": [str(Path(tmpdir) / "source_a.json"), str(Path(tmpdir) / "source_b.json")],
            "merged_from_series_configs": [source_cfg],
        },
        "sessions": [],
        "experiments": {
            "bench_0_task_0_dummy_model": {
                "benchmark_name": "bench_0_task_0",
                "model_name": "dummy/model",
                "resolved_model_path": "/resolved/model",
                "status": "failed",
                "start_time": "2026-03-25T00:00:00",
                "end_time": "2026-03-25T00:01:00",
                "time_cost_seconds": 60.0,
                "error": "old loader error",
                "output_dir": "/tmp/old_failed_run",
                "exp_id": "bench_0_task_0_dummy_model",
            },
            "bench_1_task_1_dummy_model": {
                "benchmark_name": "bench_1_task_1",
                "model_name": "dummy/model",
                "resolved_model_path": "/resolved/model",
                "status": "completed",
                "start_time": "2026-03-25T00:01:00",
                "end_time": "2026-03-25T00:02:00",
                "time_cost_seconds": 60.0,
                "error": None,
                "output_dir": "/tmp/old_completed_run",
                "exp_id": "bench_1_task_1_dummy_model",
            },
        },
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with patch(
        "emotion_experiment_engine.emotion_experiment_series_runner.MemoryExperimentSeriesRunner._check_model_existence",
        return_value="/resolved/model",
    ):
        calls = []

        def _run_failed_again(benchmark_config, model_name, exp_id):
            calls.append((benchmark_config["name"], benchmark_config["task_type"], model_name, exp_id))
            return True

        with patch(
            "emotion_experiment_engine.emotion_experiment_series_runner.MemoryExperimentSeriesRunner.run_single_experiment",
            side_effect=_run_failed_again,
        ):
            resumed = MemoryExperimentSeriesRunner(
                config_path=None,
                series_name="merged_resume_series",
                resume=str(report_path),
                dry_run=False,
            )
            resumed.run_experiment_series()

    assert calls == [("bench_0", "task_0", "/resolved/model", "bench_0_task_0_dummy_model")]
    assert resumed.report.experiments["bench_0_task_0_dummy_model"]["status"] == "completed"
