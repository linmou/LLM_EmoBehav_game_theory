#!/usr/bin/env python3
# Responsible file: emotion_experiment_engine/resource_recursive_workflow.py
# Purpose: verify the recursive resource pipeline stores its own planning metadata under the config output directory while leaving experiment result directories directly under that output directory.

from __future__ import annotations

import json
from pathlib import Path

import yaml
from emotion_experiment_engine import resource_recursive_workflow as workflow


def _write_report(path: Path, *, series_name: str, series_config: dict, experiments: dict[str, dict]) -> None:
    payload = {
        "last_updated": "2026-03-25T12:00:00",
        "series_start_time": "2026-03-25T11:00:00",
        "series_duration_seconds": 3600.0,
        "series_name": series_name,
        "series_config": series_config,
        "sessions": [],
        "experiments": experiments,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _make_run_dir(base: Path, name: str) -> Path:
    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "experiment_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "raw_results.json").write_text("[]", encoding="utf-8")
    (run_dir / "summary_results.csv").write_text("score\n1\n", encoding="utf-8")
    return run_dir


def test_orchestrate_resource_pipeline_creates_round_directories_and_final_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "original_outputs"
    source_report = tmp_path / "source_report.json"
    _write_report(
        source_report,
        series_name="resource_series",
        series_config={
            "models": ["model-a", "model-b"],
            "benchmarks": ["Prisoners_Dilemma", "Trust_Game"],
            "output_dir": str(output_dir),
            "resource_pipeline": True,
            "min_resource_gpus": 1,
            "max_resource_gpus": 2,
            "gpu_pool": ["0", "1"],
            "current_round_gpu_count": 1,
            "resource_round_index": 1,
            "stop_model_on_failure": True,
        },
        experiments={
            "exp_model_a_primary": {
                "exp_id": "exp_model_a_primary",
                "status": "pending",
                "output_dir": None,
                "model_name": "model-a",
                "benchmark_name": "Prisoners_Dilemma",
                "required_gpu_count": 1,
                "last_attempt_gpu_count": None,
                "resource_failure_blocked": False,
                "error": None,
            },
            "exp_model_a_sibling": {
                "exp_id": "exp_model_a_sibling",
                "status": "pending",
                "output_dir": None,
                "model_name": "model-a",
                "benchmark_name": "Trust_Game",
                "required_gpu_count": 1,
                "last_attempt_gpu_count": None,
                "resource_failure_blocked": False,
                "error": None,
            },
            "exp_model_b_primary": {
                "exp_id": "exp_model_b_primary",
                "status": "pending",
                "output_dir": None,
                "model_name": "model-b",
                "benchmark_name": "Trust_Game",
                "required_gpu_count": 1,
                "last_attempt_gpu_count": None,
                "resource_failure_blocked": False,
                "error": None,
            },
        },
    )

    def _fake_run_round(*, round_dir: Path, shard_reports: list[Path], **_kwargs) -> None:
        if round_dir.name == "round_01_g1":
            model_b_run = _make_run_dir(output_dir / "model_b_round1", "task_run")
            for shard_report in shard_reports:
                payload = _load_json(shard_report)
                experiments = payload["experiments"]
                if "exp_model_a_primary" in experiments:
                    experiments["exp_model_a_primary"]["status"] = "failed"
                    experiments["exp_model_a_primary"]["error"] = "oom"
                    experiments["exp_model_a_primary"]["last_attempt_gpu_count"] = 1
                    experiments["exp_model_a_sibling"]["status"] = "pending"
                    experiments["exp_model_a_sibling"]["resource_failure_blocked"] = True
                if "exp_model_b_primary" in experiments:
                    experiments["exp_model_b_primary"]["status"] = "completed"
                    experiments["exp_model_b_primary"]["output_dir"] = str(model_b_run)
                shard_report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return

        model_a_primary_run = _make_run_dir(output_dir / "model_a_primary_round2", "task_run")
        model_a_sibling_run = _make_run_dir(output_dir / "model_a_sibling_round2", "task_run")
        for shard_report in shard_reports:
            payload = _load_json(shard_report)
            experiments = payload["experiments"]
            if "exp_model_a_primary" in experiments:
                experiments["exp_model_a_primary"]["status"] = "completed"
                experiments["exp_model_a_primary"]["output_dir"] = str(model_a_primary_run)
            if "exp_model_a_sibling" in experiments:
                experiments["exp_model_a_sibling"]["status"] = "completed"
                experiments["exp_model_a_sibling"]["output_dir"] = str(model_a_sibling_run)
            shard_report.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    monkeypatch.setattr(workflow, "_run_round_watchdogs", _fake_run_round)

    final_report = workflow.orchestrate_resource_pipeline(
        report_path=source_report,
        gpu_pool=["0", "1"],
        min_resource_gpus=1,
        max_resource_gpus=2,
        conda_env="llm",
        poll_seconds=1.0,
        stall_seconds=1.0,
        idle_util_threshold=5.0,
        max_workers=2,
    )

    pipeline_root = output_dir / "resource_pipeline"
    assert (pipeline_root / "source" / "source_report.json").exists()
    assert (pipeline_root / "meta" / "logs").is_dir()
    assert (pipeline_root / "rounds" / "round_01_g1").is_dir()
    assert (pipeline_root / "rounds" / "round_02_g2").is_dir()
    assert (pipeline_root / "final" / "final_report.json").exists()
    assert final_report == pipeline_root / "final" / "final_report.json"

    final_payload = _load_json(final_report)
    assert final_payload["experiments"]["exp_model_a_primary"]["status"] == "completed"
    assert final_payload["experiments"]["exp_model_a_sibling"]["status"] == "completed"
    assert final_payload["experiments"]["exp_model_b_primary"]["status"] == "completed"

    for exp_id in [
        "exp_model_a_primary",
        "exp_model_a_sibling",
        "exp_model_b_primary",
    ]:
        run_dir = Path(final_payload["experiments"][exp_id]["output_dir"])
        assert run_dir.exists()
        assert not run_dir.is_symlink()
        assert str(run_dir).startswith(str(pipeline_root.parent))
        assert "resource_pipeline" not in run_dir.parts

    top_manifest = _load_json(pipeline_root / "manifest.json")
    assert [entry["resource_gpus"] for entry in top_manifest["rounds"]] == [1, 2]
    assert top_manifest["final_report"] == str(final_report)
    assert final_payload["series_name"] == "resource_series_final"
    assert (pipeline_root / "rounds" / "round_01_g1" / "resource_series_state_r01_g1_resume_report.json").exists()
    assert (pipeline_root / "rounds" / "round_01_g1" / "resource_series_planning_r02_resume_report.json").exists()


def test_orchestrate_resource_pipeline_bootstraps_from_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "series.yaml"
    config_payload = {
        "models": ["model-a", "model-b"],
        "emotions": ["anger"],
        "intensities": [1.0],
        "benchmarks": [
            {"name": "game_theory_decision", "task_type": "Prisoners_Dilemma"},
            {"name": "game_theory_decision", "task_type": "Trust_Game"},
        ],
        "output_dir": str(tmp_path / "results"),
        "loading_config": {
            "model_path": "model-a",
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
        "stop_model_on_failure": True,
    }
    config_path.write_text(yaml.safe_dump(config_payload), encoding="utf-8")

    output_dir = tmp_path / "results"

    def _fake_run_round(*, round_dir: Path, shard_reports: list[Path], **_kwargs) -> None:
        for shard_report in shard_reports:
            payload = _load_json(shard_report)
            for exp_id, exp in payload["experiments"].items():
                run_dir = _make_run_dir(output_dir / exp_id, "task_run")
                exp["status"] = "completed"
                exp["output_dir"] = str(run_dir)
            shard_report.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    monkeypatch.setattr(workflow, "_run_round_watchdogs", _fake_run_round)

    final_report = workflow.orchestrate_resource_pipeline(
        config_path=config_path,
        gpu_pool=["2", "3"],
        min_resource_gpus=1,
        max_resource_gpus=2,
        conda_env="llm",
        poll_seconds=1.0,
        stall_seconds=1.0,
        idle_util_threshold=5.0,
        max_workers=2,
    )

    payload = _load_json(final_report)
    assert len(payload["experiments"]) == 4
    assert payload["series_config"]["stop_model_on_failure"] is True
    assert payload["series_config"]["gpu_pool"] == ["2", "3"]
    assert final_report == output_dir / "resource_pipeline" / "final" / "final_report.json"
    assert (output_dir / "resource_pipeline" / "meta" / "pipeline_config.json").exists()
    assert payload["series_name"] == "series_final"
    assert (
        output_dir
        / "resource_pipeline"
        / "rounds"
        / "round_01_g1"
        / "series_state_r01_g1_resume_report.json"
    ).exists()
    for experiment in payload["experiments"].values():
        run_dir = Path(experiment["output_dir"])
        assert run_dir.exists()
        assert not run_dir.is_symlink()
        assert str(run_dir).startswith(str(output_dir))
        assert "resource_pipeline" not in run_dir.parts
