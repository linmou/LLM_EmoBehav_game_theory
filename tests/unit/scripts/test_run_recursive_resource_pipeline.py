#!/usr/bin/env python3
# Responsible file: emotion_experiment_engine/resource_recursive_workflow.py
# Purpose: verify the recursive resource pipeline stores its own planning metadata under the config output directory while leaving experiment result directories directly under that output directory.

from __future__ import annotations

import json
import sys
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


def test_orchestrate_resource_pipeline_stops_after_max_resource_failures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "original_outputs"
    source_report = tmp_path / "source_report.json"
    _write_report(
        source_report,
        series_name="resource_series",
        series_config={
            "models": ["model-a"],
            "benchmarks": ["Sealed_Auction", "Beauty_Contest"],
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
                "benchmark_name": "Sealed_Auction",
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
                "benchmark_name": "Beauty_Contest",
                "required_gpu_count": 1,
                "last_attempt_gpu_count": None,
                "resource_failure_blocked": False,
                "error": None,
            },
        },
    )

    def _fake_run_round(*, round_dir: Path, shard_reports: list[Path], **_kwargs) -> None:
        for shard_report in shard_reports:
            payload = _load_json(shard_report)
            for exp in payload["experiments"].values():
                exp["status"] = "failed"
                exp["error"] = f"{round_dir.name}-oom"
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
    top_manifest = _load_json(pipeline_root / "manifest.json")
    final_payload = _load_json(final_report)

    assert [entry["resource_gpus"] for entry in top_manifest["rounds"]] == [1, 2]
    assert len(top_manifest["rounds"]) == 2
    assert not (pipeline_root / "rounds" / "round_03_g2").exists()
    assert final_payload["experiments"]["exp_model_a_primary"]["status"] == "failed"
    assert final_payload["experiments"]["exp_model_a_sibling"]["status"] == "failed"


def test_orchestrate_resource_pipeline_resumes_from_last_stable_round(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Gherkin:
    Feature: resume recursive resource pipeline
      Scenario: rerun after one stable round completed
        Given a recursive pipeline root with round_01_g1 and its next planning report
        When run-recursive is started again for the same output directory
        Then it resumes from round_02_g2 instead of rebuilding round_01_g1
        And it preserves the prior round manifest entries.
    """
    output_dir = tmp_path / "results"
    pipeline_root = output_dir / "resource_pipeline"
    source_dir = pipeline_root / "source"
    meta_dir = pipeline_root / "meta"
    rounds_dir = pipeline_root / "rounds"
    final_dir = pipeline_root / "final"
    round1_dir = rounds_dir / "round_01_g1"
    for directory in [source_dir, meta_dir / "logs", round1_dir, final_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    source_report = source_dir / "source_report.json"
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
            "exp_model_a": {
                "exp_id": "exp_model_a",
                "status": "completed",
                "output_dir": str(_make_run_dir(output_dir / "model_a_round1", "task_run")),
                "model_name": "model-a",
                "benchmark_name": "Prisoners_Dilemma",
                "required_gpu_count": 1,
                "last_attempt_gpu_count": None,
                "resource_failure_blocked": False,
                "error": None,
            },
            "exp_model_b": {
                "exp_id": "exp_model_b",
                "status": "pending",
                "output_dir": None,
                "model_name": "model-b",
                "benchmark_name": "Trust_Game",
                "required_gpu_count": 2,
                "last_attempt_gpu_count": 1,
                "resource_failure_blocked": False,
                "error": None,
            },
        },
    )

    planning_report = round1_dir / "resource_series_planning_r02_resume_report.json"
    _write_report(
        planning_report,
        series_name="resource_series_planning_r02",
        series_config={
            "models": ["model-a", "model-b"],
            "benchmarks": ["Prisoners_Dilemma", "Trust_Game"],
            "output_dir": str(output_dir),
            "resource_pipeline": True,
            "min_resource_gpus": 1,
            "max_resource_gpus": 2,
            "gpu_pool": ["0", "1"],
            "current_round_gpu_count": 2,
            "resource_round_index": 2,
            "stop_model_on_failure": True,
            "source_report": str(round1_dir / "resource_series_state_r01_g1_resume_report.json"),
        },
        experiments={
            "exp_model_a": {
                "exp_id": "exp_model_a",
                "status": "completed",
                "output_dir": str(output_dir / "model_a_round1" / "task_run"),
                "model_name": "model-a",
                "benchmark_name": "Prisoners_Dilemma",
                "required_gpu_count": 1,
                "last_attempt_gpu_count": None,
                "resource_failure_blocked": False,
                "error": None,
            },
            "exp_model_b": {
                "exp_id": "exp_model_b",
                "status": "pending",
                "output_dir": None,
                "model_name": "model-b",
                "benchmark_name": "Trust_Game",
                "required_gpu_count": 2,
                "last_attempt_gpu_count": 1,
                "resource_failure_blocked": False,
                "error": None,
            },
        },
    )

    (round1_dir / "round_manifest.json").write_text(
        json.dumps(
            {
                "round_index": 1,
                "resource_gpus": 1,
                "round_dir": str(round1_dir),
                "next_planning_report": str(planning_report),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (pipeline_root / "manifest.json").write_text(
        json.dumps(
            {
                "source_report": str(source_report),
                "planning_report": str(planning_report),
                "gpu_pool": ["0", "1"],
                "rounds": [
                    {
                        "round_index": 1,
                        "resource_gpus": 1,
                        "round_dir": str(round1_dir),
                        "manifest_path": str(round1_dir / "round_manifest.json"),
                        "next_planning_report": str(planning_report),
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (meta_dir / "summary.json").write_text(
        (pipeline_root / "manifest.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (meta_dir / "pipeline_config.json").write_text(
        json.dumps(
            {
                "source_report": str(source_report),
                "config_path": None,
                "pipeline_root": str(pipeline_root),
                "pipeline_series_base": "resource_series",
                "gpu_pool": ["0", "1"],
                "min_resource_gpus": 1,
                "max_resource_gpus": 2,
                "conda_env": "llm",
                "poll_seconds": 1.0,
                "stall_seconds": 1.0,
                "idle_util_threshold": 5.0,
                "max_workers": 2,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    seen_rounds: list[str] = []

    def _fake_run_round(*, round_dir: Path, shard_reports: list[Path], **_kwargs) -> None:
        seen_rounds.append(round_dir.name)
        completed_run = _make_run_dir(output_dir / "model_b_round2", "task_run")
        for shard_report in shard_reports:
            payload = _load_json(shard_report)
            for exp in payload["experiments"].values():
                exp["status"] = "completed"
                exp["output_dir"] = str(completed_run)
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

    assert seen_rounds == ["round_02_g2"]
    top_manifest = _load_json(pipeline_root / "manifest.json")
    assert [entry["resource_gpus"] for entry in top_manifest["rounds"]] == [1, 2]
    assert top_manifest["rounds"][0]["round_index"] == 1
    assert top_manifest["rounds"][1]["round_index"] == 2
    final_payload = _load_json(final_report)
    assert final_payload["experiments"]["exp_model_a"]["status"] == "completed"
    assert final_payload["experiments"]["exp_model_b"]["status"] == "completed"


def test_orchestrate_resource_pipeline_resumes_interrupted_round_from_partial_shard_reports(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "results"
    pipeline_root = output_dir / "resource_pipeline"
    source_dir = pipeline_root / "source"
    meta_dir = pipeline_root / "meta"
    round1_dir = pipeline_root / "rounds" / "round_01_g1"
    round2_dir = pipeline_root / "rounds" / "round_02_g2"
    for directory in [source_dir, meta_dir / "logs", round1_dir, round2_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    source_report = source_dir / "source_report.json"
    _write_report(
        source_report,
        series_name="resource_series",
        series_config={
            "output_dir": str(output_dir),
            "resource_pipeline": True,
            "min_resource_gpus": 1,
            "max_resource_gpus": 2,
            "gpu_pool": ["0", "1"],
            "current_round_gpu_count": 1,
            "resource_round_index": 1,
            "stop_model_on_failure": True,
            "models": ["model-a"],
            "benchmarks": ["Trust_Game"],
        },
        experiments={
            "exp_completed": {
                "exp_id": "exp_completed",
                "status": "pending",
                "output_dir": None,
                "model_name": "model-a",
                "benchmark_name": "Trust_Game",
                "required_gpu_count": 2,
                "last_attempt_gpu_count": 1,
                "resource_failure_blocked": False,
                "error": None,
            },
            "exp_retry": {
                "exp_id": "exp_retry",
                "status": "pending",
                "output_dir": None,
                "model_name": "model-b",
                "benchmark_name": "Trust_Game",
                "required_gpu_count": 2,
                "last_attempt_gpu_count": 1,
                "resource_failure_blocked": False,
                "error": None,
            },
        },
    )
    planning_report = round1_dir / "resource_series_planning_r02_resume_report.json"
    _write_report(
        planning_report,
        series_name="resource_series_planning_r02",
        series_config={
            "output_dir": str(output_dir),
            "resource_pipeline": True,
            "min_resource_gpus": 1,
            "max_resource_gpus": 2,
            "gpu_pool": ["0", "1"],
            "current_round_gpu_count": 2,
            "resource_round_index": 2,
            "stop_model_on_failure": True,
            "models": ["model-a"],
            "benchmarks": ["Trust_Game"],
        },
        experiments={
            "exp_completed": {
                "exp_id": "exp_completed",
                "status": "pending",
                "output_dir": None,
                "model_name": "model-a",
                "benchmark_name": "Trust_Game",
                "required_gpu_count": 2,
                "last_attempt_gpu_count": 1,
                "resource_failure_blocked": False,
                "error": None,
            },
            "exp_retry": {
                "exp_id": "exp_retry",
                "status": "pending",
                "output_dir": None,
                "model_name": "model-b",
                "benchmark_name": "Trust_Game",
                "required_gpu_count": 2,
                "last_attempt_gpu_count": 1,
                "resource_failure_blocked": False,
                "error": None,
            },
        },
    )
    (round1_dir / "round_manifest.json").write_text(
        json.dumps({"next_planning_report": str(planning_report)}, indent=2),
        encoding="utf-8",
    )
    carry_report = round2_dir / "resource_series_carry_g2_resume_report.json"
    _write_report(
        carry_report,
        series_name="resource_series_carry_g2",
        series_config={"workflow_role": "resource_carry_forward"},
        experiments={},
    )
    completed_run = _make_run_dir(output_dir / "model_a_round2_partial", "task_run")
    shard_report = round2_dir / "resource_series_g2_00_resume_report.json"
    _write_report(
        shard_report,
        series_name="resource_series_g2_00",
        series_config={
            "workflow_role": "resource_round_shard",
            "assigned_gpu_ids": ["0", "1"],
            "current_round_gpu_count": 2,
        },
        experiments={
            "exp_completed": {
                "exp_id": "exp_completed",
                "status": "completed",
                "output_dir": str(completed_run),
                "model_name": "model-a",
                "benchmark_name": "Trust_Game",
                "required_gpu_count": 2,
                "last_attempt_gpu_count": 1,
                "resource_failure_blocked": False,
                "error": None,
            },
            "exp_retry": {
                "exp_id": "exp_retry",
                "status": "failed",
                "output_dir": None,
                "model_name": "model-b",
                "benchmark_name": "Trust_Game",
                "required_gpu_count": 2,
                "last_attempt_gpu_count": 1,
                "resource_failure_blocked": False,
                "error": "old bug",
            },
        },
    )
    (round2_dir / "resource_round_manifest.json").write_text(
        json.dumps(
            {
                "source_report": str(planning_report),
                "resource_gpus": 2,
                "carry_forward_report": str(carry_report),
                "shard_reports": [str(shard_report)],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    stale_marker = round2_dir / "stale.txt"
    stale_marker.write_text("old partial state", encoding="utf-8")
    (meta_dir / "pipeline_config.json").write_text(
        json.dumps(
            {
                "source_report": str(source_report),
                "config_path": None,
                "pipeline_root": str(pipeline_root),
                "pipeline_series_base": "resource_series",
                "gpu_pool": ["0", "1"],
                "min_resource_gpus": 1,
                "max_resource_gpus": 2,
                "conda_env": "llm",
                "poll_seconds": 1.0,
                "stall_seconds": 1.0,
                "idle_util_threshold": 5.0,
                "max_workers": 2,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    seen_rounds: list[str] = []

    def _fake_run_round(*, round_dir: Path, shard_reports: list[Path], **_kwargs) -> None:
        seen_rounds.append(round_dir.name)
        assert len(shard_reports) == 1
        payload = _load_json(shard_reports[0])
        assert set(payload["experiments"]) == {"exp_retry"}
        retried_run = _make_run_dir(output_dir / "model_b_round2_retried", "task_run")
        payload["experiments"]["exp_retry"]["status"] = "completed"
        payload["experiments"]["exp_retry"]["output_dir"] = str(retried_run)
        shard_reports[0].write_text(json.dumps(payload, indent=2), encoding="utf-8")

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

    assert seen_rounds == ["round_02_g2"]
    assert stale_marker.exists()
    final_payload = _load_json(final_report)
    assert final_payload["experiments"]["exp_completed"]["status"] == "completed"
    assert final_payload["experiments"]["exp_completed"]["output_dir"] == str(completed_run)
    assert final_payload["experiments"]["exp_retry"]["status"] == "completed"
    round2_manifest = _load_json(round2_dir / "round_manifest.json")
    assert round2_manifest["round_index"] == 2


def test_orchestrate_resource_pipeline_returns_existing_final_report_when_complete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "results"
    pipeline_root = output_dir / "resource_pipeline"
    source_dir = pipeline_root / "source"
    meta_dir = pipeline_root / "meta"
    final_dir = pipeline_root / "final"
    for directory in [source_dir, meta_dir, final_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    source_report = source_dir / "source_report.json"
    final_report = final_dir / "final_report.json"
    _write_report(
        source_report,
        series_name="resource_series",
        series_config={
            "output_dir": str(output_dir),
            "resource_pipeline": True,
            "min_resource_gpus": 1,
            "max_resource_gpus": 2,
            "gpu_pool": ["0", "1"],
            "current_round_gpu_count": 1,
            "resource_round_index": 1,
            "models": ["model-a"],
            "benchmarks": ["Trust_Game"],
        },
        experiments={},
    )
    _write_report(
        final_report,
        series_name="resource_series_final",
        series_config={
            "output_dir": str(output_dir),
            "resource_pipeline": True,
            "min_resource_gpus": 1,
            "max_resource_gpus": 2,
            "gpu_pool": ["0", "1"],
            "current_round_gpu_count": 2,
            "resource_round_index": 2,
            "models": ["model-a"],
            "benchmarks": ["Trust_Game"],
        },
        experiments={
            "exp_model_a": {
                "exp_id": "exp_model_a",
                "status": "completed",
                "output_dir": str(_make_run_dir(output_dir / "model_a_round2", "task_run")),
                "model_name": "model-a",
                "benchmark_name": "Trust_Game",
                "required_gpu_count": 2,
            }
        },
    )
    (meta_dir / "pipeline_config.json").write_text(
        json.dumps(
            {
                "source_report": str(source_report),
                "config_path": None,
                "pipeline_root": str(pipeline_root),
                "pipeline_series_base": "resource_series",
                "gpu_pool": ["0", "1"],
                "min_resource_gpus": 1,
                "max_resource_gpus": 2,
                "conda_env": "llm",
                "poll_seconds": 1.0,
                "stall_seconds": 1.0,
                "idle_util_threshold": 5.0,
                "max_workers": 2,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    def _fail_if_called(*_args, **_kwargs) -> None:
        raise AssertionError("resume should not schedule new rounds once final report exists")

    monkeypatch.setattr(workflow, "_run_round_watchdogs", _fail_if_called)

    resumed_report = workflow.orchestrate_resource_pipeline(
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

    assert resumed_report == final_report


def test_orchestrate_resource_pipeline_rejects_mismatched_saved_gpu_pool(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "results"
    pipeline_root = output_dir / "resource_pipeline"
    source_dir = pipeline_root / "source"
    meta_dir = pipeline_root / "meta"
    for directory in [source_dir, meta_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    source_report = source_dir / "source_report.json"
    _write_report(
        source_report,
        series_name="resource_series",
        series_config={
            "output_dir": str(output_dir),
            "resource_pipeline": True,
            "min_resource_gpus": 1,
            "max_resource_gpus": 2,
            "gpu_pool": ["0", "1"],
            "current_round_gpu_count": 1,
            "resource_round_index": 1,
            "models": ["model-a"],
            "benchmarks": ["Trust_Game"],
        },
        experiments={},
    )
    (meta_dir / "pipeline_config.json").write_text(
        json.dumps(
            {
                "source_report": str(source_report),
                "config_path": None,
                "pipeline_root": str(pipeline_root),
                "pipeline_series_base": "resource_series",
                "gpu_pool": ["0", "1"],
                "min_resource_gpus": 1,
                "max_resource_gpus": 2,
                "conda_env": "llm",
                "poll_seconds": 1.0,
                "stall_seconds": 1.0,
                "idle_util_threshold": 5.0,
                "max_workers": 2,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    try:
        workflow.orchestrate_resource_pipeline(
            report_path=source_report,
            gpu_pool=["2", "3"],
            min_resource_gpus=1,
            max_resource_gpus=2,
            conda_env="llm",
            poll_seconds=1.0,
            stall_seconds=1.0,
            idle_util_threshold=5.0,
            max_workers=2,
        )
    except ValueError as exc:
        assert "gpu_pool" in str(exc)
    else:
        raise AssertionError("expected resume to reject mismatched saved gpu_pool")


def test_orchestrate_resource_pipeline_interactively_adopts_new_config_for_existing_pipeline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "results"
    config_path = tmp_path / "series.yaml"
    new_config = {
        "models": ["model-a"],
        "emotions": ["anger"],
        "intensities": [1.0],
        "benchmarks": [{"name": "game_theory_decision", "task_type": "Trust_Game"}],
        "output_dir": str(output_dir),
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
        "stop_model_on_failure": False,
    }
    config_path.write_text(yaml.safe_dump(new_config), encoding="utf-8")

    pipeline_root = output_dir / "resource_pipeline"
    source_dir = pipeline_root / "source"
    meta_dir = pipeline_root / "meta"
    for directory in [source_dir, meta_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    source_report = source_dir / "source_report.json"
    _write_report(
        source_report,
        series_name="resource_series",
        series_config={
            "models": ["model-a"],
            "emotions": ["anger"],
            "intensities": [1.0],
            "benchmarks": [{"name": "game_theory_decision", "task_type": "Trust_Game"}],
            "output_dir": str(output_dir),
            "loading_config": new_config["loading_config"],
            "stop_model_on_failure": True,
            "resource_pipeline": True,
            "gpu_pool": ["0", "1"],
            "min_resource_gpus": 1,
            "max_resource_gpus": 2,
            "current_round_gpu_count": 1,
            "resource_round_index": 1,
        },
        experiments={},
    )
    (meta_dir / "pipeline_config.json").write_text(
        json.dumps(
            {
                "source_report": str(source_report),
                "config_path": str(config_path),
                "pipeline_root": str(pipeline_root),
                "pipeline_series_base": "series",
                "gpu_pool": ["0", "1"],
                "min_resource_gpus": 1,
                "max_resource_gpus": 2,
                "conda_env": "llm",
                "poll_seconds": 1.0,
                "stall_seconds": 1.0,
                "idle_util_threshold": 5.0,
                "max_workers": 2,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    resumed_report = workflow.orchestrate_resource_pipeline(
        config_path=config_path,
        gpu_pool=["0", "1"],
        min_resource_gpus=1,
        max_resource_gpus=2,
        conda_env="llm",
        poll_seconds=1.0,
        stall_seconds=1.0,
        idle_util_threshold=5.0,
        max_workers=2,
    )

    assert resumed_report == pipeline_root / "final" / "final_report.json"
    updated_source = _load_json(source_report)
    assert updated_source["series_config"]["stop_model_on_failure"] is False


def test_orchestrate_resource_pipeline_schedules_failed_work_from_input_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "results"
    source_report = tmp_path / "source_report.json"
    _write_report(
        source_report,
        series_name="resource_series",
        series_config={
            "models": ["model-a"],
            "benchmarks": ["Trust_Game"],
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
            "exp_model_a": {
                "exp_id": "exp_model_a",
                "status": "failed",
                "output_dir": None,
                "model_name": "model-a",
                "benchmark_name": "Trust_Game",
                "required_gpu_count": 1,
                "last_attempt_gpu_count": 1,
                "resource_failure_blocked": False,
                "error": "oom",
            },
        },
    )

    seen_rounds: list[str] = []

    def _fake_run_round(*, round_dir: Path, shard_reports: list[Path], **_kwargs) -> None:
        seen_rounds.append(round_dir.name)
        payload = _load_json(shard_reports[0])
        payload["experiments"]["exp_model_a"]["status"] = "completed"
        payload["experiments"]["exp_model_a"]["output_dir"] = str(
            _make_run_dir(output_dir / "model_a_round1_retry", "task_run")
        )
        shard_reports[0].write_text(json.dumps(payload, indent=2), encoding="utf-8")

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

    assert seen_rounds == ["round_01_g1"]
    final_payload = _load_json(final_report)
    assert final_payload["experiments"]["exp_model_a"]["status"] == "completed"


def test_orchestrate_resource_pipeline_applies_approved_config_to_resumed_round_planning(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "results"
    config_path = tmp_path / "series.yaml"
    new_config = {
        "models": ["model-a"],
        "emotions": ["anger"],
        "intensities": [1.0],
        "benchmarks": [{"name": "game_theory_decision", "task_type": "Trust_Game"}],
        "output_dir": str(output_dir),
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
        "stop_model_on_failure": False,
    }
    config_path.write_text(yaml.safe_dump(new_config), encoding="utf-8")

    pipeline_root = output_dir / "resource_pipeline"
    source_dir = pipeline_root / "source"
    meta_dir = pipeline_root / "meta"
    rounds_dir = pipeline_root / "rounds"
    final_dir = pipeline_root / "final"
    round1_dir = rounds_dir / "round_01_g1"
    for directory in [source_dir, meta_dir / "logs", round1_dir, final_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    source_report = source_dir / "source_report.json"
    _write_report(
        source_report,
        series_name="resource_series",
        series_config={
            "models": ["model-a"],
            "emotions": ["anger"],
            "intensities": [1.0],
            "benchmarks": [{"name": "game_theory_decision", "task_type": "Trust_Game"}],
            "output_dir": str(output_dir),
            "loading_config": new_config["loading_config"],
            "stop_model_on_failure": True,
            "resource_pipeline": True,
            "gpu_pool": ["0", "1"],
            "min_resource_gpus": 1,
            "max_resource_gpus": 2,
            "current_round_gpu_count": 1,
            "resource_round_index": 1,
        },
        experiments={
            "exp_model_a": {
                "exp_id": "exp_model_a",
                "status": "pending",
                "output_dir": None,
                "model_name": "model-a",
                "benchmark_name": "Trust_Game",
                "required_gpu_count": 2,
                "last_attempt_gpu_count": 1,
                "resource_failure_blocked": False,
                "error": None,
            },
        },
    )
    planning_report = round1_dir / "resource_series_planning_r02_resume_report.json"
    _write_report(
        planning_report,
        series_name="resource_series_planning_r02",
        series_config={
            "models": ["model-a"],
            "emotions": ["anger"],
            "intensities": [1.0],
            "benchmarks": [{"name": "game_theory_decision", "task_type": "Trust_Game"}],
            "output_dir": str(output_dir),
            "loading_config": new_config["loading_config"],
            "stop_model_on_failure": True,
            "resource_pipeline": True,
            "gpu_pool": ["0", "1"],
            "min_resource_gpus": 1,
            "max_resource_gpus": 2,
            "current_round_gpu_count": 2,
            "resource_round_index": 2,
            "source_report": str(source_report),
        },
        experiments={
            "exp_model_a": {
                "exp_id": "exp_model_a",
                "status": "pending",
                "output_dir": None,
                "model_name": "model-a",
                "benchmark_name": "Trust_Game",
                "required_gpu_count": 2,
                "last_attempt_gpu_count": 1,
                "resource_failure_blocked": False,
                "error": None,
            },
        },
    )
    (round1_dir / "round_manifest.json").write_text(
        json.dumps(
            {
                "round_index": 1,
                "resource_gpus": 1,
                "round_dir": str(round1_dir),
                "next_planning_report": str(planning_report),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (pipeline_root / "manifest.json").write_text(
        json.dumps(
            {
                "source_report": str(source_report),
                "planning_report": str(planning_report),
                "gpu_pool": ["0", "1"],
                "rounds": [
                    {
                        "round_index": 1,
                        "resource_gpus": 1,
                        "round_dir": str(round1_dir),
                        "manifest_path": str(round1_dir / "round_manifest.json"),
                        "next_planning_report": str(planning_report),
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (meta_dir / "summary.json").write_text(
        (pipeline_root / "manifest.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (meta_dir / "pipeline_config.json").write_text(
        json.dumps(
            {
                "source_report": str(source_report),
                "config_path": str(config_path),
                "pipeline_root": str(pipeline_root),
                "pipeline_series_base": "resource_series",
                "gpu_pool": ["0", "1"],
                "min_resource_gpus": 1,
                "max_resource_gpus": 2,
                "conda_env": "llm",
                "poll_seconds": 1.0,
                "stall_seconds": 1.0,
                "idle_util_threshold": 5.0,
                "max_workers": 2,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    captured_stop_flags: list[bool] = []

    def _fake_run_round(*, shard_reports: list[Path], **_kwargs) -> None:
        payload = _load_json(shard_reports[0])
        captured_stop_flags.append(bool(payload["series_config"]["stop_model_on_failure"]))
        payload["experiments"]["exp_model_a"]["status"] = "completed"
        payload["experiments"]["exp_model_a"]["output_dir"] = str(
            _make_run_dir(output_dir / "model_a_round2", "task_run")
        )
        shard_reports[0].write_text(json.dumps(payload, indent=2), encoding="utf-8")

    monkeypatch.setattr(workflow, "_run_round_watchdogs", _fake_run_round)

    workflow.orchestrate_resource_pipeline(
        config_path=config_path,
        gpu_pool=["0", "1"],
        min_resource_gpus=1,
        max_resource_gpus=2,
        conda_env="llm",
        poll_seconds=1.0,
        stall_seconds=1.0,
        idle_util_threshold=5.0,
        max_workers=2,
    )

    assert captured_stop_flags == [False]
