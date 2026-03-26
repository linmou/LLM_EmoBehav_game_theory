#!/usr/bin/env python3
# Responsible file: scripts/finalize_recovered_recursive_run.py
# Purpose: verify a recovered recursive run can finalize a completed latest round into final recursive artifacts without relying on fragile inline shell Python.

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "finalize_recovered_recursive_run.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "finalize_recovered_recursive_run",
    _MODULE_PATH,
)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules["finalize_recovered_recursive_run"] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _write_report(path: Path, *, series_name: str, series_config: dict, experiments: dict[str, dict]) -> None:
    payload = {
        "last_updated": "2026-03-25T18:00:00",
        "series_start_time": "2026-03-25T17:00:00",
        "series_duration_seconds": 3600.0,
        "series_name": series_name,
        "series_config": series_config,
        "sessions": [],
        "experiments": experiments,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_run_dir(base: Path, name: str) -> Path:
    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "experiment_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "raw_results.json").write_text("[]", encoding="utf-8")
    (run_dir / "summary_results.csv").write_text("score\n1\n", encoding="utf-8")
    return run_dir


def test_finalize_recovered_run_writes_final_artifacts_from_completed_latest_round(tmp_path: Path) -> None:
    recovery_root = tmp_path / "resource_pipeline_recovery_20260325_144848"
    source_dir = recovery_root / "source"
    meta_dir = recovery_root / "meta"
    round1_dir = recovery_root / "rounds" / "round_01_g1_recovered"
    round2_dir = recovery_root / "rounds" / "round_02_g2"
    for directory in [source_dir, meta_dir, round1_dir, round2_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    source_report = source_dir / "source_report.json"
    completed_run = _make_run_dir(tmp_path / "results", "model_a_round2")
    _write_report(
        source_report,
        series_name="resource_series",
        series_config={
            "output_dir": str(tmp_path / "results"),
            "max_resource_gpus": 2,
            "gpu_pool": ["2", "3"],
        },
        experiments={
            "exp_a": {
                "exp_id": "exp_a",
                "status": "completed",
                "output_dir": str(tmp_path / "results" / "model_a_round1"),
                "model_name": "model-a",
                "benchmark_name": "Prisoners_Dilemma",
                "required_gpu_count": 1,
            },
            "exp_b": {
                "exp_id": "exp_b",
                "status": "pending",
                "output_dir": None,
                "model_name": "model-b",
                "benchmark_name": "Trust_Game",
                "required_gpu_count": 1,
            },
        },
    )

    planning_r02 = round1_dir / "planning_r02.json"
    _write_report(
        planning_r02,
        series_name="resource_series_planning_r02",
        series_config={
            "output_dir": str(tmp_path / "results"),
            "current_round_gpu_count": 2,
            "resource_round_index": 2,
            "max_resource_gpus": 2,
            "gpu_pool": ["2", "3"],
        },
        experiments={
            "exp_a": {
                "exp_id": "exp_a",
                "status": "completed",
                "output_dir": str(tmp_path / "results" / "model_a_round1"),
                "model_name": "model-a",
                "benchmark_name": "Prisoners_Dilemma",
                "required_gpu_count": 1,
            },
            "exp_b": {
                "exp_id": "exp_b",
                "status": "pending",
                "output_dir": None,
                "model_name": "model-b",
                "benchmark_name": "Trust_Game",
                "required_gpu_count": 2,
                "last_attempt_gpu_count": 1,
            },
        },
    )
    (round1_dir / "round_manifest.json").write_text(
        json.dumps({"next_planning_report": str(planning_r02)}, indent=2),
        encoding="utf-8",
    )

    carry_report = round2_dir / "carry.json"
    _write_report(
        carry_report,
        series_name="carry",
        series_config={"workflow_role": "resource_carry_forward"},
        experiments={
            "exp_a": {
                "exp_id": "exp_a",
                "status": "completed",
                "output_dir": str(tmp_path / "results" / "model_a_round1"),
                "model_name": "model-a",
                "benchmark_name": "Prisoners_Dilemma",
                "required_gpu_count": 1,
            }
        },
    )
    shard_report = round2_dir / "resource_series_g2_00_resume_report.json"
    _write_report(
        shard_report,
        series_name="resource_series_g2_00",
        series_config={"workflow_role": "resource_round_shard"},
        experiments={
            "exp_b": {
                "exp_id": "exp_b",
                "status": "completed",
                "output_dir": str(completed_run),
                "model_name": "model-b",
                "benchmark_name": "Trust_Game",
                "required_gpu_count": 2,
                "last_attempt_gpu_count": 1,
            }
        },
    )
    (round2_dir / "resource_round_manifest.json").write_text(
        json.dumps(
            {
                "carry_forward_report": str(carry_report),
                "shard_reports": [str(shard_report)],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (meta_dir / "pipeline_config.json").write_text(
        json.dumps(
            {
                "gpu_pool": ["2", "3"],
                "max_resource_gpus": 2,
                "final_merged_series_name": "resource_series_final",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    final_report = _MODULE.finalize_recovered_run(recovery_root=recovery_root)

    final_payload = json.loads(final_report.read_text(encoding="utf-8"))
    assert final_payload["experiments"]["exp_a"]["status"] == "completed"
    assert final_payload["experiments"]["exp_b"]["status"] == "completed"
    assert Path(final_payload["experiments"]["exp_b"]["output_dir"]) == completed_run

    summary = json.loads((meta_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["final_report"] == str(final_report)
    assert len(summary["rounds"]) == 2
    assert (recovery_root / "manifest.json").exists()


def test_finalize_recovered_run_rejects_when_another_resource_tier_is_still_needed(tmp_path: Path) -> None:
    recovery_root = tmp_path / "resource_pipeline_recovery_20260325_200000"
    source_dir = recovery_root / "source"
    meta_dir = recovery_root / "meta"
    round1_dir = recovery_root / "rounds" / "round_01_g1_recovered"
    round2_dir = recovery_root / "rounds" / "round_02_g2"
    for directory in [source_dir, meta_dir, round1_dir, round2_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    source_report = source_dir / "source_report.json"
    _write_report(
        source_report,
        series_name="resource_series",
        series_config={
            "output_dir": str(tmp_path / "results"),
            "max_resource_gpus": 4,
            "gpu_pool": ["0", "1", "2", "3"],
        },
        experiments={
            "exp_b": {
                "exp_id": "exp_b",
                "status": "pending",
                "output_dir": None,
                "model_name": "model-b",
                "benchmark_name": "Trust_Game",
                "required_gpu_count": 2,
                "last_attempt_gpu_count": 1,
            }
        },
    )
    planning_r02 = round1_dir / "planning_r02.json"
    _write_report(
        planning_r02,
        series_name="resource_series_planning_r02",
        series_config={
            "output_dir": str(tmp_path / "results"),
            "current_round_gpu_count": 2,
            "resource_round_index": 2,
            "max_resource_gpus": 4,
            "gpu_pool": ["0", "1", "2", "3"],
        },
        experiments={
            "exp_b": {
                "exp_id": "exp_b",
                "status": "pending",
                "output_dir": None,
                "model_name": "model-b",
                "benchmark_name": "Trust_Game",
                "required_gpu_count": 2,
                "last_attempt_gpu_count": 1,
            }
        },
    )
    (round1_dir / "round_manifest.json").write_text(
        json.dumps({"next_planning_report": str(planning_r02)}, indent=2),
        encoding="utf-8",
    )
    carry_report = round2_dir / "carry.json"
    _write_report(
        carry_report,
        series_name="carry",
        series_config={"workflow_role": "resource_carry_forward"},
        experiments={},
    )
    shard_report = round2_dir / "resource_series_g2_00_resume_report.json"
    _write_report(
        shard_report,
        series_name="resource_series_g2_00",
        series_config={"workflow_role": "resource_round_shard"},
        experiments={
            "exp_b": {
                "exp_id": "exp_b",
                "status": "failed",
                "output_dir": None,
                "model_name": "model-b",
                "benchmark_name": "Trust_Game",
                "required_gpu_count": 2,
                "last_attempt_gpu_count": 1,
                "error": "still oom",
            }
        },
    )
    (round2_dir / "resource_round_manifest.json").write_text(
        json.dumps(
            {
                "carry_forward_report": str(carry_report),
                "shard_reports": [str(shard_report)],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (meta_dir / "pipeline_config.json").write_text(
        json.dumps(
            {
                "gpu_pool": ["0", "1", "2", "3"],
                "max_resource_gpus": 4,
                "final_merged_series_name": "resource_series_final",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    try:
        _MODULE.finalize_recovered_run(recovery_root=recovery_root)
    except ValueError as exc:
        assert "still has schedulable work" in str(exc)
    else:
        raise AssertionError("expected recovered finalizer to reject unfinished higher-tier work")


def test_finalize_recovered_run_handles_first_round_recovery(tmp_path: Path) -> None:
    recovery_root = tmp_path / "resource_pipeline_recovery_20260326_010000"
    source_dir = recovery_root / "source"
    meta_dir = recovery_root / "meta"
    round1_dir = recovery_root / "rounds" / "round_01_g1"
    for directory in [source_dir, meta_dir, round1_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    source_report = source_dir / "source_report.json"
    completed_run = _make_run_dir(tmp_path / "results", "model_a_round1")
    _write_report(
        source_report,
        series_name="resource_series",
        series_config={
            "output_dir": str(tmp_path / "results"),
            "max_resource_gpus": 1,
            "gpu_pool": ["0"],
            "current_round_gpu_count": 1,
            "resource_round_index": 1,
        },
        experiments={
            "exp_a": {
                "exp_id": "exp_a",
                "status": "pending",
                "output_dir": None,
                "model_name": "model-a",
                "benchmark_name": "Trust_Game",
                "required_gpu_count": 1,
            },
        },
    )
    carry_report = round1_dir / "carry.json"
    _write_report(
        carry_report,
        series_name="carry",
        series_config={"workflow_role": "resource_carry_forward"},
        experiments={},
    )
    shard_report = round1_dir / "resource_series_g1_00_resume_report.json"
    _write_report(
        shard_report,
        series_name="resource_series_g1_00",
        series_config={"workflow_role": "resource_round_shard"},
        experiments={
            "exp_a": {
                "exp_id": "exp_a",
                "status": "completed",
                "output_dir": str(completed_run),
                "model_name": "model-a",
                "benchmark_name": "Trust_Game",
                "required_gpu_count": 1,
            }
        },
    )
    (round1_dir / "resource_round_manifest.json").write_text(
        json.dumps(
            {
                "source_report": str(source_report),
                "resource_gpus": 1,
                "carry_forward_report": str(carry_report),
                "shard_reports": [str(shard_report)],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (meta_dir / "pipeline_config.json").write_text(
        json.dumps(
            {
                "gpu_pool": ["0"],
                "max_resource_gpus": 1,
                "final_merged_series_name": "resource_series_final",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    final_report = _MODULE.finalize_recovered_run(recovery_root=recovery_root)

    final_payload = json.loads(final_report.read_text(encoding="utf-8"))
    assert final_payload["experiments"]["exp_a"]["status"] == "completed"
    summary = json.loads((meta_dir / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["rounds"]) == 1
