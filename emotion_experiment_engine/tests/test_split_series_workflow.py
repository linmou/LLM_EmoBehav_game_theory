# Tests for split-series evaluation and merge workflow.
# Responsible files: emotion_experiment_engine/resource_recursive_workflow.py and scripts/run_vlm_mm_game_theory_300_split_sweeps.sh
# Purpose: ensure workflow reports can be merged into one result folder/report and evaluator tmux watchers are launched correctly after consolidating workflow helpers into one module.

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


def _make_run_dir(base: Path, name: str) -> Path:
    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "experiment_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "raw_results.json").write_text("[]", encoding="utf-8")
    (run_dir / "summary_results.csv").write_text("score\n1\n", encoding="utf-8")
    return run_dir


def _write_report(path: Path, *, series_name: str, experiments: dict[str, dict], series_config: dict) -> None:
    payload = {
        "last_updated": "2026-03-21T16:10:00",
        "series_start_time": "2026-03-21T16:00:00",
        "series_duration_seconds": 600.0,
        "series_name": series_name,
        "series_config": series_config,
        "sessions": [
            {
                "session_id": "session-1",
                "start_time": "2026-03-21T16:00:00",
                "end_time": "2026-03-21T16:10:00",
                "end_reason": "completed",
            }
        ],
        "experiments": experiments,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_merge_series_reports_creates_single_folder_and_combined_report(tmp_path: Path) -> None:
    src_a = tmp_path / "sample300_gpu01"
    src_b = tmp_path / "sample300_gpu23"
    src_a.mkdir()
    src_b.mkdir()

    run_a = _make_run_dir(src_a, "modelA_task1")
    run_b = _make_run_dir(src_b, "modelB_task1")

    report_a = src_a / "memory_experiment_series_20260321_16_memory_experiment_report.json"
    report_b = src_b / "memory_experiment_series_20260321_16_memory_experiment_report.json"
    _write_report(
        report_a,
        series_name="vlm_mm_game_theory_300_gpu01",
        series_config={"models": ["modelA"], "benchmarks": ["task1"]},
        experiments={
            "exp_a": {
                "exp_id": "exp_a",
                "status": "completed",
                "output_dir": str(run_a),
                "model_name": "modelA",
                "benchmark_name": "task1",
            }
        },
    )
    _write_report(
        report_b,
        series_name="vlm_mm_game_theory_300_gpu23",
        series_config={"models": ["modelB"], "benchmarks": ["task1"]},
        experiments={
            "exp_b": {
                "exp_id": "exp_b",
                "status": "completed",
                "output_dir": str(run_b),
                "model_name": "modelB",
                "benchmark_name": "task1",
            }
        },
    )

    from emotion_experiment_engine.resource_recursive_workflow import merge_series_reports

    merged_dir = tmp_path / "sample300_merged"
    merged_report = merge_series_reports(
        [report_a, report_b],
        merged_output_dir=merged_dir,
        merged_series_name="vlm_mm_game_theory_300_merged",
    )

    merged_payload = json.loads(merged_report.read_text(encoding="utf-8"))
    assert merged_report.parent == merged_dir
    assert len(merged_payload["experiments"]) == 2
    assert merged_payload["series_name"] == "vlm_mm_game_theory_300_merged"
    assert merged_payload["series_config"]["source_reports"] == [
        str(report_a.resolve()),
        str(report_b.resolve()),
    ]
    assert merged_payload["series_config"]["source_output_dirs"] == [
        str(src_a.resolve()),
        str(src_b.resolve()),
    ]

    merged_run_a = merged_dir / run_a.name
    merged_run_b = merged_dir / run_b.name
    assert merged_run_a.exists()
    assert merged_run_b.exists()
    assert merged_run_a.is_symlink()
    assert merged_run_b.is_symlink()
    assert Path(merged_payload["experiments"]["exp_a"]["output_dir"]) == merged_run_a
    assert Path(merged_payload["experiments"]["exp_b"]["output_dir"]) == merged_run_b


def test_launch_evaluator_tmux_sessions_uses_watch_mode_for_each_report(tmp_path: Path) -> None:
    report_a = tmp_path / "gpu01_report.json"
    report_b = tmp_path / "gpu23_report.json"
    report_a.write_text("{}", encoding="utf-8")
    report_b.write_text("{}", encoding="utf-8")

    launched_commands: list[list[str]] = []

    def _fake_run(cmd: list[str], check: bool) -> None:
        assert check is True
        launched_commands.append(cmd)

    with patch("subprocess.run", side_effect=_fake_run):
        from emotion_experiment_engine.resource_recursive_workflow import launch_eval_watchers_tmux

        session_names = launch_eval_watchers_tmux(
            report_paths=[report_a, report_b],
            env_name="llm-test",
            poll_interval_seconds=15.0,
            max_workers=6,
            session_name_prefix="vlm_eval",
        )

    assert session_names == ["vlm_eval_0", "vlm_eval_1"]
    assert len(launched_commands) == 2
    joined = [" ".join(cmd) for cmd in launched_commands]
    assert "tmux new-session -d -c " in joined[0]
    assert " -s vlm_eval_0" in joined[0]
    assert " -s vlm_eval_1" in joined[1]
    assert "python -m emotion_experiment_engine.evaluate_saved_series --report" in joined[0]
    assert "--watch" in joined[0]
    assert "--watch" in joined[1]
    assert str(report_a.resolve()) in joined[0]
    assert str(report_b.resolve()) in joined[1]


def test_split_resume_report_creates_disjoint_shards_and_requeues_incomplete_work(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    completed_run = _make_run_dir(source_dir, "completed_run")
    running_run = _make_run_dir(source_dir, "running_run")

    source_report = source_dir / "memory_experiment_series_20260323_15_memory_experiment_report.json"
    _write_report(
        source_report,
        series_name="memory_experiment_series",
        series_config={
            "models": ["Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct"],
            "benchmarks": ["Prisoners_Dilemma", "Stag_Hunt"],
            "output_dir": str(source_dir),
        },
        experiments={
            "exp_completed": {
                "exp_id": "exp_completed",
                "status": "completed",
                "output_dir": str(completed_run),
                "model_name": "Qwen2.5-0.5B-Instruct",
                "benchmark_name": "Prisoners_Dilemma",
                "start_time": "2026-03-23T15:00:00",
                "end_time": "2026-03-23T15:10:00",
                "time_cost_seconds": 600.0,
                "error": None,
            },
            "exp_running": {
                "exp_id": "exp_running",
                "status": "running",
                "output_dir": str(running_run),
                "model_name": "Qwen2.5-1.5B-Instruct",
                "benchmark_name": "Prisoners_Dilemma",
                "start_time": "2026-03-23T15:11:00",
                "end_time": None,
                "time_cost_seconds": None,
                "error": None,
            },
            "exp_pending_a": {
                "exp_id": "exp_pending_a",
                "status": "pending",
                "output_dir": None,
                "model_name": "Qwen2.5-0.5B-Instruct",
                "benchmark_name": "Stag_Hunt",
                "start_time": None,
                "end_time": None,
                "time_cost_seconds": None,
                "error": None,
            },
            "exp_failed": {
                "exp_id": "exp_failed",
                "status": "failed",
                "output_dir": str(source_dir / "failed_run"),
                "model_name": "Qwen2.5-1.5B-Instruct",
                "benchmark_name": "Stag_Hunt",
                "start_time": "2026-03-23T15:12:00",
                "end_time": "2026-03-23T15:13:00",
                "time_cost_seconds": 60.0,
                "error": "oom",
            },
        },
    )

    from emotion_experiment_engine.resource_recursive_workflow import split_resume_report

    split_dir = tmp_path / "split"
    shard_paths = split_resume_report(
        source_report,
        split_output_dir=split_dir,
        shard_series_prefix="memory_experiment_series_gpu",
        shard_labels=["0", "1"],
    )

    assert len(shard_paths) == 2
    shard_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in shard_paths]
    shard_exp_ids = [set(payload["experiments"].keys()) for payload in shard_payloads]

    assert shard_exp_ids[0].isdisjoint(shard_exp_ids[1])
    assert shard_exp_ids[0] | shard_exp_ids[1] == {
        "exp_completed",
        "exp_running",
        "exp_pending_a",
        "exp_failed",
    }
    assert "exp_completed" in shard_exp_ids[0]
    assert "exp_completed" not in shard_exp_ids[1]

    running_exp = next(
        payload["experiments"]["exp_running"]
        for payload in shard_payloads
        if "exp_running" in payload["experiments"]
    )
    assert running_exp["status"] == "pending"
    assert running_exp["output_dir"] is None
    assert running_exp["start_time"] is None
    assert running_exp["end_time"] is None
    assert running_exp["time_cost_seconds"] is None

    failed_exp = next(
        payload["experiments"]["exp_failed"]
        for payload in shard_payloads
        if "exp_failed" in payload["experiments"]
    )
    assert failed_exp["status"] == "pending"
    assert failed_exp["output_dir"] is None
    assert failed_exp["error"] is None


def test_build_resource_round_reports_groups_models_and_keeps_future_round_work_out(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    completed_run = _make_run_dir(source_dir, "completed_run")
    source_report = source_dir / "resource_state_report.json"
    _write_report(
        source_report,
        series_name="resource_series",
        series_config={
            "models": ["model-a", "model-b", "model-c"],
            "benchmarks": ["Prisoners_Dilemma", "Trust_Game"],
            "output_dir": str(source_dir),
            "resource_pipeline": True,
            "min_resource_gpus": 1,
            "max_resource_gpus": 4,
            "gpu_pool": ["0", "1", "2"],
            "current_round_gpu_count": 1,
            "resource_round_index": 1,
            "stop_model_on_failure": True,
        },
        experiments={
            "exp_completed": {
                "exp_id": "exp_completed",
                "status": "completed",
                "output_dir": str(completed_run),
                "model_name": "model-a",
                "benchmark_name": "Prisoners_Dilemma",
                "required_gpu_count": 1,
                "last_attempt_gpu_count": 1,
                "resource_failure_blocked": False,
            },
            "exp_model_a_pending_1": {
                "exp_id": "exp_model_a_pending_1",
                "status": "pending",
                "output_dir": None,
                "model_name": "model-a",
                "benchmark_name": "Trust_Game",
                "required_gpu_count": 1,
                "last_attempt_gpu_count": None,
                "resource_failure_blocked": False,
            },
            "exp_model_a_pending_2": {
                "exp_id": "exp_model_a_pending_2",
                "status": "pending",
                "output_dir": None,
                "model_name": "model-a",
                "benchmark_name": "Ultimatum_Game",
                "required_gpu_count": 1,
                "last_attempt_gpu_count": None,
                "resource_failure_blocked": False,
            },
            "exp_model_b_pending": {
                "exp_id": "exp_model_b_pending",
                "status": "pending",
                "output_dir": None,
                "model_name": "model-b",
                "benchmark_name": "Prisoners_Dilemma",
                "required_gpu_count": 1,
                "last_attempt_gpu_count": None,
                "resource_failure_blocked": False,
            },
            "exp_model_c_future": {
                "exp_id": "exp_model_c_future",
                "status": "pending",
                "output_dir": None,
                "model_name": "model-c",
                "benchmark_name": "Trust_Game",
                "required_gpu_count": 2,
                "last_attempt_gpu_count": 1,
                "resource_failure_blocked": False,
            },
        },
    )

    from emotion_experiment_engine.resource_recursive_workflow import build_resource_round_reports

    round_root = tmp_path / "round_01_g1"
    artifacts = build_resource_round_reports(
        source_report,
        round_output_dir=round_root,
        shard_series_prefix="resource_series_g1_",
        resource_gpus=1,
        gpu_pool=["0", "1", "2"],
        carry_forward_series_name="resource_series_carry_forward",
    )

    carry_payload = json.loads(artifacts["carry_forward_report"].read_text(encoding="utf-8"))
    shard_payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in artifacts["shard_reports"]
    ]
    manifest = json.loads(artifacts["manifest_path"].read_text(encoding="utf-8"))

    assert artifacts["gpu_groups"] == [["0"], ["1"], ["2"]]
    assert artifacts["ignored_gpu_ids"] == []
    assert set(carry_payload["experiments"].keys()) == {"exp_completed"}
    assert manifest["scheduled_models"] == ["model-a", "model-b"]
    assert "exp_model_c_future" not in {
        exp_id
        for payload in shard_payloads
        for exp_id in payload["experiments"]
    }
    assert all(
        "exp_model_c_future" not in payload["experiments"]
        for payload in shard_payloads
    )

    model_a_shards = [
        payload for payload in shard_payloads if "exp_model_a_pending_1" in payload["experiments"]
    ]
    assert len(model_a_shards) == 1
    assert set(model_a_shards[0]["experiments"].keys()) == {
        "exp_model_a_pending_1",
        "exp_model_a_pending_2",
    }


def test_advance_resource_round_state_promotes_failed_model_to_next_resource_tier(
    tmp_path: Path,
) -> None:
    state_report = tmp_path / "planning_report.json"
    _write_report(
        state_report,
        series_name="resource_series_state",
        series_config={
            "models": ["model-a", "model-b"],
            "benchmarks": ["Prisoners_Dilemma", "Trust_Game"],
            "output_dir": str(tmp_path),
            "resource_pipeline": True,
            "min_resource_gpus": 1,
            "max_resource_gpus": 4,
            "gpu_pool": ["0", "1", "2", "3"],
            "current_round_gpu_count": 1,
            "resource_round_index": 1,
            "stop_model_on_failure": True,
        },
        experiments={
            "exp_model_a_failed": {
                "exp_id": "exp_model_a_failed",
                "status": "failed",
                "output_dir": None,
                "model_name": "model-a",
                "benchmark_name": "Prisoners_Dilemma",
                "required_gpu_count": 1,
                "last_attempt_gpu_count": 1,
                "resource_failure_blocked": False,
                "error": "CUDA out of memory",
            },
            "exp_model_a_skipped": {
                "exp_id": "exp_model_a_skipped",
                "status": "pending",
                "output_dir": None,
                "model_name": "model-a",
                "benchmark_name": "Trust_Game",
                "required_gpu_count": 1,
                "last_attempt_gpu_count": None,
                "resource_failure_blocked": True,
                "error": None,
            },
            "exp_model_b_pending": {
                "exp_id": "exp_model_b_pending",
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

    from emotion_experiment_engine.resource_recursive_workflow import advance_resource_round_state

    next_report = advance_resource_round_state(
        state_report,
        output_dir=tmp_path / "round_state_next",
        merged_series_name="resource_series_state_round2",
        current_round_gpu_count=1,
        max_resource_gpus=4,
    )

    payload = json.loads(next_report.read_text(encoding="utf-8"))
    assert payload["series_config"]["current_round_gpu_count"] == 2
    assert payload["series_config"]["resource_round_index"] == 2
    assert payload["experiments"]["exp_model_a_failed"]["status"] == "pending"
    assert payload["experiments"]["exp_model_a_failed"]["required_gpu_count"] == 2
    assert payload["experiments"]["exp_model_a_failed"]["last_attempt_gpu_count"] == 1
    assert payload["experiments"]["exp_model_a_failed"]["error"] is None
    assert payload["experiments"]["exp_model_a_skipped"]["status"] == "pending"
    assert payload["experiments"]["exp_model_a_skipped"]["required_gpu_count"] == 2
    assert payload["experiments"]["exp_model_b_pending"]["required_gpu_count"] == 1


def test_merge_round_reports_for_state_preserves_future_round_experiments(
    tmp_path: Path,
) -> None:
    source_report = tmp_path / "planning_report.json"
    _write_report(
        source_report,
        series_name="resource_series_state",
        series_config={
            "models": ["model-a", "model-b", "model-c"],
            "benchmarks": ["Prisoners_Dilemma", "Trust_Game"],
            "output_dir": str(tmp_path),
            "resource_pipeline": True,
            "current_round_gpu_count": 1,
            "resource_round_index": 1,
        },
        experiments={
            "exp_completed": {
                "exp_id": "exp_completed",
                "status": "completed",
                "output_dir": None,
                "model_name": "model-a",
                "benchmark_name": "Prisoners_Dilemma",
                "required_gpu_count": 1,
            },
            "exp_round_one": {
                "exp_id": "exp_round_one",
                "status": "pending",
                "output_dir": None,
                "model_name": "model-b",
                "benchmark_name": "Trust_Game",
                "required_gpu_count": 1,
            },
            "exp_future_round": {
                "exp_id": "exp_future_round",
                "status": "pending",
                "output_dir": None,
                "model_name": "model-c",
                "benchmark_name": "Trust_Game",
                "required_gpu_count": 2,
            },
        },
    )

    carry_report = tmp_path / "carry.json"
    _write_report(
        carry_report,
        series_name="carry",
        series_config={"workflow_role": "resource_carry_forward"},
        experiments={
            "exp_completed": {
                "exp_id": "exp_completed",
                "status": "completed",
                "output_dir": None,
                "model_name": "model-a",
                "benchmark_name": "Prisoners_Dilemma",
                "required_gpu_count": 1,
            }
        },
    )

    shard_report = tmp_path / "shard_00.json"
    _write_report(
        shard_report,
        series_name="shard00",
        series_config={"workflow_role": "resource_round_shard"},
        experiments={
            "exp_round_one": {
                "exp_id": "exp_round_one",
                "status": "failed",
                "output_dir": None,
                "model_name": "model-b",
                "benchmark_name": "Trust_Game",
                "required_gpu_count": 1,
                "error": "boom",
            }
        },
    )

    from emotion_experiment_engine.resource_recursive_workflow import merge_round_reports_for_state

    merged_report = merge_round_reports_for_state(
        source_report,
        carry_forward_report=carry_report,
        shard_reports=[shard_report],
        merged_output_dir=tmp_path / "merged_state",
        merged_series_name="resource_series_state_merged",
    )

    payload = json.loads(merged_report.read_text(encoding="utf-8"))
    assert set(payload["experiments"].keys()) == {
        "exp_completed",
        "exp_round_one",
        "exp_future_round",
    }
    assert payload["experiments"]["exp_round_one"]["status"] == "failed"
    assert payload["experiments"]["exp_future_round"]["required_gpu_count"] == 2


def test_split_filtered_resume_creates_carry_forward_and_excludes_deferred_models(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    completed_run = _make_run_dir(source_dir, "completed_run")
    source_report = source_dir / "memory_experiment_series_20260323_16_memory_experiment_report.json"
    deferred_model = "/models/large-3b"

    _write_report(
        source_report,
        series_name="memory_experiment_series",
        series_config={
            "models": ["small-a", "small-b", deferred_model],
            "benchmarks": ["Prisoners_Dilemma", "Ultimatum_Game_Responder"],
            "output_dir": str(source_dir),
        },
        experiments={
            "exp_completed": {
                "exp_id": "exp_completed",
                "status": "completed",
                "output_dir": str(completed_run),
                "model_name": "small-a",
                "benchmark_name": "Prisoners_Dilemma",
                "start_time": "2026-03-23T15:00:00",
                "end_time": "2026-03-23T15:10:00",
                "time_cost_seconds": 600.0,
                "error": None,
            },
            "exp_small_running": {
                "exp_id": "exp_small_running",
                "status": "running",
                "output_dir": str(source_dir / "small_running"),
                "model_name": "small-a",
                "benchmark_name": "Ultimatum_Game_Responder",
                "start_time": "2026-03-23T15:11:00",
                "end_time": None,
                "time_cost_seconds": None,
                "error": None,
            },
            "exp_small_pending": {
                "exp_id": "exp_small_pending",
                "status": "pending",
                "output_dir": None,
                "model_name": "small-b",
                "benchmark_name": "Ultimatum_Game_Responder",
                "start_time": None,
                "end_time": None,
                "time_cost_seconds": None,
                "error": None,
            },
            "exp_large_pending": {
                "exp_id": "exp_large_pending",
                "status": "pending",
                "output_dir": None,
                "model_name": deferred_model,
                "benchmark_name": "Ultimatum_Game_Responder",
                "start_time": None,
                "end_time": None,
                "time_cost_seconds": None,
                "error": None,
            },
        },
    )

    from emotion_experiment_engine.resource_recursive_workflow import split_filtered_resume_report

    split_dir = tmp_path / "split"
    plan = split_filtered_resume_report(
        source_report,
        split_output_dir=split_dir,
        shard_series_prefix="memory_experiment_series_gpu",
        shard_labels=["2", "3"],
        deferred_models=[deferred_model],
        carry_forward_series_name="memory_experiment_series_carry_forward",
    )

    carry_payload = json.loads(plan["carry_forward_report"].read_text(encoding="utf-8"))
    shard_payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in plan["shard_reports"]
    ]

    assert set(carry_payload["experiments"].keys()) == {"exp_completed"}
    assert carry_payload["experiments"]["exp_completed"]["status"] == "completed"

    shard_exp_ids = [set(payload["experiments"].keys()) for payload in shard_payloads]
    assert shard_exp_ids[0].isdisjoint(shard_exp_ids[1])
    assert shard_exp_ids[0] | shard_exp_ids[1] == {"exp_small_running", "exp_small_pending"}
    assert all("exp_large_pending" not in payload["experiments"] for payload in shard_payloads)

    running_exp = next(
        payload["experiments"]["exp_small_running"]
        for payload in shard_payloads
        if "exp_small_running" in payload["experiments"]
    )
    assert running_exp["status"] == "pending"
    assert running_exp["output_dir"] is None
    assert running_exp["error"] is None

    manifest_payload = json.loads(plan["manifest_path"].read_text(encoding="utf-8"))
    assert manifest_payload["deferred_models"] == [deferred_model]
    assert manifest_payload["carry_forward_report"] == str(plan["carry_forward_report"])


def test_build_recovery_report_only_includes_non_completed_deferred_models(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    completed_run = _make_run_dir(source_dir, "completed_large")
    deferred_model = "/models/large-4b"
    source_report = source_dir / "memory_experiment_series_20260323_17_memory_experiment_report.json"

    _write_report(
        source_report,
        series_name="memory_experiment_series",
        series_config={"models": [deferred_model, "small-a"], "output_dir": str(source_dir)},
        experiments={
            "exp_large_completed": {
                "exp_id": "exp_large_completed",
                "status": "completed",
                "output_dir": str(completed_run),
                "model_name": deferred_model,
                "benchmark_name": "Trust_Game_Trustee",
                "start_time": "2026-03-23T15:00:00",
                "end_time": "2026-03-23T15:10:00",
                "time_cost_seconds": 600.0,
                "error": None,
            },
            "exp_large_failed": {
                "exp_id": "exp_large_failed",
                "status": "failed",
                "output_dir": str(source_dir / "failed_large"),
                "model_name": deferred_model,
                "benchmark_name": "Ultimatum_Game_Responder",
                "start_time": "2026-03-23T15:11:00",
                "end_time": "2026-03-23T15:12:00",
                "time_cost_seconds": 60.0,
                "error": "oom",
            },
            "exp_small_pending": {
                "exp_id": "exp_small_pending",
                "status": "pending",
                "output_dir": None,
                "model_name": "small-a",
                "benchmark_name": "Ultimatum_Game_Responder",
                "start_time": None,
                "end_time": None,
                "time_cost_seconds": None,
                "error": None,
            },
        },
    )

    from emotion_experiment_engine.resource_recursive_workflow import build_recovery_resume_report

    recovery_report = build_recovery_resume_report(
        source_report,
        output_dir=tmp_path / "recovery",
        series_name="memory_experiment_series_large_recovery",
        deferred_models=[deferred_model],
    )

    recovery_payload = json.loads(recovery_report.read_text(encoding="utf-8"))
    assert set(recovery_payload["experiments"].keys()) == {"exp_large_failed"}
    assert recovery_payload["experiments"]["exp_large_failed"]["status"] == "pending"
    assert recovery_payload["experiments"]["exp_large_failed"]["output_dir"] is None
    assert recovery_payload["experiments"]["exp_large_failed"]["error"] is None


def test_merge_reports_for_resume_preserves_source_series_config_and_current_states(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    source_report = source_dir / "memory_experiment_series_20260324_04_memory_experiment_report.json"
    _write_report(
        source_report,
        series_name="memory_experiment_series",
        series_config={
            "models": ["small-a", "/models/large-3b"],
            "benchmarks": ["Prisoners_Dilemma", "Stag_Hunt"],
            "output_dir": str(source_dir),
        },
        experiments={
            "exp_completed": {
                "exp_id": "exp_completed",
                "status": "completed",
                "output_dir": str(source_dir / "completed_run"),
                "model_name": "small-a",
                "benchmark_name": "Prisoners_Dilemma",
                "start_time": "2026-03-24T04:00:00",
                "end_time": "2026-03-24T04:10:00",
                "time_cost_seconds": 600.0,
                "error": None,
            },
            "exp_failed_deferred": {
                "exp_id": "exp_failed_deferred",
                "status": "failed",
                "output_dir": None,
                "model_name": "/models/large-3b",
                "benchmark_name": "Stag_Hunt",
                "start_time": None,
                "end_time": None,
                "time_cost_seconds": None,
                "error": "CUDA out of memory",
            },
            "exp_pending_small": {
                "exp_id": "exp_pending_small",
                "status": "pending",
                "output_dir": None,
                "model_name": "small-a",
                "benchmark_name": "Stag_Hunt",
                "start_time": None,
                "end_time": None,
                "time_cost_seconds": None,
                "error": None,
            },
        },
    )

    from emotion_experiment_engine.resource_recursive_workflow import (
        merge_reports_for_resume,
        split_filtered_resume_report,
    )

    split_dir = tmp_path / "split"
    artifacts = split_filtered_resume_report(
        source_report,
        split_output_dir=split_dir,
        shard_series_prefix="memory_experiment_series_gpu",
        shard_labels=["2", "3"],
        deferred_models=["/models/large-3b"],
        carry_forward_series_name="carry_forward",
    )

    shard0_payload = json.loads(Path(artifacts["shard_reports"][0]).read_text(encoding="utf-8"))
    shard0_payload["experiments"]["exp_pending_small"]["status"] = "failed"
    shard0_payload["experiments"]["exp_pending_small"]["error"] = "CUDA out of memory while loading model"
    Path(artifacts["shard_reports"][0]).write_text(json.dumps(shard0_payload, indent=2), encoding="utf-8")

    merged_report = merge_reports_for_resume(
        report_paths=[
            artifacts["carry_forward_report"],
            *artifacts["shard_reports"],
        ],
        resume_source_report=source_report,
        merged_output_dir=tmp_path / "state",
        merged_series_name="memory_experiment_series_state",
        extra_config={"workflow_role": "single_gpu_state"},
    )

    merged_payload = json.loads(merged_report.read_text(encoding="utf-8"))
    assert merged_payload["series_config"]["benchmarks"] == ["Prisoners_Dilemma", "Stag_Hunt"]
    assert merged_payload["series_config"]["models"] == ["small-a", "/models/large-3b"]
    assert merged_payload["series_config"]["source_report"] == str(source_report.resolve())
    assert merged_payload["series_config"]["workflow_role"] == "single_gpu_state"
    assert merged_payload["experiments"]["exp_completed"]["status"] == "completed"
    assert merged_payload["experiments"]["exp_pending_small"]["status"] == "failed"
    assert merged_payload["experiments"]["exp_pending_small"]["error"] == "CUDA out of memory while loading model"


def test_bootstrap_source_report_from_config_preserves_completed_seed_and_requeues_incomplete_work(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "seed_source"
    source_dir.mkdir()
    completed_run = _make_run_dir(source_dir, "completed_run")

    seed_report = source_dir / "merged_seed_report.json"
    _write_report(
        seed_report,
        series_name="merged_seed_series",
        series_config={
            "models": ["model_alpha", "model_beta"],
            "benchmarks": [
                {"name": "bench_a", "task_type": "task_x"},
                {"name": "bench_a", "task_type": "task_y"},
            ],
            "output_dir": str(source_dir),
        },
        experiments={
            "bench_a_task_x_model_alpha": {
                "exp_id": "bench_a_task_x_model_alpha",
                "status": "completed",
                "output_dir": str(completed_run),
                "model_name": "model_alpha",
                "benchmark_name": "bench_a_task_x",
                "start_time": "2026-04-02T00:00:00",
                "end_time": "2026-04-02T00:02:00",
                "time_cost_seconds": 120.0,
                "error": None,
            },
            "bench_a_task_x_model_beta": {
                "exp_id": "bench_a_task_x_model_beta",
                "status": "failed",
                "output_dir": str(source_dir / "failed_run"),
                "model_name": "model_beta",
                "benchmark_name": "bench_a_task_x",
                "start_time": "2026-04-02T00:02:00",
                "end_time": "2026-04-02T00:03:00",
                "time_cost_seconds": 60.0,
                "error": "oom",
            },
            "bench_a_task_y_model_alpha": {
                "exp_id": "bench_a_task_y_model_alpha",
                "status": "running",
                "output_dir": str(source_dir / "running_run"),
                "model_name": "model_alpha",
                "benchmark_name": "bench_a_task_y",
                "start_time": "2026-04-02T00:04:00",
                "end_time": None,
                "time_cost_seconds": None,
                "error": None,
            },
        },
    )

    config_path = tmp_path / "updated_config.yaml"
    config_path.write_text(
        """
experiment_name: "seeded_bootstrap"
models:
  - "model_alpha"
  - "model_beta"
benchmarks:
  - name: "bench_a"
    task_type: "task_x"
  - name: "bench_a"
    task_type: "task_y"
output_dir: "__OUTPUT_DIR__"
emotions: []
intensities: [0.0]
loading_config:
  gpu_memory_utilization: 0.8
  enforce_eager: true
  quantization: null
  max_model_len: 1024
  trust_remote_code: true
  dtype: "float16"
  seed: 1
  disable_custom_all_reduce: false
  additional_vllm_kwargs: {}
        """.strip().replace("__OUTPUT_DIR__", str(tmp_path / "results")),
        encoding="utf-8",
    )

    from emotion_experiment_engine.resource_recursive_workflow import _bootstrap_source_report_from_config

    bootstrapped_report = _bootstrap_source_report_from_config(
        config_path=config_path,
        destination=tmp_path / "bootstrapped_source_report.json",
        series_name="seeded_bootstrap",
        gpu_pool=["0", "1", "2", "3"],
        min_resource_gpus=1,
        max_resource_gpus=4,
        seed_report_path=seed_report,
    )

    payload = json.loads(bootstrapped_report.read_text(encoding="utf-8"))
    assert payload["series_config"]["models"] == ["model_alpha", "model_beta"]
    assert payload["series_config"]["benchmarks"] == [
        {"name": "bench_a", "task_type": "task_x"},
        {"name": "bench_a", "task_type": "task_y"},
    ]

    experiments = payload["experiments"]
    assert set(experiments) == {
        "bench_a_task_x_model_alpha",
        "bench_a_task_x_model_beta",
        "bench_a_task_y_model_alpha",
        "bench_a_task_y_model_beta",
    }

    assert experiments["bench_a_task_x_model_alpha"]["status"] == "completed"
    assert experiments["bench_a_task_x_model_alpha"]["output_dir"] == str(completed_run)
    assert experiments["bench_a_task_x_model_beta"]["status"] == "pending"
    assert experiments["bench_a_task_x_model_beta"]["output_dir"] is None
    assert experiments["bench_a_task_x_model_beta"]["error"] is None
    assert experiments["bench_a_task_y_model_alpha"]["status"] == "pending"
    assert experiments["bench_a_task_y_model_alpha"]["output_dir"] is None
    assert experiments["bench_a_task_y_model_alpha"]["start_time"] is None
    assert experiments["bench_a_task_y_model_beta"]["status"] == "pending"


def test_split_filtered_resume_restores_runner_config_from_merged_state_report(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    original_report = source_dir / "original_report.json"
    _write_report(
        original_report,
        series_name="memory_experiment_series",
        series_config={
            "models": ["small-a", "/models/large-3b"],
            "benchmarks": ["Prisoners_Dilemma", "Stag_Hunt"],
            "output_dir": str(source_dir),
        },
        experiments={
            "exp_completed": {
                "exp_id": "exp_completed",
                "status": "completed",
                "output_dir": None,
                "model_name": "small-a",
                "benchmark_name": "Prisoners_Dilemma",
                "start_time": None,
                "end_time": None,
                "time_cost_seconds": None,
                "error": None,
            }
        },
    )

    merged_report = tmp_path / "merged_state_report.json"
    _write_report(
        merged_report,
        series_name="memory_experiment_series_state",
        series_config={
            "source_reports": [str(original_report)],
            "merged_from_series_configs": [
                {
                    "models": ["small-a", "/models/large-3b"],
                    "benchmarks": ["Prisoners_Dilemma", "Stag_Hunt"],
                    "output_dir": str(source_dir),
                    "source_report": str(original_report),
                    "workflow_role": "carry_forward",
                }
            ],
        },
        experiments={
            "exp_pending_small": {
                "exp_id": "exp_pending_small",
                "status": "pending",
                "output_dir": None,
                "model_name": "small-a",
                "benchmark_name": "Stag_Hunt",
                "start_time": None,
                "end_time": None,
                "time_cost_seconds": None,
                "error": None,
            }
        },
    )

    from emotion_experiment_engine.resource_recursive_workflow import split_filtered_resume_report

    artifacts = split_filtered_resume_report(
        merged_report,
        split_output_dir=tmp_path / "split_from_state",
        shard_series_prefix="memory_experiment_series_gpu",
        shard_labels=["2", "3"],
        deferred_models=["/models/large-3b"],
        carry_forward_series_name="carry_forward",
    )

    shard_payload = json.loads(Path(artifacts["shard_reports"][0]).read_text(encoding="utf-8"))
    assert shard_payload["series_config"]["benchmarks"] == ["Prisoners_Dilemma", "Stag_Hunt"]
    assert shard_payload["series_config"]["models"] == ["small-a", "/models/large-3b"]
