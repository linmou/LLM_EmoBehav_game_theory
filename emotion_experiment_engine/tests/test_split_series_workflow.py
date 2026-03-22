# Tests for split-series evaluation and merge workflow.
# Responsible files: emotion_experiment_engine/split_series_workflow.py and scripts/run_vlm_mm_game_theory_300_split_sweeps.sh
# Purpose: ensure split sweep reports can be merged into one result folder/report and evaluator tmux watchers are launched correctly.

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

    from emotion_experiment_engine.split_series_workflow import merge_series_reports

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
        from emotion_experiment_engine.split_series_workflow import launch_eval_watchers_tmux

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
