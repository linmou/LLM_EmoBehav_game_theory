"""Responsible file: scripts/watch_prebuild_and_launch_full.py.

Purpose: ensure the watchdog waits for the prebuild launcher to finish and then
starts the full experiment command exactly once.
"""

from pathlib import Path

from scripts.watch_prebuild_and_launch_full import (
    build_full_run_command,
    should_wait_for_prebuild,
)


def test_build_full_run_command_uses_config_and_gpu_ids():
    command = build_full_run_command(
        config_path=Path("config/vlm_mm_game_theory_decision_300.yaml"),
        gpu_ids="0,1,2,3",
    )

    assert command[0] == "python"
    assert command[1:4] == [
        "-m",
        "emotion_experiment_engine.emotion_experiment_series_runner",
        "--config",
    ]
    assert command[4] == "config/vlm_mm_game_theory_decision_300.yaml"


def test_should_wait_for_prebuild_detects_matching_process():
    process_lines = [
        "629071 python scripts/prebuild_vlm_readers.py --config config/vlm_mm_game_theory_decision_300.yaml --gpus 0,1,2,3",
        "777777 python something_else.py",
    ]

    assert should_wait_for_prebuild(
        process_lines=process_lines,
        config_path=Path("config/vlm_mm_game_theory_decision_300.yaml"),
    )


def test_should_wait_for_prebuild_ignores_other_processes():
    process_lines = [
        "777777 python something_else.py",
        "888888 bash -lc echo hi",
    ]

    assert not should_wait_for_prebuild(
        process_lines=process_lines,
        config_path=Path("config/vlm_mm_game_theory_decision_300.yaml"),
    )
