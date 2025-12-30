"""
Unit tests: backfill chosen_behavior into existing detailed_results.csv.

Covers: emotion_experiment_engine/scripts/post_process_scripts/add_chosen_behavior.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _write_run_dir(run_dir: Path, *, score: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = [
        {
            "emotion": "anger",
            "intensity": 0.1,
            "item_id": "pd-1",
            "task_name": "Prisoners_Dilemma",
            "prompt": "",
            "response": "",
            "ground_truth": None,
            "score": score,
            "repeat_id": 0,
            "metadata": {
                "benchmark": "game_theory",
                "item_metadata": {
                    "options": [
                        {"id": 1, "text": "Cooperate", "behavior": "cooperate"},
                        {"id": 2, "text": "Defect", "behavior": "defect"},
                    ]
                },
            },
            "error": None,
        }
    ]
    (run_dir / "raw_results.json").write_text(json.dumps(raw_rows), encoding="utf-8")

    detailed = pd.DataFrame(
        [
            {
                "emotion": "anger",
                "intensity": 0.1,
                "item_id": "pd-1",
                "task_name": "Prisoners_Dilemma",
                "response": "",
                "ground_truth": "None",
                "score": score,
                "benchmark": "game_theory",
                "repeat_id": 0,
                "error": None,
            }
        ]
    )
    detailed.to_csv(run_dir / "detailed_results.csv", index=False)


def test_script_adds_column_recursively(tmp_path: Path) -> None:
    # Responsible for: verify recursion + correct mapping from option_id -> behavior.
    from emotion_experiment_engine.scripts.post_process_scripts.add_chosen_behavior import (
        add_chosen_behavior_under_root,
    )

    _write_run_dir(tmp_path / "a" / "run1", score=1.0)
    _write_run_dir(tmp_path / "b" / "run2", score=2.0)

    updated = add_chosen_behavior_under_root(tmp_path, strict=True)
    assert updated == 2

    df1 = pd.read_csv(tmp_path / "a" / "run1" / "detailed_results.csv")
    assert df1["chosen_behavior"].tolist() == ["cooperate"]

    df2 = pd.read_csv(tmp_path / "b" / "run2" / "detailed_results.csv")
    assert df2["chosen_behavior"].tolist() == ["defect"]


def test_script_supports_parallel_jobs(tmp_path: Path) -> None:
    # Responsible for: parallel execution should still update all runs correctly.
    from emotion_experiment_engine.scripts.post_process_scripts.add_chosen_behavior import (
        add_chosen_behavior_under_root,
    )

    _write_run_dir(tmp_path / "a" / "run1", score=1.0)
    _write_run_dir(tmp_path / "b" / "run2", score=2.0)

    updated = add_chosen_behavior_under_root(tmp_path, strict=True, jobs=2)
    assert updated == 2

    df1 = pd.read_csv(tmp_path / "a" / "run1" / "detailed_results.csv")
    assert df1["chosen_behavior"].tolist() == ["cooperate"]

    df2 = pd.read_csv(tmp_path / "b" / "run2" / "detailed_results.csv")
    assert df2["chosen_behavior"].tolist() == ["defect"]


def test_script_can_resume_by_skipping_finished_files(tmp_path: Path) -> None:
    # Responsible for: resume runs should skip already-filled CSVs.
    from emotion_experiment_engine.scripts.post_process_scripts.add_chosen_behavior import (
        add_chosen_behavior_under_root,
    )

    _write_run_dir(tmp_path / "run", score=1.0)
    assert add_chosen_behavior_under_root(tmp_path, strict=True) == 1

    # Second pass: nothing left to fill, should do no work and not fail.
    assert add_chosen_behavior_under_root(tmp_path, strict=True) == 0


def test_script_strict_rejects_missing_option_id(tmp_path: Path) -> None:
    # Responsible for: verify strict assertion catches inconsistent score/options.
    from emotion_experiment_engine.scripts.post_process_scripts.add_chosen_behavior import (
        add_chosen_behavior_under_root,
    )

    _write_run_dir(tmp_path / "run", score=3.0)
    with pytest.raises(ValueError, match="option_id"):
        add_chosen_behavior_under_root(tmp_path, strict=True)


def test_script_missing_option_id_error_includes_context(tmp_path: Path) -> None:
    # Responsible for: error should include paths + available option ids for debugging.
    from emotion_experiment_engine.scripts.post_process_scripts.add_chosen_behavior import (
        add_chosen_behavior_under_root,
    )

    run_dir = tmp_path / "run"
    _write_run_dir(run_dir, score=3.0)
    with pytest.raises(ValueError) as excinfo:
        add_chosen_behavior_under_root(tmp_path, strict=True)

    msg = str(excinfo.value)
    assert "detailed_results.csv" in msg
    assert "raw_results.json" in msg
    assert "option_id=3" in msg
    assert "available_option_ids=[1, 2]" in msg


def test_script_strict_rejects_mismatched_existing_value(tmp_path: Path) -> None:
    # Responsible for: strict mode must catch existing chosen_behavior that disagrees.
    from emotion_experiment_engine.scripts.post_process_scripts.add_chosen_behavior import (
        add_chosen_behavior_under_root,
    )

    run_dir = tmp_path / "run"
    _write_run_dir(run_dir, score=1.0)
    df = pd.read_csv(run_dir / "detailed_results.csv")
    df["chosen_behavior"] = "defect"
    df.to_csv(run_dir / "detailed_results.csv", index=False)

    with pytest.raises(ValueError, match="chosen_behavior mismatch"):
        add_chosen_behavior_under_root(
            tmp_path, strict=True, overwrite=False, skip_finished=False
        )


def test_script_handles_negative_score_as_unknown(tmp_path: Path) -> None:
    # Responsible for: skip option_id <= 0 (e.g., -1) without failing.
    from emotion_experiment_engine.scripts.post_process_scripts.add_chosen_behavior import (
        add_chosen_behavior_under_root,
    )

    run_dir = tmp_path / "run"
    _write_run_dir(run_dir, score=-1.0)
    updated = add_chosen_behavior_under_root(tmp_path, strict=True)
    assert updated == 1

    df = pd.read_csv(run_dir / "detailed_results.csv")
    assert df["chosen_behavior"].isna().all()
