"""Tests for `result_analysis/visualize_emotion_dashboard.py`.

Responsible file: result_analysis/visualize_emotion_dashboard.py
Purpose: ensure the dashboard script can (1) discover run dirs, (2) parse
summary_behavior_ratio.csv, (3) compute JS divergence + majority deltas, and
(4) derive invalid rates from raw_results.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _write_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "ModelX_game_theory_decision_Prisoners_Dilemma_20250101_000000"
    run_dir.mkdir(parents=True)

    (run_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                "model_path": "/models/ModelX",
                "emotions": ["anger"],
                "intensities": [1.0],
                "benchmark": {"name": "game_theory_decision", "task_type": "Prisoners_Dilemma"},
            }
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {"emotion": "neutral", "intensity": 0.0, "behavior_label": "cooperate", "ratio": 0.75},
            {"emotion": "neutral", "intensity": 0.0, "behavior_label": "defect", "ratio": 0.25},
            {"emotion": "anger", "intensity": 1.0, "behavior_label": "cooperate", "ratio": 0.25},
            {"emotion": "anger", "intensity": 1.0, "behavior_label": "defect", "ratio": 0.75},
        ]
    ).to_csv(run_dir / "summary_behavior_ratio.csv", index=False)

    rows = [
        {
            "emotion": "neutral",
            "intensity": 0.0,
            "item_id": 0,
            "task_name": "Prisoners_Dilemma",
            "prompt": "",
            "response": "",
            "ground_truth": None,
            "score": 1.0,
            "repeat_id": 0,
            "metadata": {
                "item_metadata": {"options": [{"id": 1, "behavior": "cooperate"}, {"id": 2, "behavior": "defect"}]}
            },
            "error": None,
        },
        {
            "emotion": "anger",
            "intensity": 1.0,
            "item_id": 0,
            "task_name": "Prisoners_Dilemma",
            "prompt": "",
            "response": "",
            "ground_truth": None,
            "score": None,
            "repeat_id": 0,
            "metadata": {
                "item_metadata": {"options": [{"id": 1, "behavior": "cooperate"}, {"id": 2, "behavior": "defect"}]}
            },
            "error": "parse failed",
        },
    ]
    (run_dir / "raw_results.json").write_text(json.dumps(rows), encoding="utf-8")
    return run_dir


def test_discover_runs(tmp_path: Path) -> None:
    from result_analysis.visualize_emotion_dashboard import discover_run_dirs

    run_dir = _write_run(tmp_path)
    found = discover_run_dirs(tmp_path)
    assert run_dir in found


def test_compute_js_and_majority_delta(tmp_path: Path) -> None:
    from result_analysis.visualize_emotion_dashboard import compute_js_divergence, compute_majority_behavior_effects

    run_dir = _write_run(tmp_path)
    df = pd.read_csv(run_dir / "summary_behavior_ratio.csv")

    js = compute_js_divergence(df)
    assert set(js.columns) >= {"emotion", "intensity", "js_divergence"}
    assert (js[js["emotion"] == "neutral"]["js_divergence"] == 0.0).all()

    maj = compute_majority_behavior_effects(df)
    # Neutral majority is cooperate (0.75)
    row = maj[(maj["emotion"] == "anger") & (maj["intensity"] == 1.0)].iloc[0]
    assert row["neutral_majority_behavior"] == "cooperate"
    assert row["delta_majority_ratio"] == pytest.approx(0.25 - 0.75)


def test_invalid_rate_from_raw_results(tmp_path: Path) -> None:
    from result_analysis.visualize_emotion_dashboard import compute_invalid_rates_from_raw_results

    run_dir = _write_run(tmp_path)
    invalid = compute_invalid_rates_from_raw_results(run_dir / "raw_results.json")
    # anger has one invalid row out of 1
    row = invalid[(invalid["emotion"] == "anger") & (invalid["intensity"] == 1.0)].iloc[0]
    assert row["invalid_count"] == 1
    assert row["total_count"] == 1
    assert row["invalid_rate"] == 1.0
