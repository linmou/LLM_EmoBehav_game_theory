"""
Responsible: auto_experiments/task_similarity/emotion_pd_delta_similarity.py
Purpose: Ensure we can source prompts from EmotionExperiment raw_results.json (no dataset build).
"""

from __future__ import annotations

import json
from pathlib import Path


def test_load_prompts_from_raw_results_filters_emotion_and_dedupes(tmp_path: Path) -> None:
    from auto_experiments.task_similarity.emotion_pd_delta_similarity import (
        load_prompts_from_raw_results,
    )

    raw = [
        {"emotion": "anger", "intensity": 1.5, "item_id": 2, "prompt": "P2"},
        {"emotion": "anger", "intensity": 0.5, "item_id": 2, "prompt": "P2"},
        {"emotion": "anger", "intensity": 1.5, "item_id": 1, "prompt": "P1"},
        {"emotion": "happiness", "intensity": 1.5, "item_id": 1, "prompt": "H1"},
        {"emotion": "anger", "intensity": 1.5, "item_id": 3, "prompt": ""},
    ]
    path = tmp_path / "raw_results.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    item_ids, prompts = load_prompts_from_raw_results(path, emotion="anger")
    assert item_ids == [1, 2, 3]
    assert prompts == ["P1", "P2", ""]


def test_load_prompts_from_raw_results_rejects_conflicting_prompts(tmp_path: Path) -> None:
    from auto_experiments.task_similarity.emotion_pd_delta_similarity import (
        load_prompts_from_raw_results,
    )

    raw = [
        {"emotion": "anger", "intensity": 1.5, "item_id": 7, "prompt": "A"},
        {"emotion": "anger", "intensity": 0.5, "item_id": 7, "prompt": "B"},
    ]
    path = tmp_path / "raw_results.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    try:
        load_prompts_from_raw_results(path, emotion="anger")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

