#!/usr/bin/env python3
"""
Responsible file: emotion_experiment_engine/datasets/games.py
Purpose: Regression test for option-shuffle: behavior ratios must not cache
         option metadata by item_id across emotions/intensities.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from emotion_experiment_engine.data_models import BenchmarkConfig, BenchmarkItem, ResultRecord
from emotion_experiment_engine.datasets.games import GameTheoryDataset


def _stub_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        name="game_theory",
        task_type="Prisoners_Dilemma",
        data_path=None,
        base_data_dir=None,
        sample_limit=None,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=1.0,
        llm_eval_config=None,
    )


def test_behavior_choice_ratio_does_not_cache_options_across_emotions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.GameTheoryDataset._load_and_parse_data",
        lambda self: [
            BenchmarkItem(
                id="item-1",
                input_text="",
                context=None,
                ground_truth=None,
                metadata={"options": [{"id": 1, "text": "A", "behavior": "cooperate"}]},
            )
        ],
    )
    dataset = GameTheoryDataset(config=_stub_config(), prompt_wrapper=None, answer_wrapper=None)

    meta_anger: Dict[str, Any] = {
        "item_metadata": {
            "options": [
                {"id": 1, "text": "A", "behavior": "cooperate"},
                {"id": 2, "text": "B", "behavior": "defect"},
            ]
        }
    }
    meta_neutral: Dict[str, Any] = {
        "item_metadata": {
            "options": [
                {"id": 1, "text": "A", "behavior": "defect"},
                {"id": 2, "text": "B", "behavior": "cooperate"},
            ]
        }
    }

    records: List[ResultRecord] = [
        ResultRecord(
            emotion="anger",
            intensity=1.0,
            item_id="item-1",
            task_name="Prisoners_Dilemma",
            prompt="",
            response="",
            ground_truth=None,
            score=1.0,  # cooperate in anger metadata
            repeat_id=0,
            metadata=meta_anger,
        ),
        ResultRecord(
            emotion="neutral",
            intensity=0.0,
            item_id="item-1",
            task_name="Prisoners_Dilemma",
            prompt="",
            response="",
            ground_truth=None,
            score=1.0,  # defect in neutral metadata
            repeat_id=0,
            metadata=meta_neutral,
        ),
    ]

    metrics = dataset.compute_split_metrics(records)
    overall = metrics["behavior_choice_ratio"]["overall"]
    by_key = {(r["emotion"], float(r["intensity"]), r["behavior_label"]): r["ratio"] for r in overall}

    assert by_key[("anger", 1.0, "cooperate")] == pytest.approx(1.0)
    assert by_key[("neutral", 0.0, "defect")] == pytest.approx(1.0)

