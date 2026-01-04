#!/usr/bin/env python3
"""
Responsible file: emotion_experiment_engine/datasets/games.py
Purpose: Ensure GameTheoryDataset can optionally shuffle choice order and
         renumber option ids deterministically via config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

from emotion_experiment_engine.data_models import BenchmarkConfig
from emotion_experiment_engine.datasets.games import GameTheoryDataset


@dataclass
class _StubChoices:
    choices: List[str]

    def get_choices(self) -> List[str]:
        return self.choices


class _StubScenario:
    model_fields: Dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        self._choices = kwargs.get("choices") or ["Cooperate", "Defect"]

    def get_behavior_choices(self) -> _StubChoices:
        return _StubChoices(list(self._choices))

    def find_behavior_from_decision(self, decision: str) -> str:
        return ""

    def __str__(self) -> str:
        return "stub scenario"


def _base_benchmark_config() -> BenchmarkConfig:
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


def test_game_options_shuffle_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.get_game_config",
        lambda task_type: {
            "scenario_class": _StubScenario,
            "payoff_matrix": {},
            "shuffle_options": True,
            "behavior_ratio": 1,  # random.Random(1) swaps 2-element lists
        },
    )
    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.GameTheoryDataset._load_raw_scenarios",
        lambda self: [{"id": "x", "choices": ["A", "B"]}],
    )

    dataset = GameTheoryDataset(
        config=_base_benchmark_config(),
        prompt_wrapper=None,
        answer_wrapper=None,
    )
    item = dataset.items[0]
    options = item.metadata["options"]

    assert [o["text"] for o in options] == ["B", "A"]
    assert [o["id"] for o in options] == [1, 2]
    assert item.metadata.get("behavior_ratio_used") == 1


def test_game_options_shuffle_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.get_game_config",
        lambda task_type: {
            "scenario_class": _StubScenario,
            "payoff_matrix": {},
            "shuffle_options": False,
        },
    )
    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.GameTheoryDataset._load_raw_scenarios",
        lambda self: [{"id": "x", "choices": ["A", "B"]}],
    )

    dataset = GameTheoryDataset(
        config=_base_benchmark_config(),
        prompt_wrapper=None,
        answer_wrapper=None,
    )
    item = dataset.items[0]
    options = item.metadata["options"]

    assert [o["text"] for o in options] == ["A", "B"]
    assert [o["id"] for o in options] == [1, 2]
