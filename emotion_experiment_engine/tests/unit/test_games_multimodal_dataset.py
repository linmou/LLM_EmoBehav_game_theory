"""Tests for multimodal game-theory dataset plumbing.

Responsible files:
- emotion_experiment_engine/datasets/games_multimodal.py

Purpose:
- Ensure a game-theory dataset variant can carry image paths, load PIL images on demand,
  and expose them via __getitem__/collate_fn for future multimodal inference support.
"""

from __future__ import annotations

from typing import Dict, List

import pytest
from PIL import Image

from emotion_experiment_engine.data_models import BenchmarkConfig


class _DummyChoices:
    option_a: str
    option_b: str

    def __init__(self, option_a: str, option_b: str) -> None:
        self.option_a = option_a
        self.option_b = option_b

    def get_choices(self) -> list[str]:
        return [self.option_a, self.option_b]


class _DummyScenario:
    scenario: str
    description: str
    participants: list[Dict[str, object]]
    behavior_choices: _DummyChoices

    def __init__(
        self,
        scenario: str,
        description: str,
        participants: list[Dict[str, object]],
        behavior_choices: dict,
        **kwargs,
    ) -> None:
        del kwargs
        self.scenario = scenario
        self.description = description
        self.participants = participants
        self.behavior_choices = _DummyChoices(**behavior_choices)

    def get_scenario_info(self) -> dict:
        return {"scenario": self.scenario, "description": self.description}

    def get_behavior_choices(self) -> _DummyChoices:
        return self.behavior_choices

    def find_behavior_from_decision(self, decision: str) -> str:
        if decision == self.behavior_choices.option_a:
            return "cat_a"
        if decision == self.behavior_choices.option_b:
            return "cat_b"
        raise ValueError("Unknown decision")

    @staticmethod
    def example(image_path: str) -> dict:
        return {
            "scenario": "Dummy scenario",
            "description": "Dummy description",
            "participants": [{"name": "You"}, {"name": "Other"}],
            "behavior_choices": {"option_a": "Choose A", "option_b": "Choose B"},
            "payoff_matrix": {},
            "game_name": "DummyGame",
            "image_path": image_path,
        }

    def __str__(self) -> str:
        info = self.get_scenario_info()
        return (
            f"Scenario: {info.get('scenario')}\n"
            f"Description: {info.get('description')}\n"
            f"Participants: {self.participants}\n"
            f"Behavior Choices: {self.behavior_choices.get_choices()}"
        )


def _make_mm_benchmark_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        name="game_theory_mm",
        task_type="DummyGameMM",
        data_path=None,
        base_data_dir=None,
        sample_limit=None,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=1.0,
        llm_eval_config=None,
    )


def test_multimodal_dataset_exposes_images(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """I am starting with a failing test. This is the Red phase."""
    img_path = tmp_path / "img.png"
    Image.new("RGB", (32, 32), "blue").save(img_path)

    from emotion_experiment_engine.datasets.games_multimodal import GameTheoryMultimodalDataset

    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.get_game_config",
        lambda task_type: {
            "game_name": "DummyGame",
            "scenario_class": _DummyScenario,
            "payoff_matrix": {},
            "scenarios": [_DummyScenario.example(str(img_path))],
        },
    )

    cfg = _make_mm_benchmark_config()
    dataset = GameTheoryMultimodalDataset(config=cfg, prompt_wrapper=None, answer_wrapper=None)

    sample = dataset[0]
    assert "images" in sample
    assert isinstance(sample["images"], list)
    assert len(sample["images"]) == 1
    assert isinstance(sample["images"][0], Image.Image)

    item = sample["item"]
    assert item.metadata is not None
    assert item.metadata.get("image_path") == str(img_path)


def test_multimodal_collate_fn_includes_images(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """I am starting with a failing test. This is the Red phase."""
    img_path = tmp_path / "img.png"
    Image.new("RGB", (16, 16), "red").save(img_path)

    from emotion_experiment_engine.datasets.games_multimodal import GameTheoryMultimodalDataset

    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.get_game_config",
        lambda task_type: {
            "game_name": "DummyGame",
            "scenario_class": _DummyScenario,
            "payoff_matrix": {},
            "scenarios": [_DummyScenario.example(str(img_path))],
        },
    )

    cfg = _make_mm_benchmark_config()
    dataset = GameTheoryMultimodalDataset(config=cfg, prompt_wrapper=None, answer_wrapper=None)

    batch = dataset.collate_fn([dataset[0], dataset[0]])
    assert "images" in batch
    assert len(batch["images"]) == 2
    assert all(isinstance(imgs, list) and len(imgs) == 1 for imgs in batch["images"])
    assert all(isinstance(imgs[0], Image.Image) for imgs in batch["images"])


def test_multimodal_dataset_can_load_from_config_data_path(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """I am starting with a failing test. This is the Red phase.

    Ensure we can point `BenchmarkConfig.data_path` at an arbitrary JSON file
    (no need to edit `games/game_configs.py`) while still using a scenario_class.
    """
    img_path = tmp_path / "img.png"
    Image.new("RGB", (8, 8), "green").save(img_path)

    dataset_path = tmp_path / "mm_game.json"
    dataset_path.write_text(
        __import__("json").dumps([_DummyScenario.example(str(img_path))]),
        encoding="utf-8",
    )

    from emotion_experiment_engine.datasets.games_multimodal import GameTheoryMultimodalDataset

    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.get_game_config",
        lambda task_type: {
            "game_name": "DummyGame",
            "scenario_class": _DummyScenario,
            "payoff_matrix": {},
            "data_path": "this_would_normally_be_used.json",
        },
    )

    cfg = _make_mm_benchmark_config()
    cfg.data_path = dataset_path

    ds = GameTheoryMultimodalDataset(config=cfg, prompt_wrapper=None, answer_wrapper=None)
    assert len(ds) == 1
    sample = ds[0]
    assert sample["item"].metadata is not None
    assert sample["item"].metadata["image_path"] == str(img_path)
