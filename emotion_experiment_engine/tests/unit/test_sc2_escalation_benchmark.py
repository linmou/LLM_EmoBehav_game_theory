"""
Responsible file: emotion_experiment_engine/benchmark_component_registry.py
Purpose: New benchmark wiring for SC2 escalation dataset and basic dataset behavior.
"""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any

from emotion_experiment_engine.benchmark_component_registry import (
    create_benchmark_components,
)
from emotion_experiment_engine.data_models import BenchmarkConfig


class _DummyPromptFormat:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def build(self, *args: Any, **kwargs: Any) -> str:  # pragma: no cover - passthrough stub
        self.calls.append((args, kwargs))
        return ""


def test_sc2_escalation_benchmark_registry_and_dataset() -> None:
    cfg = BenchmarkConfig(
        name="sc2_escalation",
        task_type="Escalation_Game",
        data_path=None,
        base_data_dir=None,
        sample_limit=None,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=1.0,
        llm_eval_config=None,
    )

    prompt_format = _DummyPromptFormat()

    prompt_wrapper, answer_wrapper, dataset = create_benchmark_components(
        benchmark_name=cfg.name,
        task_type=cfg.task_type,
        config=cfg,
        prompt_format=prompt_format,
    )

    # Lazy imports to avoid unnecessary heavy imports at module load time
    from emotion_experiment_engine.game_prompt_wrapper import GameBenchmarkPromptWrapper
    from emotion_experiment_engine.datasets.sc2_escalation import (
        SC2EscalationDataset,
    )

    assert callable(prompt_wrapper)
    assert callable(answer_wrapper)
    assert isinstance(dataset, SC2EscalationDataset)

    # The dataset should be wired through the game benchmark prompt wrapper
    bound_call = dataset.prompt_wrapper.func  # type: ignore[attr-defined]
    wrapper_instance = getattr(bound_call, "__self__", None)
    assert isinstance(wrapper_instance, GameBenchmarkPromptWrapper)

    # Basic dataset sanity: should expose items from the SC2 JSON file
    assert len(dataset) >= 10  # mirrors the SC2 dataset structure test

    batch = dataset[0]
    assert set(batch.keys()) == {"item", "prompt", "ground_truth"}
    assert isinstance(batch["prompt"], str) and "Option 1" in batch["prompt"]

    # The first scenario should come from the SC2 escalation dataset file
    data_path = Path("data/sc2/escalation_game.json")
    assert data_path.exists(), "SC2 escalation dataset file must exist"
    assert isinstance(batch["item"].input_text, str)
    assert batch["item"].input_text.strip(), "Scenario description should be non-empty"


def test_sc2_escalation_dataset_uses_custom_evaluate_response_docstring() -> None:
    from emotion_experiment_engine.datasets.sc2_escalation import (
        SC2EscalationDataset,
    )

    doc = SC2EscalationDataset.evaluate_response.__doc__
    assert doc is not None, "evaluate_response should carry the custom override docstring"
    assert (
        "Override to avoid referencing GameTheoryDataset._match_option" in doc
    ), "Expected the custom evaluate_response override to be active"


def test_sc2_escalation_dataset_uses_config_data_path(tmp_path) -> None:
    from emotion_experiment_engine.datasets.sc2_escalation import (
        SC2EscalationDataset,
    )

    custom_path = tmp_path / "custom_sc2.json"
    scenario = {
        "id": 99,
        "description": "Custom scenario for override verification.",
        "you_play_as": "Protoss",
        "players": {
            "player_1": {
                "race": "Protoss",
                "role": "You",
                "economy": "rich",
                "army": "tech",
                "advantage": "map control",
            },
            "player_2": {
                "race": "Terran",
                "role": "Opponent",
                "economy": "balanced",
                "army": "bio",
                "advantage": "timings",
            },
        },
        "behaviour_decisions": {
            "escalate": [
                "Warp in disruptors and blink into the Terran natural.",
                "Strike with double-prong attacks to cripple infrastructure.",
            ],
            "withdraw": [
                "Stabilize with shield batteries across every base.",
                "Cut losses and pivot into a sky Protoss composition.",
            ],
        },
    }
    custom_path.write_text(json.dumps([scenario]))

    cfg = BenchmarkConfig(
        name="sc2_escalation_custom",
        task_type="Escalation_Game",
        data_path=str(custom_path),
        base_data_dir=None,
        sample_limit=None,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=1.0,
        llm_eval_config=None,
    )

    dataset = SC2EscalationDataset(
        config=cfg,
        prompt_wrapper=lambda **_: "",
        max_context_length=None,
        tokenizer=None,
        truncation_strategy="right",
        answer_wrapper=None,
    )

    assert dataset.config.data_path == custom_path
    assert len(dataset) == 1
