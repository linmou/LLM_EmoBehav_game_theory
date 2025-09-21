# Tests for emotion_experiment_engine/datasets/games.py
"""Unit tests for game theory dataset evaluation logic."""

import pytest

from emotion_experiment_engine.data_models import BenchmarkConfig, BenchmarkItem
from emotion_experiment_engine.datasets.games import GameTheoryDataset


@pytest.fixture()
def benchmark_config() -> BenchmarkConfig:
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


@pytest.fixture()
def dataset(monkeypatch, benchmark_config) -> GameTheoryDataset:
    monkeypatch.setattr(
        GameTheoryDataset,
        "_load_and_parse_data",
        lambda self: [
            BenchmarkItem(
                id="pd-1",
                input_text="Prisoners dilemma event",
                context=None,
                ground_truth=None,
                metadata={
                    "options": [
                        {"id": 1, "text": "Cooperate"},
                        {"id": 2, "text": "Defect"},
                    ]
                },
            )
        ],
    )

    return GameTheoryDataset(
        config=benchmark_config,
        prompt_wrapper=None,
        answer_wrapper=None,
    )


def test_evaluate_response_regex_only(dataset: GameTheoryDataset):
    prompt = (
        "Scenario: Prisoners dilemma\n"
        "Option 1. Cooperate\n"
        "Option 2. Defect\n"
    )
    response = '{"analysis": "...", "decision": "defect"}'

    choice = dataset.evaluate_response(
        response=response,
        ground_truth=None,
        task_name="Prisoners_Dilemma",
        prompt=prompt,
    )

    assert choice == pytest.approx(2.0)


def test_evaluate_response_llm_fallback(monkeypatch, dataset: GameTheoryDataset):
    captured = {}

    def _fake_oai(prompt, client=None, model=None, response_format=None):
        captured["response_format"] = response_format
        return dataset._ExtractionSchema(option_id=1, rationale="", decision="")

    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.oai_response",
        _fake_oai,
    )

    prompt = (
        "Scenario: Prisoners dilemma\n"
        "Option 1. Cooperate\n"
        "Option 2. Defect\n"
    )
    response = "No decision present"

    choice = dataset.evaluate_response(
        response=response,
        ground_truth=None,
        task_name="Prisoners_Dilemma",
        prompt=prompt,
    )

    assert choice == pytest.approx(1.0)
    assert captured["response_format"] is dataset._ExtractionSchema
