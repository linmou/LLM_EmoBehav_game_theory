"""Tests for Diplomacy PD benchmark dataset and prompt wrapper."""

from pathlib import Path
from functools import partial

import pytest

from emotion_experiment_engine.data_models import BenchmarkConfig


class DummyPromptFormat:
    def build(self, text, user_messages, enable_thinking=False):
        # Mirror the simple plaintext behavior used in production when no formatter exists.
        return text


def _make_config():
    return BenchmarkConfig(
        name="diplomacy_pd",
        task_type="v1b",
        data_path=Path("data/diplomacy/diplomacy_pd_escalation_20251117.jsonl"),
        base_data_dir="data/diplomacy",
        sample_limit=None,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=1.0,
        llm_eval_config=None,
    )


def test_prompt_wrapper_renders_header_and_options():
    from emotion_experiment_engine.diplomacy_prompts import DiplomacyOptionsPromptWrapper

    wrapper = DiplomacyOptionsPromptWrapper(DummyPromptFormat())
    prompt = wrapper(
        context="Your Country: Italy\nGame: Standard Diplomacy",
        question="Scenario text",
        user_messages=["Respond"],
        enable_thinking=False,
        augmentation_config=None,
        answer=None,
        emotion=None,
        options=[
            {"id": 1, "text": "Maintain the stalemate line"},
            {"id": 2, "text": "Advance into Tyrolia"},
        ],
    )

    assert "Your Country: Italy" in prompt
    assert "Game: Standard Diplomacy" in prompt
    assert "Option 1. Maintain the stalemate line" in prompt
    assert "Respond with the option text." in prompt


def test_dataset_loads_items_and_scores_responses():
    from emotion_experiment_engine.diplomacy_prompts import DiplomacyOptionsPromptWrapper
    from emotion_experiment_engine.datasets.diplomacy_gradient import DiplomacyGradientDataset

    wrapper = DiplomacyOptionsPromptWrapper(DummyPromptFormat())
    prompt_wrapper = partial(
        wrapper.__call__,
        user_messages=["Please decide"],
        enable_thinking=False,
        augmentation_config=None,
        emotion=None,
    )

    dataset = DiplomacyGradientDataset(_make_config(), prompt_wrapper)

    assert len(dataset) >= 3

    first_item = dataset.items[0]
    assert "Your Country:" in (first_item.context or "")
    assert first_item.metadata and len(first_item.metadata.get("options", [])) == 2

    batch_entry = dataset[0]
    prompt = batch_entry["prompt"]
    score_numeric = dataset.evaluate_response("Option 2", None, "v1b", prompt)
    assert score_numeric == pytest.approx(2.0)

    # Option-text matching should also work irrespective of case differences.
    option_text = first_item.metadata["options"][0]["text"].upper()
    score_text = dataset.evaluate_response(option_text, None, "v1b", prompt)
    assert score_text == pytest.approx(1.0)


def test_dataset_prefers_description_and_gradient_options():
    """
    Validate the escalation Diplomacy set keeps description text, prefers behavior_choices,
    and carries the whose_option label when constructing items.
    """
    from emotion_experiment_engine.datasets.diplomacy_gradient import DiplomacyGradientDataset
    from emotion_experiment_engine.diplomacy_prompts import DiplomacyOptionsPromptWrapper

    wrapper = DiplomacyOptionsPromptWrapper(DummyPromptFormat())
    prompt_wrapper = partial(
        wrapper.__call__,
        user_messages=["Please decide"],
        enable_thinking=False,
        augmentation_config=None,
        emotion=None,
    )

    dataset = DiplomacyGradientDataset(_make_config(), prompt_wrapper)

    assert len(dataset) >= 10
    first_item = dataset.items[0]

    assert first_item.input_text.startswith("In the spring of 1901")
    assert first_item.metadata["whose_option"] == "RUSSIA"
    assert "Diplomacy_RUSSIA_vs_TURKEY_BLA_Spring1901_Orders" in (first_item.context or "")
    option_texts = [opt["text"].lower() for opt in first_item.metadata["options"]]
    assert len(option_texts) == 2  # behavior_choices: withdraw/escalate
    assert any("withdraw" in opt or "reduce naval activity" in opt for opt in option_texts)
    assert any("escalate" in opt or "increase naval presence" in opt for opt in option_texts)
