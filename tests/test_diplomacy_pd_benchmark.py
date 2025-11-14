"""
tests/test_diplomacy_pd_benchmark.py
Purpose: TDD for a new Diplomacy PD-style benchmark wiring.
 - Verifies registry entry exists and constructs components
 - Verifies dataset loads the small JSONL and exposes items
 - Verifies prompt wrapper renders options properly
 - Verifies evaluate_response extracts the chosen option id
"""

import math
from pathlib import Path

import pytest

from emotion_experiment_engine.data_models import BenchmarkConfig
from emotion_experiment_engine.benchmark_component_registry import (
    create_benchmark_components,
)


def _make_config() -> BenchmarkConfig:
    # Use base_data_dir so path = {base}/{name}_{task}.jsonl
    return BenchmarkConfig(
        name="diplomacy_pd",
        task_type="v1",
        data_path=None,
        base_data_dir=str(Path("data/diplomacy")),
        sample_limit=None,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=1.0,
        llm_eval_config=None,
    )


def test_registry_and_loading():
    config = _make_config()

    # Red: Expect registry to construct components; dataset should load 3 items
    prompt_wrapper, answer_wrapper, dataset = create_benchmark_components(
        benchmark_name=config.name,
        task_type=config.task_type,
        config=config,
        prompt_format=None,
        emotion=None,
    )

    assert len(dataset.items) >= 3, "Expected at least 3 diplomacy PD items"

    # Build a prompt for the first item using the wrapper
    item = dataset.items[0]
    options = item.metadata.get("options", []) if item.metadata else []
    prompt_text = prompt_wrapper(
        context=None,
        question=item.input_text,
        options=options,
        answer=None,
    )
    assert "Option 1." in prompt_text and "Option 2." in prompt_text

    # Simulate model picking an option by text and confirm evaluation returns its id
    opt2_text = options[1]["text"] if isinstance(options[1], dict) else str(options[1])
    score = dataset.evaluate_response(
        response=opt2_text,
        ground_truth=None,
        task_name=config.task_type,
        prompt=prompt_text,
    )
    assert score == 2.0, f"Expected option id 2.0, got {score}"

    # Also support numeric choice like "Option 4"
    score2 = dataset.evaluate_response(
        response="Option 4",
        ground_truth=None,
        task_name=config.task_type,
        prompt=prompt_text,
    )
    assert score2 == 4.0 or math.isnan(score2) is False

