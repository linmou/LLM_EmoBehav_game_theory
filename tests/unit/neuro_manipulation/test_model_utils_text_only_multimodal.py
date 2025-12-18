# Tests for neuro_manipulation/model_utils.py text-only flow with multimodal models
from unittest.mock import MagicMock
import sys

import pytest


def test_text_only_pipeline_passes_image_processor(monkeypatch):
    # Stub vllm to avoid heavy import during unit test
    sys.modules["vllm"] = MagicMock()

    # Minimal config to drive load_emotion_readers
    config = {
        "emotions": ["anger", "happiness"],
        "data_dir": "dummy",
        "model_name_or_path": "Qwen/Qwen2.5-VL-3B-Instruct",
        "rep_token": -1,
        "hidden_layers": [-1],
        "n_difference": 1,
        "direction_method": "pca",
        "multimodal_intent": False,
        "rebuild": True,
    }

    # Force feasibility to text_only
    monkeypatch.setattr(
        "neuro_manipulation.utils.validate_multimodal_experiment_feasibility",
        lambda cfg: {
            "feasible": True,
            "mode": "text_only",
            "reasons": ["Text-only experiment feasible"],
            "data_status": {},
            "model_is_multimodal": True,
        },
    )

    # Stub dataset and reader builder
    dummy_dataset = {"anger": {"train": {"data": ["a"], "labels": [1]}, "test": {"data": ["a"], "labels": [1]}},
                     "happiness": {"train": {"data": ["b"], "labels": [1]}, "test": {"data": ["b"], "labels": [1]}}}
    monkeypatch.setattr(
        "neuro_manipulation.model_utils.primary_emotions_concept_dataset",
        lambda *a, **k: dummy_dataset,
    )

    calls = {}

    def fake_pipeline(task, model=None, tokenizer=None, image_processor=None):
        calls["task"] = task
        calls["model"] = model
        calls["tokenizer"] = tokenizer
        calls["image_processor"] = image_processor
        return MagicMock()

    monkeypatch.setattr("neuro_manipulation.model_utils.pipeline", fake_pipeline)

    monkeypatch.setattr(
        "neuro_manipulation.model_utils.all_emotion_rep_reader",
        lambda *a, **k: {"ok": True},
    )

    from neuro_manipulation import model_utils

    processor = object()
    model_utils.load_emotion_readers(
        config=config,
        model="m",
        tokenizer="t",
        hidden_layers=[-1],
        processor=processor,
        enable_thinking=False,
    )

    assert calls["task"] == "rep-reading"
    assert calls["image_processor"] is processor
