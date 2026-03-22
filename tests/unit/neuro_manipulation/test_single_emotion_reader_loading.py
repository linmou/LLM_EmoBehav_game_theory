"""Responsible file: neuro_manipulation/model_utils.py.

Purpose: ensure an anger-only experiment still builds a single anger reader
using other available dataset emotions as contrast examples.
"""

import io

from neuro_manipulation import model_utils


def test_load_emotion_readers_uses_available_dataset_emotions_for_single_target(monkeypatch):
    dataset_call = {}
    reader_call = {}

    monkeypatch.setattr(
        "neuro_manipulation.utils.validate_multimodal_experiment_feasibility",
        lambda config: {
            "feasible": True,
            "mode": "multimodal",
            "reasons": ["ok"],
        },
    )
    monkeypatch.setattr(
        model_utils,
        "pipeline",
        lambda *args, **kwargs: "fake-pipeline",
    )
    monkeypatch.setattr(
        model_utils,
        "detect_emotion_data_type",
        lambda data_dir, emotions=None: {
            "available_emotions": ["anger", "happiness", "fear"],
        },
    )

    def fake_primary_dataset(*args, **kwargs):
        dataset_call["emotions"] = kwargs["emotions"]
        return {
            "anger": {
                "train": {"data": ["sample"], "labels": [[True, False]]},
                "test": {"data": ["sample"], "labels": [[True, False]]},
            }
        }

    def fake_all_emotion_rep_reader(
        data,
        emotions,
        rep_reading_pipeline,
        hidden_layers,
        rep_token,
        n_difference,
        direction_method,
        save_path=None,
        read_args=None,
    ):
        reader_call["emotions"] = emotions
        return {"anger": "reader", "layer_acc": {}, "args": read_args}

    monkeypatch.setattr(
        model_utils,
        "primary_emotions_concept_dataset",
        fake_primary_dataset,
    )
    monkeypatch.setattr(
        model_utils,
        "all_emotion_rep_reader",
        fake_all_emotion_rep_reader,
    )

    config = {
        "emotions": ["anger"],
        "data_dir": "unused",
        "model_name_or_path": "fake/model",
        "rep_token": -1,
        "n_difference": 1,
        "direction_method": "pca",
        "multimodal_intent": True,
        "rebuild": True,
    }

    readers = model_utils.load_emotion_readers(
        config=config,
        model="fake-model",
        tokenizer="fake-tokenizer",
        hidden_layers=[-1],
        processor="fake-processor",
    )

    assert dataset_call["emotions"] == ["anger", "happiness", "fear"]
    assert reader_call["emotions"] == ["anger"]
    assert readers["anger"] == "reader"


def test_load_emotion_readers_rebuilds_when_cached_reader_is_missing_target(monkeypatch):
    dataset_called = {"value": False}

    monkeypatch.setattr(
        "neuro_manipulation.utils.validate_multimodal_experiment_feasibility",
        lambda config: {
            "feasible": True,
            "mode": "multimodal",
            "reasons": ["ok"],
        },
    )
    monkeypatch.setattr(model_utils, "pipeline", lambda *args, **kwargs: "fake-pipeline")
    monkeypatch.setattr(
        model_utils,
        "detect_emotion_data_type",
        lambda data_dir, emotions=None: {
            "available_emotions": ["anger", "happiness"],
        },
    )
    monkeypatch.setattr(
        "builtins.open",
        lambda *args, **kwargs: io.BytesIO(),
    )
    monkeypatch.setattr(
        model_utils.pickle,
        "load",
        lambda fh: {
            "args": {
                "emotions": ["anger"],
                "data_dir": "unused",
                "model_name_or_path": "fake/model",
                "rep_token": -1,
                "hidden_layers": [-1],
                "n_difference": 1,
                "direction_method": "pca",
                "experiment_mode": "multimodal",
                "multimodal_intent": True,
                "emotion_data_seed": 0,
            },
            "layer_acc": {},
        },
    )

    def fake_primary_dataset(*args, **kwargs):
        dataset_called["value"] = True
        return {
            "anger": {
                "train": {"data": ["sample"], "labels": [[True, False]]},
                "test": {"data": ["sample"], "labels": [[True, False]]},
            }
        }

    monkeypatch.setattr(model_utils, "primary_emotions_concept_dataset", fake_primary_dataset)
    monkeypatch.setattr(
        model_utils,
        "all_emotion_rep_reader",
        lambda *args, **kwargs: {"anger": "reader", "layer_acc": {}, "args": kwargs.get("read_args")},
    )

    config = {
        "emotions": ["anger"],
        "data_dir": "unused",
        "model_name_or_path": "fake/model",
        "rep_token": -1,
        "n_difference": 1,
        "direction_method": "pca",
        "multimodal_intent": True,
        "rebuild": False,
    }

    readers = model_utils.load_emotion_readers(
        config=config,
        model="fake-model",
        tokenizer="fake-tokenizer",
        hidden_layers=[-1],
        processor="fake-processor",
    )

    assert dataset_called["value"] is True
    assert readers["anger"] == "reader"
