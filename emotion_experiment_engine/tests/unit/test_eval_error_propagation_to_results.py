"""
Unit test: ensure evaluation errors from dataset.evaluate_batch are propagated
into EmotionExperiment results and persisted by _save_results.

Covers: emotion_experiment_engine/experiment.py
- _post_process_batch should read dataset._last_eval_errors and set ResultRecord.error
- _save_results should include an 'error' column in detailed_results.csv/DataFrame
"""

from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch
import importlib
import sys

import math
import pandas as pd

from emotion_experiment_engine.data_models import BenchmarkItem
from emotion_experiment_engine.tests.test_utils import (
    create_mock_experiment_config,
)


class DummyDataset:
    """Minimal dataset stub exposing evaluate_batch and collate_fn-like expectations."""

    def __init__(self) -> None:
        self._last_eval_errors: List[Any] = []

    def evaluate_batch(
        self,
        responses: List[str],
        ground_truths: List[Any],
        task_names: List[str],
        prompts: List[str],
    ) -> List[float]:
        # Simulate one error and one success
        self._last_eval_errors = ["mock eval error", None]
        # Return NaN for the errored item, and 1.0 for the successful item
        return [float("nan"), 1.0]


def test_eval_error_is_propagated_to_results(tmp_path: Path):
    # Arrange minimal experiment
    # Stub out the repe_eng_config import to avoid pydantic dependency chain
    fake_repe_module = type(sys)("neuro_manipulation.configs.experiment_config")
    def _fake_get_repe_eng_config(*args, **kwargs):
        return {}
    fake_repe_module.get_repe_eng_config = _fake_get_repe_eng_config  # type: ignore[attr-defined]
    sys.modules["neuro_manipulation.configs.experiment_config"] = fake_repe_module

    # Also stub the utils module to avoid importing torch and heavy deps
    fake_utils_module = type(sys)("neuro_manipulation.utils")
    def _fake_load_tokenizer_only(*args, **kwargs):
        return (MagicMock(), None)
    fake_utils_module.load_tokenizer_only = _fake_load_tokenizer_only  # type: ignore[attr-defined]
    sys.modules["neuro_manipulation.utils"] = fake_utils_module

    # Stub other heavy neuro_manipulation modules to avoid torch deps on import
    fake_mld_module = type(sys)("neuro_manipulation.model_layer_detector")
    class _FakeMLD:
        @staticmethod
        def num_layers(model=None):
            return 4
    fake_mld_module.ModelLayerDetector = _FakeMLD  # type: ignore[attr-defined]
    sys.modules["neuro_manipulation.model_layer_detector"] = fake_mld_module

    fake_mu_module = type(sys)("neuro_manipulation.model_utils")
    def _fake_setup_model_and_tokenizer(*args, **kwargs):
        return (MagicMock(), MagicMock(), "chat", MagicMock())
    def _fake_load_emotion_readers(*args, **kwargs):
        return {"anger": MagicMock(), "neutral": MagicMock()}
    fake_mu_module.setup_model_and_tokenizer = _fake_setup_model_and_tokenizer  # type: ignore[attr-defined]
    fake_mu_module.load_emotion_readers = _fake_load_emotion_readers  # type: ignore[attr-defined]
    sys.modules["neuro_manipulation.model_utils"] = fake_mu_module

    fake_pipelines_module = type(sys)("neuro_manipulation.repe.pipelines")
    def _fake_get_pipeline(*args, **kwargs):
        return MagicMock()
    fake_pipelines_module.get_pipeline = _fake_get_pipeline  # type: ignore[attr-defined]
    sys.modules["neuro_manipulation.repe.pipelines"] = fake_pipelines_module

    # Stub the benchmark registry to avoid importing real datasets (which require torch)
    fake_registry_module = type(sys)("emotion_experiment_engine.benchmark_component_registry")
    def _fake_create_benchmark_components(**kwargs):
        # Return (prompt_wrapper, answer_wrapper, dataset)
        return (MagicMock(), MagicMock(), [1, 2])
    fake_registry_module.create_benchmark_components = _fake_create_benchmark_components  # type: ignore[attr-defined]
    sys.modules["emotion_experiment_engine.benchmark_component_registry"] = fake_registry_module

    # Import after stubbing to avoid import-time errors
    experiment_module = importlib.import_module("emotion_experiment_engine.experiment")
    EmotionExperiment = experiment_module.EmotionExperiment

    config = create_mock_experiment_config("passkey", 2)
    config.output_dir = str(tmp_path)

    exp = EmotionExperiment(config, dry_run=True)
    exp.is_vllm = False  # ensure non-vLLM branch in post-process
    exp.dataset = DummyDataset()  # inject our dummy dataset
    exp.cur_emotion = "anger"
    exp.cur_intensity = 1.0

    # Build a minimal batch with two items
    batch = {
        "prompts": ["P1", "P2"],
        "items": [
            BenchmarkItem(id="a", input_text="q1", context=None, ground_truth="gt1", metadata=None),
            BenchmarkItem(id="b", input_text="q2", context=None, ground_truth="gt2", metadata=None),
        ],
        "ground_truths": ["gt1", "gt2"],
    }
    control_outputs = [
        [{"generated_text": "resp1"}],
        [{"generated_text": "resp2"}],
    ]

    # Act: post-process batch to produce ResultRecord entries
    results = exp._post_process_batch(batch, control_outputs, batch_idx=0)

    # Assert: first item has error and NaN score; second has no error and score 1.0
    assert len(results) == 2
    assert results[0].error == "mock eval error"
    assert math.isnan(results[0].score if results[0].score is not None else float("nan"))
    assert results[1].error is None
    assert results[1].score == 1.0

    # Act: save results and load DataFrame
    df = exp._save_results(results)

    # Assert: error column exists and values match
    assert isinstance(df, pd.DataFrame)
    assert "error" in df.columns
    row0 = df[df["item_id"] == "a"].iloc[0]
    row1 = df[df["item_id"] == "b"].iloc[0]
    assert row0["error"] == "mock eval error"
    assert pd.isna(row1["error"]) or row1["error"] in (None, "")
