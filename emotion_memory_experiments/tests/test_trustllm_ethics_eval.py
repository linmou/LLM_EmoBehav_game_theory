"""
Red tests for TrustLLMEthicsDataset.evaluate_response using GPT-4o-mini judge.

We mock evaluation_utils.llm_evaluate_response to return deterministic JSON
and assert that dataset.evaluate_response returns correct numeric scores.
"""

from unittest.mock import patch
import sys, types

# Ensure evaluation_utils can import without real OpenAI dependency
if "openai" not in sys.modules:
    sys.modules["openai"] = types.SimpleNamespace(OpenAI=lambda **kwargs: None)

from emotion_memory_experiments.data_models import BenchmarkConfig
from emotion_memory_experiments.datasets.trustllm_ethics import TrustLLMEthicsDataset


def _make_ds():
    cfg = BenchmarkConfig(
        name="trustllm_ethics",
        task_type="*",
        data_path=None,
        base_data_dir=None,
        sample_limit=None,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=0.8,
        llm_eval_config={"model": "gpt-4o-mini", "temperature": 0.0},
    )
    return TrustLLMEthicsDataset(config=cfg, prompt_wrapper=None)


@patch("emotion_memory_experiments.evaluation_utils.llm_evaluate_response")
def test_explicit_low_binary_correct(mock_llm):
    # Stub openai module to avoid import errors inside evaluation_utils
    if "openai" not in sys.modules:
        sys.modules["openai"] = types.SimpleNamespace(OpenAI=lambda **kwargs: None)
    mock_llm.return_value = {"score": 1}
    ds = _make_ds()
    score = ds.evaluate_response("Answer: 1", ground_truth="1", task_name="explicit_low", prompt="p")
    assert score == 1.0


@patch("emotion_memory_experiments.evaluation_utils.llm_evaluate_response")
def test_explicit_high_binary_incorrect(mock_llm):
    if "openai" not in sys.modules:
        sys.modules["openai"] = types.SimpleNamespace(OpenAI=lambda **kwargs: None)
    mock_llm.return_value = {"score": 0}
    ds = _make_ds()
    score = ds.evaluate_response("Not sure", ground_truth="A", task_name="explicit_high", prompt="p")
    assert score == 0.0


@patch("emotion_memory_experiments.evaluation_utils.llm_evaluate_response")
def test_implicit_ethics_label_match(mock_llm):
    if "openai" not in sys.modules:
        sys.modules["openai"] = types.SimpleNamespace(OpenAI=lambda **kwargs: None)
    mock_llm.return_value = {"label": "wrong"}
    ds = _make_ds()
    score = ds.evaluate_response("It is wrong", ground_truth="wrong", task_name="implicit_ethics", prompt="p")
    assert score == 1.0


@patch("emotion_memory_experiments.evaluation_utils.llm_evaluate_response")
def test_social_norm_label_mismatch(mock_llm):
    if "openai" not in sys.modules:
        sys.modules["openai"] = types.SimpleNamespace(OpenAI=lambda **kwargs: None)
    mock_llm.return_value = {"label": "neutral"}
    ds = _make_ds()
    score = ds.evaluate_response("This seems neutral", ground_truth="good", task_name="social_norm", prompt="p")
    assert score == 0.0
