"""
Tests responsible for emotion_memory_experiments/datasets/trustllm_fairness.py
Purpose: evaluator routes to GPT-4o-mini JSON and returns correct numeric score.
"""

from unittest.mock import patch
import sys, types

# Stub openai before importing evaluation_utils via patch targets
if "openai" not in sys.modules:
    sys.modules["openai"] = types.SimpleNamespace(OpenAI=lambda **kwargs: None)

from emotion_memory_experiments.data_models import BenchmarkConfig
from emotion_memory_experiments.datasets.trustllm_fairness import TrustLLMFairnessDataset


def _make_ds(task: str):
    cfg = BenchmarkConfig(
        name="trustllm_fairness",
        task_type=task,
        data_path=None,
        base_data_dir=None,
        sample_limit=None,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=0.8,
        llm_eval_config={"model": "gpt-4o-mini", "temperature": 0.0},
    )
    return TrustLLMFairnessDataset(config=cfg, prompt_wrapper=None)


@patch("emotion_memory_experiments.evaluation_utils.llm_evaluate_response")
def test_eval_stereotype_recognition_label_match(mock_llm):
    if "openai" not in sys.modules:
        sys.modules["openai"] = types.SimpleNamespace(OpenAI=lambda **kwargs: None)
    mock_llm.return_value = {"label": "anti-stereotype"}
    ds = _make_ds("stereotype_recognition")
    score = ds.evaluate_response("resp", ground_truth="anti-stereotype", task_name="stereotype_recognition", prompt="p")
    assert score == 1.0


@patch("emotion_memory_experiments.evaluation_utils.llm_evaluate_response")
def test_eval_stereotype_agreement_yes_no(mock_llm):
    if "openai" not in sys.modules:
        sys.modules["openai"] = types.SimpleNamespace(OpenAI=lambda **kwargs: None)
    mock_llm.return_value = {"label": "yes"}
    ds = _make_ds("stereotype_agreement")
    score = ds.evaluate_response("resp", ground_truth="yes", task_name="stereotype_agreement", prompt="p")
    assert score == 1.0
