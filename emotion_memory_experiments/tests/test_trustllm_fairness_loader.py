"""
Tests responsible for emotion_memory_experiments/datasets/trustllm_fairness.py
Purpose: loader parses minimal TrustLLM fairness JSON fixtures without transforms.
"""

from typing import Optional
from emotion_memory_experiments.data_models import BenchmarkConfig
from emotion_memory_experiments.datasets.trustllm_fairness import TrustLLMFairnessDataset


def _make_cfg(task_type: str, data_path: Optional[str]):
    return BenchmarkConfig(
        name="trustllm_fairness",
        task_type=task_type,
        data_path=data_path,
        base_data_dir=None,
        sample_limit=None,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=0.8,
        llm_eval_config={"model": "gpt-4o-mini", "temperature": 0.0},
    )


def test_loader_stereotype_recognition_minimal_fixture(tmp_path):
    data = [
        {"prompt": "Text1", "label": "stereotype"},
        {"prompt": "Text2", "label": "anti-stereotype"},
        {"prompt": "Text3", "label": "unrelated"},
    ]
    p = tmp_path / "fairness_recognition_sample.json"
    p.write_text(__import__("json").dumps(data), encoding="utf-8")

    ds = TrustLLMFairnessDataset(config=_make_cfg("stereotype_recognition", str(p)), prompt_wrapper=None)
    assert len(ds) == 3
    it0 = ds[0]
    assert it0["prompt"].startswith("Text1")
    assert it0["ground_truth"] == "stereotype"


def test_loader_stereotype_agreement_minimal_fixture(tmp_path):
    data = [
        {"prompt": "Agree?", "label": "yes"},
        {"prompt": "Agree?", "label": "no"},
    ]
    p = tmp_path / "fairness_agreement_sample.json"
    p.write_text(__import__("json").dumps(data), encoding="utf-8")

    ds = TrustLLMFairnessDataset(config=_make_cfg("stereotype_agreement", str(p)), prompt_wrapper=None)
    assert len(ds) == 2
    assert ds[1]["ground_truth"].lower() in ("yes", "no")
