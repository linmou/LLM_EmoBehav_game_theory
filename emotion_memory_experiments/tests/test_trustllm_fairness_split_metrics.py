"""
Tests responsible for emotion_memory_experiments/datasets/trustllm_fairness.py
Purpose: split-level metrics for fairness tasks (macro acc + counts for multi-class; micro acc for yes/no).
"""

from emotion_memory_experiments.data_models import BenchmarkConfig, ResultRecord
from emotion_memory_experiments.datasets.trustllm_fairness import TrustLLMFairnessDataset


def _make_cfg(task_type: str):
    return BenchmarkConfig(
        name="trustllm_fairness",
        task_type=task_type,
        data_path=None,
        base_data_dir=None,
        sample_limit=None,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=0.8,
        llm_eval_config=None,
    )


def _rec(task: str, gt: str, correct: bool) -> ResultRecord:
    return ResultRecord(
        emotion="anger",
        intensity=1.0,
        item_id=f"{gt}",
        task_name=task,
        prompt="p",
        response="r",
        ground_truth=gt,
        score=1.0 if correct else 0.0,
        repeat_id=0,
    )


def test_split_metrics_stereotype_recognition_macro_counts():
    ds = TrustLLMFairnessDataset(config=_make_cfg("stereotype_recognition"), prompt_wrapper=None)
    # stereotype: 2/3, anti: 1/2, unrelated: 0/1
    recs = [
        _rec("stereotype_recognition", "stereotype", True),
        _rec("stereotype_recognition", "stereotype", True),
        _rec("stereotype_recognition", "stereotype", False),
        _rec("stereotype_recognition", "anti-stereotype", True),
        _rec("stereotype_recognition", "anti-stereotype", False),
        _rec("stereotype_recognition", "unrelated", False),
    ]
    m = ds.compute_split_metrics(recs)
    assert "stereotype_recognition" in m
    fm = m["stereotype_recognition"]
    assert fm["counts"]["stereotype"] == 3
    assert fm["counts"]["anti-stereotype"] == 2
    assert fm["counts"]["unrelated"] == 1
    assert abs(fm["per_class_acc"]["stereotype"] - (2/3)) < 1e-9
    assert abs(fm["per_class_acc"]["anti-stereotype"] - 0.5) < 1e-9
    assert abs(fm["per_class_acc"]["unrelated"] - 0.0) < 1e-9
    expected_macro = ((2/3) + 0.5 + 0.0) / 3
    assert abs(fm["macro_acc"] - expected_macro) < 1e-9


def test_split_metrics_stereotype_agreement_micro():
    ds = TrustLLMFairnessDataset(config=_make_cfg("stereotype_agreement"), prompt_wrapper=None)
    recs = [
        _rec("stereotype_agreement", "yes", True),
        _rec("stereotype_agreement", "no", False),
        _rec("stereotype_agreement", "yes", True),
    ]
    m = ds.compute_split_metrics(recs)
    # overall micro accuracy only is fine; still ensure it's the mean
    assert "overall" in m
    assert abs(m["overall"] - (2/3)) < 1e-9

