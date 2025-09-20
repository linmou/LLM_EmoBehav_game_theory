"""
Red phase test for TrustLLMEthicsDataset split-level aggregation.

We construct minimal ResultRecord fixtures and expect the dataset to compute
an 'overall' metric as the mean of item-level scores (0..1).
"""

from typing import List

from emotion_memory_experiments.data_models import ResultRecord
from emotion_memory_experiments.datasets.trustllm_ethics import TrustLLMEthicsDataset
from emotion_memory_experiments.data_models import BenchmarkConfig


def _make_record(score: float) -> ResultRecord:
    return ResultRecord(
        emotion="anger",
        intensity=1.0,
        item_id="x",
        task_name="explicit_low",
        prompt="p",
        response="r",
        ground_truth="g",
        score=score,
        repeat_id=0,
    )


def test_ethics_compute_split_metrics_overall_mean():
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
        llm_eval_config=None,
    )
    ds = TrustLLMEthicsDataset(config=cfg, prompt_wrapper=None)
    records: List[ResultRecord] = [_make_record(1.0), _make_record(0.0), _make_record(1.0)]

    metrics = ds.compute_split_metrics(records)
    assert isinstance(metrics, dict)
    assert "overall" in metrics
    assert abs(metrics["overall"] - (2.0 / 3.0)) < 1e-9


def _make_labeled_record(label: str, correct: bool) -> ResultRecord:
    # score=1.0 if model predicted label correctly; 0.0 otherwise
    return ResultRecord(
        emotion="anger",
        intensity=1.0,
        item_id=f"{label}",
        task_name="social_norm",
        prompt="p",
        response="r",
        ground_truth=label,
        score=1.0 if correct else 0.0,
        repeat_id=0,
    )


def test_social_norm_macro_and_counts():
    cfg = BenchmarkConfig(
        name="trustllm_ethics",
        task_type="social_norm",
        data_path=None,
        base_data_dir=None,
        sample_limit=None,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=0.8,
        llm_eval_config=None,
    )
    ds = TrustLLMEthicsDataset(config=cfg, prompt_wrapper=None)

    # Build 3 good items (2 correct), 2 neutral (1 correct), 1 bad (0 correct)
    records = [
        _make_labeled_record("good", True),
        _make_labeled_record("good", True),
        _make_labeled_record("good", False),
        _make_labeled_record("neutral", True),
        _make_labeled_record("neutral", False),
        _make_labeled_record("bad", False),
    ]

    m = ds.compute_split_metrics(records)
    assert "social_norm" in m
    sn = m["social_norm"]
    # per-class counts
    assert sn["counts"]["good"] == 3
    assert sn["counts"]["neutral"] == 2
    assert sn["counts"]["bad"] == 1
    # per-class accs
    assert abs(sn["per_class_acc"]["good"] - (2/3)) < 1e-9
    assert abs(sn["per_class_acc"]["neutral"] - 0.5) < 1e-9
    assert abs(sn["per_class_acc"]["bad"] - 0.0) < 1e-9
    # macro = average of class accuracies
    expected_macro = ((2/3) + 0.5 + 0.0) / 3
    assert abs(sn["macro_acc"] - expected_macro) < 1e-9


def _make_ethics_record(gt: str, correct: bool) -> ResultRecord:
    return ResultRecord(
        emotion="anger",
        intensity=1.0,
        item_id=f"{gt}",
        task_name="implicit_ethics",
        prompt="p",
        response="r",
        ground_truth=gt,
        score=1.0 if correct else 0.0,
        repeat_id=0,
    )


def test_implicit_ethics_macro_and_counts():
    cfg = BenchmarkConfig(
        name="trustllm_ethics",
        task_type="implicit_ethics",
        data_path=None,
        base_data_dir=None,
        sample_limit=None,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=0.8,
        llm_eval_config=None,
    )
    ds = TrustLLMEthicsDataset(config=cfg, prompt_wrapper=None)

    # Build: wrong (2/3 correct), not wrong (1/2 correct), not sure (0/1 correct)
    recs = [
        _make_ethics_record("wrong", True),
        _make_ethics_record("wrong", True),
        _make_ethics_record("wrong", False),
        _make_ethics_record("not wrong", True),
        _make_ethics_record("not wrong", False),
        _make_ethics_record("not sure", False),
    ]
    m = ds.compute_split_metrics(recs)
    assert "implicit_ethics" in m
    ie = m["implicit_ethics"]
    assert ie["counts"]["wrong"] == 3
    assert ie["counts"]["not wrong"] == 2
    assert ie["counts"]["not sure"] == 1
    assert abs(ie["per_class_acc"]["wrong"] - (2/3)) < 1e-9
    assert abs(ie["per_class_acc"]["not wrong"] - 0.5) < 1e-9
    assert abs(ie["per_class_acc"]["not sure"] - 0.0) < 1e-9
    # Parity: macro excludes 'not sure'
    expected_macro = ((2/3) + 0.5) / 2
    assert abs(ie["macro_acc"] - expected_macro) < 1e-9
