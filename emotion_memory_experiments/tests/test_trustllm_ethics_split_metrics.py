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

