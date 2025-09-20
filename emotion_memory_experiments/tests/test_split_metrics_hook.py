"""
Red phase tests for split-level metrics integration.

We start by asserting that datasets expose a split-level aggregation hook and
that the experiment persistence layer writes out the aggregated metrics.

Both tests are expected to FAIL initially (TDD Red) until the production
implementations are added.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import List

import pytest

from emotion_memory_experiments.datasets.base import BaseBenchmarkDataset
from emotion_memory_experiments.data_models import ResultRecord
from emotion_memory_experiments.experiment import EmotionExperiment


def test_base_dataset_exposes_split_level_hook():
    """
    Expect BaseBenchmarkDataset to define a split-level metrics hook so all
    concrete datasets can implement dataset-scope aggregation.

    This test drives the addition of a default method on the base class.
    """
    assert hasattr(
        BaseBenchmarkDataset, "compute_split_metrics"
    ), "BaseBenchmarkDataset must expose compute_split_metrics(records)"

    hook = getattr(BaseBenchmarkDataset, "compute_split_metrics")
    assert callable(hook), "compute_split_metrics must be callable"


def test_experiment_persists_split_metrics(tmp_path: Path):
    """
    Expect EmotionExperiment to persist a split_metrics.json file produced by
    the dataset's compute_split_metrics hook when saving results.

    We avoid constructing a full experiment by creating a minimal instance
    and patching only the attributes consumed by _save_results.
    """
    # Minimal fake dataset exposing the hook
    class _FakeDataset:
        def compute_split_metrics(self, records: List[ResultRecord]):
            return {"overall": 0.5}

    # Create a bare EmotionExperiment instance without running __init__
    exp = EmotionExperiment.__new__(EmotionExperiment)
    exp.logger = logging.getLogger("test_exp")
    exp.logger.addHandler(logging.NullHandler())
    exp.output_dir = tmp_path
    exp.dataset = _FakeDataset()

    # _save_results calls these helpers; stub to no-op
    def _noop_save_config():
        cfg_file = tmp_path / "experiment_config.json"
        cfg_file.write_text("{}")

    exp._save_experiment_config = _noop_save_config  # type: ignore[attr-defined]

    # Build a tiny result list for save path
    records = [
        ResultRecord(
            emotion="anger",
            intensity=1.0,
            item_id="id-1",
            task_name="dummy",
            prompt="q",
            response="r",
            ground_truth="g",
            score=1.0,
            repeat_id=0,
        )
    ]

    # Execute save path
    exp._save_results(records)

    # Assert split-level metrics file exists and has expected content
    metrics_path = tmp_path / "split_metrics.json"
    assert metrics_path.exists(), "split_metrics.json must be persisted by _save_results()"
    content = json.loads(metrics_path.read_text())
    assert content == {"overall": 0.5}

