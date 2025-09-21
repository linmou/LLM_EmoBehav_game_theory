# Tests for emotion_experiment_engine/experiment.py
"""Ensure EmotionExperiment persists dataset-driven choice ratio summaries."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from emotion_experiment_engine.data_models import (
    BenchmarkConfig,
    ExperimentConfig,
    ResultRecord,
)
from emotion_experiment_engine.experiment import EmotionExperiment


class _ChoiceRatioDataset:
    def __init__(self, payload: Dict[str, Any]):
        self.payload = payload
        self.calls: int = 0

    def compute_split_metrics(self, records: List[ResultRecord]) -> Dict[str, Any]:
        self.calls += 1
        return self.payload


@pytest.fixture()
def tmp_benchmark_config(tmp_path: Path) -> BenchmarkConfig:
    data_path = tmp_path / "games.jsonl"
    data_path.write_text("[]", encoding="utf-8")
    return BenchmarkConfig(
        name="game_theory",
        task_type="Prisoners_Dilemma",
        data_path=data_path,
        base_data_dir=None,
        sample_limit=None,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=1.0,
        llm_eval_config=None,
    )


def _build_results() -> List[ResultRecord]:
    return [
        ResultRecord(
            emotion="anger",
            intensity=0.1,
            item_id="pd-1",
            task_name="Prisoners_Dilemma",
            prompt="",
            response="",
            ground_truth=None,
            score=1.0,
            repeat_id=0,
        ),
        ResultRecord(
            emotion="anger",
            intensity=0.1,
            item_id="pd-2",
            task_name="Prisoners_Dilemma",
            prompt="",
            response="",
            ground_truth=None,
            score=2.0,
            repeat_id=1,
        ),
    ]


def test_save_results_writes_choice_ratio_csv(tmp_path: Path, tmp_benchmark_config: BenchmarkConfig) -> None:
    payload = {
        "choice_ratio": {
            "overall": [
                {"emotion": "anger", "intensity": 0.1, "option_id": 1, "ratio": 0.4},
                {"emotion": "anger", "intensity": 0.1, "option_id": 2, "ratio": 0.6},
            ],
            "by_repeat": [
                {
                    "emotion": "anger",
                    "intensity": 0.1,
                    "repeat_id": 0,
                    "option_id": 1,
                    "ratio": 1.0,
                },
                {
                    "emotion": "anger",
                    "intensity": 0.1,
                    "repeat_id": 1,
                    "option_id": 2,
                    "ratio": 1.0,
                },
            ],
        }
    }

    dataset = _ChoiceRatioDataset(payload)

    experiment_config = ExperimentConfig(
        model_path="/dev/null",
        emotions=["anger"],
        intensities=[0.1],
        benchmark=tmp_benchmark_config,
        output_dir=str(tmp_path),
        batch_size=1,
        generation_config=None,
        loading_config=None,
        repe_eng_config=None,
        max_evaluation_workers=1,
        pipeline_queue_size=1,
    )

    experiment = EmotionExperiment.__new__(EmotionExperiment)
    experiment.config = experiment_config
    experiment.logger = logging.getLogger("test-choice-ratio")
    experiment.logger.addHandler(logging.NullHandler())
    experiment.output_dir = tmp_path
    experiment.dataset = dataset
    experiment._save_experiment_config = lambda: None

    df = experiment._save_results(_build_results())
    assert not df.empty
    assert dataset.calls == 1

    split_path = tmp_path / "split_metrics.json"
    choice_ratio_path = tmp_path / "summary_choice_ratio.csv"
    choice_ratio_repeat_path = tmp_path / "summary_choice_ratio_by_repeat.csv"

    assert split_path.exists()
    persisted_payload = json.loads(split_path.read_text(encoding="utf-8"))
    assert persisted_payload == payload

    assert choice_ratio_path.exists()
    overall_df = pd.read_csv(choice_ratio_path)
    assert set(overall_df.columns) == {"emotion", "intensity", "option_id", "ratio"}
    assert len(overall_df) == 2
    overall_ratios = overall_df.sort_values("option_id")["ratio"].tolist()
    assert overall_ratios == pytest.approx([0.4, 0.6])

    assert choice_ratio_repeat_path.exists()
    repeat_df = pd.read_csv(choice_ratio_repeat_path)
    assert set(repeat_df.columns) == {"emotion", "intensity", "repeat_id", "option_id", "ratio"}
    assert len(repeat_df) == 2
    repeat_ratios = repeat_df.sort_values(["repeat_id", "option_id"])["ratio"].tolist()
    assert repeat_ratios == pytest.approx([1.0, 1.0])
