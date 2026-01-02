"""
Responsible: auto_experiments/task_similarity/pipeline_config_reader.py
Purpose: Parse experiment_config.json for model/emotions/intensities.
"""

from pathlib import Path

import pytest


def test_read_pipeline_config_roundtrip(tmp_path: Path):
    from auto_experiments.task_similarity.pipeline_config_reader import read_pipeline_config

    cfg = {
        "model_path": "/models/qwen",
        "emotions": ["anger", "sadness"],
        "intensities": [0.6, 1.2],
    }
    p = tmp_path / "experiment_config.json"
    p.write_text(__import__("json").dumps(cfg), encoding="utf-8")

    out = read_pipeline_config(p)
    assert out.model_path == "/models/qwen"
    assert out.emotions == ["anger", "sadness"]
    assert out.intensities == [0.6, 1.2]


def test_read_pipeline_config_requires_keys(tmp_path: Path):
    from auto_experiments.task_similarity.pipeline_config_reader import read_pipeline_config

    p = tmp_path / "experiment_config.json"
    p.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="model_path"):
        read_pipeline_config(p)

