#!/usr/bin/env python3
# Tests for auto_experiments.layer_vector_sim.pd_steering_similarity.config_schema: config parsing and validation.

from pathlib import Path

import pytest

from auto_experiments.layer_vector_sim.pd_steering_similarity import config_schema


def write_yaml(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(content)
    return path


def test_load_config_parses_required_fields(tmp_path: Path) -> None:
    config_path = write_yaml(
        tmp_path,
        """
model:
  name: Qwen2.5-1.5B-Instruct
  path: /models/qwen
benchmark:
  name: game_theory
  task: Prisoners_Dilemma
  raw_results_path: /data/raw_results.json
steering:
  emotions: ["anger", "fear"]
  intensities: [0.5, 1.0]
  loader: emotion_experiment_engine.experiment.EmotionExperiment
pd_defection_vectors:
  dir: /vectors/pd_defection
output:
  dir: /tmp/output
""",
    )

    cfg = config_schema.load_config(config_path)

    assert cfg.model.name == "Qwen2.5-1.5B-Instruct"
    assert cfg.model.path == Path("/models/qwen")
    assert cfg.benchmark.task == "Prisoners_Dilemma"
    assert cfg.steering.emotions == ["anger", "fear"]
    assert cfg.steering.intensities == [0.5, 1.0]
    assert cfg.pd_defection_vectors.dir == Path("/vectors/pd_defection")
    assert cfg.output.dir == Path("/tmp/output")


def test_load_config_raises_on_missing_required_section(tmp_path: Path) -> None:
    config_path = write_yaml(
        tmp_path,
        """
model:
  name: Qwen2.5-1.5B-Instruct
  path: /models/qwen
# benchmark section missing
steering:
  emotions: ["anger"]
  intensities: [1.0]
  loader: emotion_experiment_engine.experiment.EmotionExperiment
pd_defection_vectors:
  dir: /vectors/pd_defection
output:
  dir: /tmp/output
""",
    )

    with pytest.raises(ValueError):
        config_schema.load_config(config_path)
