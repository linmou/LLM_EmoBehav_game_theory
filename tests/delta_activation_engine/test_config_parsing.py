# Responsible file: delta_activation_engine/config.py
# Purpose: YAML parsing into DeltaActivationJobConfig; validate required fields and types.

import io
import os
import sys
import textwrap
import pytest

# We expect these to exist in implementation phase
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from delta_activation_engine.config import (
    DeltaActivationJobConfig,
    load_job_config_from_yaml,
)


def test_load_job_config_success():
    # Minimal valid YAML (no defaults in dataclasses)
    yaml_str = textwrap.dedent(
        """
        model_path: /models/Qwen2.5-0.5B
        emotions: [anger, happiness]
        intensities: [0.5, 1.0]
        output_dir: results/delta_activations
        loading_config:
          model_path: /models/Qwen2.5-0.5B
          max_model_len: 4096
        repe_eng_config:
          control_method: reading_vec
          block_name: decoder_block
          rep_token: "<REP>"
          data_dir: data/stimulus/text/
          n_difference: 16
          direction_method: mean-diff
          emotions: [anger, happiness]
        """
    )

    cfg = load_job_config_from_yaml(io.StringIO(yaml_str))
    assert isinstance(cfg, DeltaActivationJobConfig)
    assert cfg.model_path == "/models/Qwen2.5-0.5B"
    assert cfg.emotions == ["anger", "happiness"]
    assert cfg.intensities == [0.5, 1.0]
    assert cfg.output_dir == "results/delta_activations"
    assert isinstance(cfg.loading_config, dict)
    assert isinstance(cfg.repe_eng_config, dict)


def test_load_job_config_missing_required():
    # Missing intensities should raise
    yaml_str = textwrap.dedent(
        """
        model_path: /models/Qwen2.5-0.5B
        emotions: [anger, happiness]
        output_dir: results/delta_activations
        loading_config: {max_model_len: 4096}
        repe_eng_config: {control_method: reading_vec, block_name: decoder_block}
        """
    )

    with pytest.raises((ValueError, TypeError)):
        _ = load_job_config_from_yaml(io.StringIO(yaml_str))
