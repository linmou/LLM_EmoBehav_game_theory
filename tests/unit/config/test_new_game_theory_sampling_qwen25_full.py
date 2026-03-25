#!/usr/bin/env python3
# Responsible file: config/new_game_theory_sampling_qwen25_full.yaml
# Purpose: verify the merged qwen2.5 sampling config targets the game_theory benchmark and includes the three requested models.

from __future__ import annotations

from pathlib import Path

import yaml


def test_merged_qwen25_sampling_config_targets_game_theory() -> None:
    config_path = Path("config/new_game_theory_sampling_qwen25_full.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["experiment_name"] == "new_game_theory_qwen25_full"
    assert config["models"] == [
        "/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct",
        "/home/jjl7137/huggingface_models/Qwen/Qwen2.5-1.5B-Instruct",
        "/home/jjl7137/huggingface_models/Qwen/Qwen2.5-3B-Instruct",
    ]
    assert all(bench["name"] == "game_theory" for bench in config["benchmarks"])
    assert config["intensities"] == [0.8, 1.0, 1.2]
    assert config["repeat_runs"] == 3
    assert config["generation_config"]["do_sample"] is True
    assert config["repe_eng_config"]["control_layers"]["strategy"] == "middle_third"
