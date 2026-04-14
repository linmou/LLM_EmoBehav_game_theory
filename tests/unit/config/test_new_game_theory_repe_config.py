# Tests for config/new_game_theory_config.yaml to ensure RepE settings are exposed for runners.

import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config" / "new_game_theory_config.yaml"


def test_new_game_theory_config_has_repe_eng_config_block() -> None:
    with CONFIG_PATH.open("r") as f:
        cfg = yaml.safe_load(f)

    assert "repe_eng_config" in cfg, "Expected top-level repe_eng_config in new_game_theory_config.yaml"
    repe_cfg = cfg["repe_eng_config"]

    assert repe_cfg.get("data_dir") == "data_creation/stimulus_data"
    assert repe_cfg.get("emotions") == cfg.get("emotions")
