# Responsible file: download_vlm_models.sh
# Purpose: verify the downloader targets exactly the model paths required by config/vlm_mm_game_theory.yaml.

from __future__ import annotations

import ast
import pathlib
import re

import yaml


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "vlm_mm_game_theory.yaml"
SCRIPT_PATH = REPO_ROOT / "download_vlm_models.sh"


def _load_config_model_paths() -> list[str]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config["models"]


def _load_script_models_and_base_dir() -> tuple[list[str], str]:
    script_text = SCRIPT_PATH.read_text(encoding="utf-8")

    models_start = script_text.find("MODELS=(")
    assert models_start != -1, "MODELS array not found in download_vlm_models.sh"
    models_end = script_text.find("\n)", models_start)
    assert models_end != -1, "MODELS array closing marker not found in download_vlm_models.sh"
    models_block = script_text[models_start + len("MODELS=(") : models_end]
    active_model_lines = [
        line.strip()
        for line in models_block.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    model_ids = [
        ast.literal_eval(line)
        for line in active_model_lines
        if line.startswith('"') and line.endswith('"')
    ]

    base_dir_match = re.search(r'^BASE_DIR="([^"\n]+)"', script_text, re.MULTILINE)
    assert base_dir_match, "BASE_DIR assignment not found in download_vlm_models.sh"
    base_dir = ast.literal_eval(f'"{base_dir_match.group(1)}"')

    return model_ids, base_dir


def test_download_script_matches_vlm_config_model_paths() -> None:
    config_model_paths = _load_config_model_paths()
    model_ids, base_dir = _load_script_models_and_base_dir()
    script_model_paths = [f"{base_dir}/{model_id}" for model_id in model_ids]

    assert script_model_paths == config_model_paths
