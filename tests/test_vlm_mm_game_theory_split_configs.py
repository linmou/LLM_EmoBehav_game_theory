# Tests for config/vlm_mm_game_theory_300_gpu01.yaml and config/vlm_mm_game_theory_300_gpu23.yaml; ensure the split sweeps preserve benchmark semantics while partitioning models for 2-GPU runs.
from __future__ import annotations

from pathlib import Path

import yaml


def _load_yaml(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def test_split_configs_partition_models_and_keep_same_benchmarks() -> None:
    base_cfg = _load_yaml("config/vlm_mm_game_theory_300.yaml")
    gpu01_cfg = _load_yaml("config/vlm_mm_game_theory_300_gpu01.yaml")
    gpu23_cfg = _load_yaml("config/vlm_mm_game_theory_300_gpu23.yaml")

    base_models = set(base_cfg["models"])
    gpu01_models = set(gpu01_cfg["models"])
    gpu23_models = set(gpu23_cfg["models"])

    assert gpu01_models, "gpu01 split must include at least one model"
    assert gpu23_models, "gpu23 split must include at least one model"
    assert gpu01_models.isdisjoint(gpu23_models), "split configs must not overlap models"
    assert gpu01_models | gpu23_models == base_models, "split configs must cover exactly the base models"

    assert gpu01_cfg["benchmarks"] == base_cfg["benchmarks"]
    assert gpu23_cfg["benchmarks"] == base_cfg["benchmarks"]
    assert gpu01_cfg["emotions"] == base_cfg["emotions"]
    assert gpu23_cfg["emotions"] == base_cfg["emotions"]
    assert gpu01_cfg["intensities"] == base_cfg["intensities"]
    assert gpu23_cfg["intensities"] == base_cfg["intensities"]

    assert gpu01_cfg["loading_config"]["tensor_parallel_size"] == 2
    assert gpu23_cfg["loading_config"]["tensor_parallel_size"] == 2
