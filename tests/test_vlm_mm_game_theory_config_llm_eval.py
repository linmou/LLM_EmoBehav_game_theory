# Tests for config/vlm_mm_game_theory_300.yaml; ensure all multimodal game-theory tasks use the non-decision benchmark with Gemini judges.
from __future__ import annotations

from pathlib import Path

import yaml


def test_all_vlm_mm_game_theory_benchmarks_use_game_theory_with_gemini() -> None:
    config_path = Path("config/vlm_mm_game_theory_300.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    wrong_benchmark_name: list[str] = []
    missing_client: list[str] = []
    missing_model: list[str] = []

    for bench in config.get("benchmarks", []):
        task = str(bench.get("task_type"))
        if bench.get("name") != "game_theory":
            wrong_benchmark_name.append(task)
        llm_cfg = bench.get("llm_eval_config")
        if not isinstance(llm_cfg, dict) or llm_cfg.get("client", "").lower() != "gemini":
            missing_client.append(task)
            continue
        if not llm_cfg.get("model"):
            missing_model.append(task)

    assert not wrong_benchmark_name, f"Unexpected benchmark names for tasks: {wrong_benchmark_name}"
    assert not missing_client, f"Missing Gemini llm_eval_config for tasks: {missing_client}"
    assert not missing_model, f"Missing Gemini model for tasks: {missing_model}"
