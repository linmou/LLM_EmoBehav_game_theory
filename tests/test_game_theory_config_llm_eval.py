# Tests for config/new_game_theory_decision_config.yaml; ensure all game-theory decision tasks use Gemini judges.
from __future__ import annotations

from pathlib import Path

import yaml


def test_all_game_theory_decision_benchmarks_use_gemini_llm_eval() -> None:
    config_path = Path("config/new_game_theory_decision_config.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    missing_client: list[str] = []
    missing_model: list[str] = []

    for bench in config.get("benchmarks", []):
        if bench.get("name") != "game_theory_decision":
            continue
        llm_cfg = bench.get("llm_eval_config")
        task = str(bench.get("task_type"))
        if not isinstance(llm_cfg, dict) or llm_cfg.get("client", "").lower() != "gemini":
            missing_client.append(task)
            continue
        if not llm_cfg.get("model"):
            missing_model.append(task)

    assert not missing_client, f"Missing Gemini llm_eval_config for tasks: {missing_client}"
    assert not missing_model, f"Missing Gemini model for tasks: {missing_model}"
