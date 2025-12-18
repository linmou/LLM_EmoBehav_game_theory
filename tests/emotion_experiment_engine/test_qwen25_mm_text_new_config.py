# Tests for config/qwen25_MM_text_game_theory.yaml to ensure it uses text stimulus with multimodal model
import yaml
from pathlib import Path


def test_qwen25_mm_text_config_uses_text_stimulus():
    cfg_path = Path("config/qwen25_MM_text_game_theory.yaml")
    assert cfg_path.exists(), "Expected new-format config file is missing"

    cfg = yaml.safe_load(cfg_path.read_text())

    # Top-level keys should mirror new_game_theory_config format
    assert "models" in cfg and isinstance(cfg["models"], list)
    assert any("Qwen2.5-VL" in m for m in cfg["models"]), "Should target Qwen2.5 multimodal model"
    assert cfg.get("benchmarks"), "Benchmarks section required"

    repe_cfg = cfg.get("repe_eng_config", {})
    assert repe_cfg.get("data_dir") == "data_creation/stimulus_data"
    assert repe_cfg.get("multimodal_intent", False) is False
    assert "emotions" in repe_cfg and len(repe_cfg["emotions"]) >= 2
