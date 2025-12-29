# Tests for config/qwen2.5_MM_text_Series_Prisoners_Dilemm.yaml ensuring multimodal model uses text stimuli
import yaml
from pathlib import Path


def test_qwen25_mm_text_config_sets_text_stimulus_dir():
    cfg_path = Path("config/qwen2.5_MM_text_Series_Prisoners_Dilemm.yaml")
    assert cfg_path.exists(), "New text-stimulus config is missing"

    config = yaml.safe_load(cfg_path.read_text())

    repe_section = config.get("repe_config", {})
    assert repe_section.get("data_dir") == "data_creation/stimulus_data"
    assert repe_section.get("multimodal_intent", False) is False

    models = config.get("experiment", {}).get("models", [])
    assert any("Qwen2.5-VL" in m for m in models), "Config should target a Qwen2.5 multimodal model"
