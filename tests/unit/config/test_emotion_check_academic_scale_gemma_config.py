# Tests for config/emotion_check_academic_scale_gemma.yaml; ensure the Gemma sweep keeps all intended Gemma model paths enabled.

from pathlib import Path

import yaml


def test_emotion_check_academic_scale_gemma_config_enables_all_gemma_models():
    config_path = Path("config/emotion_check_academic_scale_gemma.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["models"] == [
        "${USER_HOME}/huggingface_models/google/gemma-3-270m-it",
        "${USER_HOME}/huggingface_models/google/gemma-3-1b-it",
        "${USER_HOME}/huggingface_models/google/gemma-3-4b-it",
    ]
