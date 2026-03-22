"""Responsible file: neuro_manipulation/configs/experiment_config.py.

Purpose: ensure experiment-level emotions propagate into RepE config so
anger-only runs do not silently build readers for every default emotion.
"""

from neuro_manipulation.configs.experiment_config import get_repe_eng_config


def test_get_repe_eng_config_uses_top_level_emotions_when_repe_section_omits_them():
    config = {
        "emotions": ["anger"],
        "repe_eng_config": {
            "data_dir": "multimodal_crow_envnt/emotion_envent",
            "multimodal_intent": True,
        },
    }

    repe_config = get_repe_eng_config("fake/model", yaml_config=config)

    assert repe_config["emotions"] == ["anger"]
    assert repe_config["data_dir"] == "multimodal_crow_envnt/emotion_envent"


def test_get_repe_eng_config_keeps_explicit_repe_emotions_when_provided():
    config = {
        "emotions": ["anger"],
        "repe_eng_config": {
            "emotions": ["fear", "surprise"],
            "data_dir": "multimodal_crow_envnt/emotion_envent",
        },
    }

    repe_config = get_repe_eng_config("fake/model", yaml_config=config)

    assert repe_config["emotions"] == ["fear", "surprise"]
