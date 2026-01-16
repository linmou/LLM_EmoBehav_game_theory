"""Tests for auto_experiments/sc2_dataset/generate_sc2_scenarios_from_intent_map_with_gemini.py.
Responsible file: auto_experiments/sc2_dataset/generate_sc2_scenarios_from_intent_map_with_gemini.py
Purpose: Ensure `--game_name` drives prompt + validation via the selected game's schema."""

from __future__ import annotations

import pytest

from auto_experiments.sc2_dataset import generate_sc2_scenarios_from_intent_map_with_gemini as mod


def test_behavior_choice_keys_loaded_from_game_schema_sealed_auction() -> None:
    keys = mod._behavior_choice_keys_for_game_name("Sealed_Auction")
    assert keys == ("devote_none", "devote_low", "devote_high")


def test_behavior_choice_keys_loaded_from_game_schema_prisoners_dilemma() -> None:
    keys = mod._behavior_choice_keys_for_game_name("Prisoners_Dilemma")
    assert keys == ("cooperate", "defect")


def test_build_prompt_mentions_game_name_and_expected_keys() -> None:
    record = {
        "map_image": "frame.png",
        "description": "A brief map slice.",
        "intent_category": "air",
        "metadata": {"source_line_num": 1},
    }
    prompt = mod._build_prompt(
        instruction=["Keep it short."],
        examples=[{"scenario": "X", "description": "Y", "participants": [{"name": "A"}, {"name": "B"}], "behavior_choices": {"devote_none": "n", "devote_low": "l", "devote_high": "h"}}],
        record=record,
        game_name="Sealed_Auction",
    )
    assert "GameName MUST be exactly: Sealed_Auction" in prompt
    assert "behavior_choices keys MUST be:" in prompt
    assert "devote_none" in prompt and "devote_low" in prompt and "devote_high" in prompt


def test_validate_scenario_format_uses_game_specific_behavior_keys() -> None:
    scenario = {
        "scenario": "Any_AirControl",
        "description": "Desc",
        "participants": [{"name": "P1"}, {"name": "P2"}],
        "behavior_choices": {"cooperate": "c", "defect": "d"},
    }
    with pytest.raises(ValueError, match="behavior_choices\\.devote_none"):
        mod._validate_scenario_format(scenario, behavior_choice_keys=("devote_none", "devote_low", "devote_high"))

