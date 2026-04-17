# tests/games/test_escalation_game_config.py
# Purpose: validate Escalation Game scenario loading, explicit previous_actions handling, and fallback history rules.

import pytest

from constants import GameNames
from games.escalation_game import EscalationGameDecision, EscalationGameScenario
from games.game_configs import get_game_config


def _base_escalation_payload() -> dict:
    config = get_game_config(GameNames.ESCALATION_GAME)
    return {
        "scenario": "Border Canal Water Standoff",
        "description": (
            "You and your neighbor must decide whether to keep normal water usage or "
            "escalate pumping from the same canal during a drought."
        ),
        "participants": [
            {"name": "You"},
            {"name": "Neighbor"},
        ],
        "behavior_choices": {
            "escalate": "Pump more water from the shared canal.",
            "withdraw": "Keep to the normal pumping level.",
        },
        "payoff_matrix": config["payoff_matrix"],
    }


def test_escalation_game_config():
    config = get_game_config(GameNames.ESCALATION_GAME)

    assert config["game_name"] == GameNames.ESCALATION_GAME.value
    assert config["scenario_class"] is EscalationGameScenario
    assert config["decision_class"] is EscalationGameDecision


def test_escalation_game_scenario_uses_explicit_previous_actions_when_present():
    scenario = EscalationGameScenario(
        **_base_escalation_payload(),
        previous_actions=[
            ["Neighbor", "Pump more water from the shared canal."],
        ],
        previous_actions_length=1,
    )

    assert scenario.previous_actions == [
        ("Neighbor", "Pump more water from the shared canal."),
    ]
    assert "Neighbor" in str(scenario)


def test_escalation_game_scenario_falls_back_to_previous_actions_length_when_history_absent():
    scenario = EscalationGameScenario(
        **_base_escalation_payload(),
        previous_actions_length=1,
    )

    assert scenario.previous_actions == [
        ("Neighbor", "Pump more water from the shared canal."),
    ]


def test_escalation_game_scenario_rejects_mismatched_previous_actions_length():
    with pytest.raises(ValueError, match="previous_actions_length"):
        EscalationGameScenario(
            **_base_escalation_payload(),
            previous_actions=[
                ["Neighbor", "Pump more water from the shared canal."],
            ],
            previous_actions_length=0,
        )


def test_escalation_game_scenario_accepts_transform_pipeline_output_with_provenance():
    scenario = EscalationGameScenario(
        **_base_escalation_payload(),
        previous_actions=[
            ["Neighbor", "Pump more water from the shared canal."],
        ],
        previous_actions_length=1,
        provenance={
            "id": "sg_case_1",
            "source_game_id": "game_1",
            "source_dataset": "standard_no_press.jsonl",
        },
    )

    assert scenario.game_name == "Escalation_Game"
    assert scenario.provenance["id"] == "sg_case_1"
    assert scenario.previous_actions == [
        ("Neighbor", "Pump more water from the shared canal."),
    ]
