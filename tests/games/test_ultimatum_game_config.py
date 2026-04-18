#!/usr/bin/env python3
# Purpose: validate Ultimatum Game scenario loading, explicit previous_actions handling, and fallback history rules for proposer/responder views.

import pytest

from constants import GameNames
from games.game_configs import get_game_config
from games.ultimatum_game import (
    UltimatumGameDecision,
    UltimatumGameProposerScenario,
    UltimatumGameResponderScenario,
)


def _base_ultimatum_payload() -> dict:
    config = get_game_config(GameNames.ULTIMATUM_GAME_RESPONDER)
    return {
        "scenario": "Balkan Supply Allocation",
        "description": (
            "You are Germany commander. France has already made an offer over the next "
            "division of gains and you must decide whether to accept it."
        ),
        "participants": [
            {"name": "France", "profile": "Proposer_Power", "role": "Proposer"},
            {"name": "Germany", "profile": "Responder_Power", "role": "Responder"},
        ],
        "proposer_behavior_choices": {
            "offer_low": "offer 20% of the resulting territorial gain to Germany and keep 80% for France",
            "offer_medium": "offer 40% of the resulting territorial gain to Germany and keep 60% for France",
            "offer_high": "offer 50% of the resulting territorial gain to Germany and keep 50% for France",
        },
        "responder_behavior_choices": {
            "accept": "accept the proposed territorial split",
            "reject": "reject the proposed territorial split",
        },
        "payoff_matrix": config["payoff_matrix"],
    }


def test_ultimatum_game_config():
    config = get_game_config(GameNames.ULTIMATUM_GAME_RESPONDER)

    assert config["game_name"] == GameNames.ULTIMATUM_GAME_RESPONDER.value
    assert config["scenario_class"] is UltimatumGameResponderScenario
    assert config["decision_class"] is UltimatumGameDecision


def test_ultimatum_game_proposer_rejects_non_empty_explicit_previous_actions():
    with pytest.raises(ValueError, match="Proposer"):
        UltimatumGameProposerScenario(
            **_base_ultimatum_payload(),
            previous_actions=[
                ["France", "offer 20% of the resulting territorial gain to Germany and keep 80% for France"],
                ["Germany", "reject the proposed territorial split"],
            ],
            previous_actions_length=2,
        )


def test_ultimatum_game_responder_uses_explicit_previous_actions_when_present():
    scenario = UltimatumGameResponderScenario(
        **_base_ultimatum_payload(),
        previous_actions=[
            ["France", "offer 40% of the resulting territorial gain to Germany and keep 60% for France"],
        ],
        previous_actions_length=1,
        previous_offer_level=1,
    )

    assert scenario.previous_actions == [
        ("France", "offer 40% of the resulting territorial gain to Germany and keep 60% for France"),
    ]
    assert "France" in str(scenario)


def test_ultimatum_game_responder_explicit_previous_actions_do_not_require_previous_offer_level():
    scenario = UltimatumGameResponderScenario(
        **_base_ultimatum_payload(),
        previous_actions=[
            ["France", "offer 20% of the resulting territorial gain to Germany and keep 80% for France"],
            ["Germany", "reject the proposed territorial split"],
        ],
        previous_actions_length=2,
    )

    assert scenario.previous_actions == [
        ("France", "offer 20% of the resulting territorial gain to Germany and keep 80% for France"),
        ("Germany", "reject the proposed territorial split"),
    ]


def test_ultimatum_game_responder_falls_back_to_previous_offer_level_when_history_absent():
    scenario = UltimatumGameResponderScenario(
        **_base_ultimatum_payload(),
        previous_actions_length=1,
        previous_offer_level=2,
    )

    assert scenario.previous_actions == [
        ("France", "offer 50% of the resulting territorial gain to Germany and keep 50% for France"),
    ]


def test_ultimatum_game_responder_rejects_mismatched_previous_actions_length():
    with pytest.raises(ValueError, match="previous_actions_length"):
        UltimatumGameResponderScenario(
            **_base_ultimatum_payload(),
            previous_actions=[
                ["France", "offer 40% of the resulting territorial gain to Germany and keep 60% for France"],
            ],
            previous_actions_length=0,
            previous_offer_level=1,
        )


def test_ultimatum_game_responder_rejects_previous_offer_level_that_conflicts_with_explicit_history():
    with pytest.raises(ValueError, match="previous_offer_level"):
        UltimatumGameResponderScenario(
            **_base_ultimatum_payload(),
            previous_actions=[
                ["France", "offer 20% of the resulting territorial gain to Germany and keep 80% for France"],
            ],
            previous_actions_length=1,
            previous_offer_level=2,
        )
