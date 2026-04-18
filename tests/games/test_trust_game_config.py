#!/usr/bin/env python3
# Purpose: validate Trust Game scenario loading, explicit previous_actions handling, and fallback history rules for trustor/trustee views.

import pytest

from constants import GameNames
from games.game_configs import get_game_config
from games.trust_game import TrustGameDecision, TrustGameTrusteeScenario, TrustGameTrustorScenario


def _base_trust_payload() -> dict:
    config = get_game_config(GameNames.TRUST_GAME_TRUSTEE)
    return {
        "scenario": "Berlin Corridor Return Decision",
        "description": (
            "You are England commander. Austria has already committed support to your "
            "campaign and you must decide how much help to return."
        ),
        "participants": [
            {"name": "Austria", "profile": "Trustor_Power", "role": "Trustor"},
            {"name": "England", "profile": "Trustee_Power", "role": "Trustee"},
        ],
        "trustor_behavior_choices": {
            "trust_none": "commit 0% of Austria's available operational support to England",
            "trust_low": "commit about 30% of Austria's available operational support to England",
            "trust_high": "commit more than 80% of Austria's available operational support to England",
        },
        "trustee_behavior_choices": {
            "return_none": "return 0% of England's resulting operational support to Austria",
            "return_medium": "return about 40-50% of England's resulting operational support to Austria",
            "return_high": "return more than 80% of England's resulting operational support to Austria",
        },
        "payoff_matrix": config["payoff_matrix"],
    }


def test_trust_game_config():
    config = get_game_config(GameNames.TRUST_GAME_TRUSTEE)

    assert config["game_name"] == GameNames.TRUST_GAME_TRUSTEE.value
    assert config["scenario_class"] is TrustGameTrusteeScenario
    assert config["decision_class"] is TrustGameDecision


def test_trust_game_trustor_rejects_non_empty_explicit_previous_actions():
    with pytest.raises(ValueError, match="Trustor"):
        TrustGameTrustorScenario(
            **_base_trust_payload(),
            previous_actions=[
                ["Austria", "commit about 30% of Austria's available operational support to England"],
                ["England", "return about 40-50% of England's resulting operational support to Austria"],
            ],
            previous_actions_length=2,
        )


def test_trust_game_trustee_uses_explicit_previous_actions_when_present():
    scenario = TrustGameTrusteeScenario(
        **_base_trust_payload(),
        previous_actions=[
            ["Austria", "commit about 30% of Austria's available operational support to England"],
        ],
        previous_actions_length=1,
        previous_trust_level=0,
    )

    assert scenario.previous_actions == [
        ("Austria", "commit about 30% of Austria's available operational support to England"),
    ]
    assert "Austria" in str(scenario)


def test_trust_game_trustee_explicit_previous_actions_do_not_require_previous_trust_level():
    scenario = TrustGameTrusteeScenario(
        **_base_trust_payload(),
        previous_actions=[
            ["Austria", "commit about 30% of Austria's available operational support to England"],
            ["England", "return about 40-50% of England's resulting operational support to Austria"],
        ],
        previous_actions_length=2,
    )

    assert scenario.previous_actions == [
        ("Austria", "commit about 30% of Austria's available operational support to England"),
        ("England", "return about 40-50% of England's resulting operational support to Austria"),
    ]


def test_trust_game_trustee_falls_back_to_previous_trust_level_when_history_absent():
    scenario = TrustGameTrusteeScenario(
        **_base_trust_payload(),
        previous_actions_length=1,
        previous_trust_level=1,
    )

    assert scenario.previous_actions == [
        ("Austria", "commit more than 80% of Austria's available operational support to England"),
    ]


def test_trust_game_trustee_rejects_mismatched_previous_actions_length():
    with pytest.raises(ValueError, match="previous_actions_length"):
        TrustGameTrusteeScenario(
            **_base_trust_payload(),
            previous_actions=[
                ["Austria", "commit about 30% of Austria's available operational support to England"],
            ],
            previous_actions_length=0,
            previous_trust_level=0,
        )


def test_trust_game_trustee_rejects_previous_trust_level_that_conflicts_with_explicit_history():
    with pytest.raises(ValueError, match="previous_trust_level"):
        TrustGameTrusteeScenario(
            **_base_trust_payload(),
            previous_actions=[
                ["Austria", "commit about 30% of Austria's available operational support to England"],
            ],
            previous_actions_length=1,
            previous_trust_level=1,
        )
