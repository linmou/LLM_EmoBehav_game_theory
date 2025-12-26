"""Tests for `games/sealed_auction.py`: scenario parsing and behavior mapping.

Responsible file: games/sealed_auction.py
Purpose: Ensure Sealed_Auction scenarios load from JSON records and map option text -> behavior label.
"""

import pytest


def test_sealed_auction_scenario_maps_decisions_to_behavior_labels():
    from games.sealed_auction import SealedAuctionScenario

    record = {
        "scenario": "Crimson_Relay_Air_Drop_Bid",
        "description": (
            "You, the Protoss commander, are playing on **Crimson Relay** with long air lanes. "
            "Both sides secretly commit resources; higher commitment gains initiative."
        ),
        "participants": [{"name": "Protoss"}, {"name": "Terran"}],
        "behavior_choices": {
            "devote_none": "Keep scouting and position defensively with minimal extra commitment.",
            "devote_low": "Invest moderately: one timing setup and limited units with some risk.",
            "devote_high": "Commit heavily: multiple production cycles and major resource allocation.",
        },
    }

    scenario = SealedAuctionScenario(**record)
    choices = scenario.get_behavior_choices().get_choices()
    assert len(choices) == 3

    assert scenario.find_behavior_from_decision(choices[0]) == "devote_none"
    assert scenario.find_behavior_from_decision(choices[1]) == "devote_low"
    assert scenario.find_behavior_from_decision(choices[2]) == "devote_high"

    with pytest.raises(ValueError):
        scenario.find_behavior_from_decision("not a real choice")


def test_sealed_auction_game_config_loads_dataset_items():
    import json
    from pathlib import Path

    from games.sealed_auction import SealedAuctionScenario
    from games.game_configs import get_game_config

    game_config = get_game_config("Sealed_Auction")
    assert game_config["game_name"] == "Sealed_Auction"

    data_path = Path(game_config["data_path"])
    assert data_path.exists()

    records = json.loads(data_path.read_text(encoding="utf-8"))
    assert isinstance(records, list) and records

    scenario = SealedAuctionScenario(**records[0])
    assert len(scenario.get_behavior_choices().get_choices()) == 3
