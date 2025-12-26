"""Tests for data/sc2/escalation_game.json: ensure SC2 escalation dataset structure and coverage.

Responsible file: data/sc2/escalation_game.json
Purpose: Validate basic schema, race coverage, and optional option metadata gradient.
"""

import json
from pathlib import Path
from typing import Any, Dict, List


DATA_PATH = Path("data/sc2/escalation_game.json")
EXPECTED_RACES = {"Protoss", "Terran", "Zerg"}


def _load_dataset() -> List[Dict[str, Any]]:
    text = DATA_PATH.read_text()
    data = json.loads(text)
    assert isinstance(
        data, list
    ), "SC2 escalation dataset must be a JSON list of scenario objects"
    return data


def test_sc2_escalation_dataset_structure_and_race_coverage() -> None:
    dataset = _load_dataset()

    # Basic size requirement so we have a meaningful dataset, not a single example.
    assert (
        len(dataset) >= 10
    ), "Expected at least 10 SC2 escalation scenarios in the dataset"

    races_seen = set()
    for scenario in dataset:
        assert isinstance(scenario.get("description"), str) and scenario[
            "description"
        ].strip(), "Each scenario must have a non-empty description"

        you_play_as = scenario.get("you_play_as")
        assert (
            you_play_as in EXPECTED_RACES
        ), f"'you_play_as' must be one of {sorted(EXPECTED_RACES)}, got {you_play_as!r}"
        races_seen.add(you_play_as)

        behaviour = scenario.get("behaviour_decisions")
        assert isinstance(
            behaviour, dict
        ), "Each scenario must have a 'behaviour_decisions' mapping"
        for key in ("escalate", "withdraw"):
            assert key in behaviour, f"'behaviour_decisions' missing key {key!r}"
            options = behaviour[key]
            assert isinstance(
                options, list
            ), f"'behaviour_decisions[{key}]' must be a list of strings"
            assert (
                options
            ), f"'behaviour_decisions[{key}]' must contain at least one option"
            for opt in options:
                assert isinstance(
                    opt, str
                ) and opt.strip(), "Behaviour decision options must be non-empty strings"

        players = scenario.get("players")
        assert isinstance(players, dict), "Each scenario must define 'players'"
        for player_key in ("player_1", "player_2"):
            assert (
                player_key in players
            ), f"'players' mapping must contain {player_key!r}"
            player = players[player_key]
            assert isinstance(
                player, dict
            ), f"{player_key!r} entry must be a mapping with player details"
            for field in ("race", "role", "economy", "army", "advantage"):
                value = player.get(field)
                assert isinstance(
                    value, str
                ) and value.strip(), f"{player_key!r}.{field} must be a non-empty string"
            assert (
                player["race"] in EXPECTED_RACES
            ), f"{player_key!r}.race must be a StarCraft II race, got {player['race']!r}"

        # Ensure that the player listed as "You" matches the 'you_play_as' race.
        player_1 = players.get("player_1", {})
        assert (
            player_1.get("role") == "You"
        ), "player_1.role should be 'You' in SC2 escalation scenarios"
        assert (
            player_1.get("race") == you_play_as
        ), "player_1.race should match 'you_play_as'"

        # Optional structured option metadata: if present, enforce gradient semantics.
        options_meta = scenario.get("options")
        if options_meta is not None:
            assert isinstance(
                options_meta, list
            ), "'options' metadata must be a list when present"

            # Default view should expose two options: strongest escalate and strongest withdraw.
            assert (
                len(options_meta) == 2
            ), "'options' metadata must expose two primary options by default"

            ids = [opt.get("id") for opt in options_meta]
            assert ids == [1, 2], "Option ids must be 1,2 in order for primary options"

            categories = [opt.get("category") for opt in options_meta]
            assert categories == [
                "escalate",
                "withdraw",
            ], "Primary options must be escalate then withdraw"

            strengths = [opt.get("escalation_strength") for opt in options_meta]
            assert strengths == [
                2,
                -2,
            ], "Primary escalation_strength must be [2,-2]"

            # When all_options is provided, enforce full gradient for completeness.
            all_options = scenario.get("all_options")
            if all_options is not None:
                assert isinstance(
                    all_options, list
                ), "'all_options' must be a list when present"
                assert len(all_options) == 4, "'all_options' must contain four entries"
                all_ids = [opt.get("id") for opt in all_options]
                assert all_ids == [1, 2, 3, 4], "all_options ids must be 1,2,3,4"
                all_categories = [opt.get("category") for opt in all_options]
                assert all_categories[:2] == [
                    "escalate",
                    "escalate",
                ], "all_options 1,2 must be escalate"
                assert all_categories[2:] == [
                    "withdraw",
                    "withdraw",
                ], "all_options 3,4 must be withdraw"
                all_strengths = [opt.get("escalation_strength") for opt in all_options]
                assert all_strengths == [
                    2,
                    1,
                    -1,
                    -2,
                ], "all_options escalation_strength must be [2,1,-1,-2]"

    # Dataset-level coverage: make sure all three core SC2 races appear.
    assert EXPECTED_RACES.issubset(
        races_seen
    ), "Dataset must include scenarios where you play as Protoss, Terran, and Zerg"
