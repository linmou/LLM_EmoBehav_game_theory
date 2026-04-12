# tests/games/test_beauty_contest_game_config.py
# Purpose: validate Beauty Contest game config wiring and scenario defaults + parsing.
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from constants import GameNames
from games.beauty_contest import BeautyContestDecision, BeautyContestScenario
from games.game_configs import get_game_config


def test_beauty_contest_game_config():
    config = get_game_config(GameNames.BEAUTY_CONTEST)

    assert config["game_name"] == GameNames.BEAUTY_CONTEST.value
    assert config["scenario_class"] is BeautyContestScenario
    assert config["decision_class"] is BeautyContestDecision
    assert isinstance(config["payoff_matrix"], dict)


def test_beauty_contest_scenario_allows_multiple_participants():
    scenario = BeautyContestScenario(
        scenario="Roundtable Forecast",
        description="Each participant submits a number. Closest wins.",
        participants=[
            {"name": "You (Tech Leader)"},
            {"name": "10 Competing Tech Firms"},
        ],
        payoff_matrix={},
        behavior_choices={
            "commit_0": "Commit 0 engineers",
            "commit_1": "Commit 25 engineers",
            "commit_2": "Commit 50 engineers",
            "commit_3": "Commit 100 engineers",
        },
    )

    assert len(scenario.get_participants()) == 2
    assert len(scenario.get_behavior_choices().get_choices()) == 4


def test_beauty_contest_scenario_parses_generated_file_shape():
    path = Path(
        "data_creation/scenario_creation/langgraph_creation/Beauty_Contest_all_data_samples.json"
    )
    if not path.exists():
        pytest.skip(f"Generated sample file not present: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(raw, list) and raw, "Expected non-empty scenario list"

    config = get_game_config(GameNames.BEAUTY_CONTEST)
    scenario_class = config["scenario_class"]
    record = dict(raw[0])
    record.setdefault("payoff_matrix", config["payoff_matrix"])

    scenario = scenario_class(**record)
    choices = scenario.get_behavior_choices().get_choices()
    assert len(choices) == 4
    assert scenario.find_behavior_from_decision(choices[0]).startswith("commit_")


def test_beauty_contest_previous_actions_accepts_canonical_rounds_and_renders_in_str():
    scenario = BeautyContestScenario(
        scenario="Moscow Contest",
        description="Choose how hard to join the current contest.",
        participants=[
            {"name": "England"},
            {"name": "France"},
            {"name": "Germany"},
        ],
        payoff_matrix={},
        behavior_choices={
            "commit_0": "Devote nothing to the contested push this turn",
            "commit_1": "Issue one SUPPORT_ATTACK order into the contested push",
            "commit_2": "Issue one ATTACK and one SUPPORT_ATTACK order against the contested target",
            "commit_3": "Issue one ATTACK and two SUPPORT_ATTACK orders against the contested target",
        },
        previous_actions=[
            {
                "time": "S1905M",
                "participant_actions": [
                    [
                        "England",
                        "Issued one SUPPORT_ATTACK order into the contested push",
                    ],
                    [
                        "France",
                        "Issued one SUPPORT_ATTACK order into the contested push",
                    ],
                    [
                        "Germany",
                        "Issued one ATTACK and one SUPPORT_ATTACK order against the contested target",
                    ],
                ],
                "round_summary": "England and France got the best relative payoff in that round.",
            }
        ],
    )

    expected_previous_actions = [
        {
            "time": "S1905M",
            "participant_actions": [
                [
                    "England",
                    "Issued one SUPPORT_ATTACK order into the contested push",
                ],
                [
                    "France",
                    "Issued one SUPPORT_ATTACK order into the contested push",
                ],
                [
                    "Germany",
                    "Issued one ATTACK and one SUPPORT_ATTACK order against the contested target",
                ],
            ],
            "round_summary": "England and France got the best relative payoff in that round.",
        }
    ]

    assert scenario.previous_actions() == expected_previous_actions
    assert "Previous Actions:" in str(scenario)
    assert "S1905M" in str(scenario)


def test_beauty_contest_previous_actions_rejects_invalid_shape():
    with pytest.raises(ValueError, match="previous_actions"):
        BeautyContestScenario(
            scenario="Broken Contest",
            description="Broken previous actions payload.",
            participants=[
                {"name": "England"},
                {"name": "France"},
            ],
            payoff_matrix={},
            behavior_choices={
                "commit_0": "Devote nothing to the contested push this turn",
                "commit_1": "Issue one SUPPORT_ATTACK order into the contested push",
                "commit_2": "Issue one ATTACK and one SUPPORT_ATTACK order against the contested target",
                "commit_3": "Issue one ATTACK and two SUPPORT_ATTACK orders against the contested target",
            },
            previous_actions=[
                {
                    "time": "S1905M",
                    "participant_actions": [["England"]],
                }
            ],
        )


def test_beauty_contest_scenario_accepts_transform_pipeline_output_with_provenance():
    scenario = BeautyContestScenario(
        scenario="Adriatic Coalition Push",
        description=(
            "You are a Italy commander. The coalition expects a measured contribution, "
            "and each commitment level changes both your leverage and your exposure."
        ),
        participants=[
            {"name": "Italy"},
            {"name": "Austria"},
            {"name": "France"},
        ],
        payoff_matrix={},
        behavior_choices={
            "commit_0": "Commit no units to the coordinated push this phase.",
            "commit_1": "Commit one unit to the coordinated push this phase.",
            "commit_2": "Commit two units to the coordinated push this phase.",
            "commit_3": "Commit three units to the coordinated push this phase.",
        },
        previous_actions=[],
        game_category="BC2",
        provenance={
            "id": "sg_case_1",
            "source_game_id": "game_1",
            "source_dataset": "standard_no_press.jsonl",
        },
    )

    assert scenario.game_name == "Beauty_Contest"
    assert scenario.provenance["id"] == "sg_case_1"
    assert scenario.get_behavior_choices().get_choices()[0].startswith("Commit no units")


def test_beauty_contest_scenario_constructor_accepts_noncanonical_game_name():
    scenario = BeautyContestScenario(
        scenario="Adriatic Coalition Push",
        description="A valid Beauty Contest scenario can still carry a noncanonical game_name string.",
        participants=[
            {"name": "Italy"},
            {"name": "Austria"},
        ],
        payoff_matrix={},
        behavior_choices={
            "commit_0": "Commit no units to the coordinated push this phase.",
            "commit_1": "Commit one unit to the coordinated push this phase.",
            "commit_2": "Commit two units to the coordinated push this phase.",
            "commit_3": "Commit three units to the coordinated push this phase.",
        },
        previous_actions=[],
        game_name="Not_The_Canonical_Name",
    )

    assert scenario.game_name == "Not_The_Canonical_Name"


def test_beauty_contest_scenario_rejects_extra_behavior_choice_keys():
    with pytest.raises(ValidationError, match="commit_4"):
        BeautyContestScenario(
            scenario="Adriatic Coalition Push",
            description="Extra behavior choice keys should be rejected by the real contract.",
            participants=[
                {"name": "Italy"},
                {"name": "Austria"},
            ],
            payoff_matrix={},
            behavior_choices={
                "commit_0": "Commit no units to the coordinated push this phase.",
                "commit_1": "Commit one unit to the coordinated push this phase.",
                "commit_2": "Commit two units to the coordinated push this phase.",
                "commit_3": "Commit three units to the coordinated push this phase.",
                "commit_4": "Commit four units to the coordinated push this phase.",
            },
            previous_actions=[],
        )
