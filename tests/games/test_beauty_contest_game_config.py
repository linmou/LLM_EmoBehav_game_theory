# tests/games/test_beauty_contest_game_config.py
# Purpose: validate Beauty Contest game config wiring and scenario defaults + parsing.
import json
from pathlib import Path

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
