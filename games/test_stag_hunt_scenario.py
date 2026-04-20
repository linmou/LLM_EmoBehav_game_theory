#!/usr/bin/env python3
# Purpose: test games/stag_hunt.py so StagHuntScenario.__str__ renders explicit previous_actions when present.

import unittest

from games.game import BehaviorChoices, GameScenario
from games.payoff_matrices import PayoffMatrix, stag_hunt as stag_hunt_payoff_leaves
from games.stag_hunt import StagHuntScenario


class _DummyBehaviors(BehaviorChoices):
    cooperate: str
    defect: str

    def is_valid_choice(self, choice: str) -> bool:
        return choice in self.get_choices()

    def get_choices(self) -> list[str]:
        return [self.cooperate, self.defect]

    @staticmethod
    def example() -> dict:
        return {"cooperate": "Work together", "defect": "Work alone"}


class _DummyScenario(GameScenario):
    scenario: str
    description: str
    participants: list[dict]
    behavior_choices: _DummyBehaviors

    def get_scenario_info(self) -> dict:
        return {"scenario": self.scenario, "description": self.description}

    def get_behavior_choices(self) -> _DummyBehaviors:
        return self.behavior_choices

    def find_behavior_from_decision(self, decision: str) -> str:
        return "cooperate" if decision == self.behavior_choices.cooperate else "defect"

    @staticmethod
    def example() -> dict:
        return {}


class TestStagHuntScenario(unittest.TestCase):
    def test_game_scenario_base_str_includes_previous_actions_when_present(self) -> None:
        scenario = _DummyScenario(
            scenario="Dummy pact",
            description="Dummy description",
            participants=[{"name": "You"}, {"name": "Other"}],
            behavior_choices={"cooperate": "Work together", "defect": "Work alone"},
            payoff_matrix=PayoffMatrix(player_num=2, payoff_leaves=stag_hunt_payoff_leaves),
            previous_actions=[{"round_summary": "Round one", "actions": [{"participant": "You", "action": "Work together"}]}],
        )

        rendered = str(scenario)

        self.assertIn("Previous Actions:", rendered)
        self.assertIn("Round one", rendered)

    def test_str_includes_previous_actions_when_present(self) -> None:
        scenario = StagHuntScenario(
            scenario="Naples Joint Offensive Pact",
            description="Austria and France are weighing whether to keep a coordinated Naples attack.",
            participants=[
                {"name": "You", "profile": "Commander of Austria"},
                {"name": "France", "profile": "Commander of France"},
            ],
            behavior_choices={
                "cooperate": "Press Naples along the line that changes the season if France presses too this turn.",
                "defect": "Press Naples along the line that still works if France does less this turn.",
            },
            payoff_matrix=PayoffMatrix(player_num=2, payoff_leaves=stag_hunt_payoff_leaves),
            previous_actions=[
                {
                    "round_summary": "In S1912M, you and France both stayed with the full Naples understanding.",
                    "actions": [
                        {
                            "participant": "You",
                            "action": "Stay with the full Naples understanding.",
                        },
                        {
                            "participant": "France",
                            "action": "Stay with the full Naples understanding.",
                        },
                    ],
                }
            ],
        )

        rendered = str(scenario)

        self.assertIn("Previous Actions:", rendered)
        self.assertIn("S1912M", rendered)
        self.assertIn("France", rendered)


if __name__ == "__main__":
    unittest.main()
