import copy
import json
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import Field

from games.game import BehaviorChoices, GameDecision, GameScenario
from games.payoff_matrices import PayoffMatrix


class ChickenBehaviorChoices(BehaviorChoices):
    withdraw: str
    escalate: str

    def get_choices(self):
        return [self.withdraw, self.escalate]

    def is_valid_choice(self, choice: str) -> bool:
        return choice in self.get_choices()

    def __str__(self):
        return f"Behavior Choices: {self.get_choices()}"

    @staticmethod
    def example():
        return {
            "withdraw": "Withdraw from the confrontation",
            "escalate": "Escalate the confrontation",
        }


class ChickenScenario(GameScenario):
    scenario: str
    description: str
    participants: List[Dict[str, Any]]
    behavior_choices: ChickenBehaviorChoices
    payoff_matrix: PayoffMatrix
    game_name: str = "Chicken_Game"

    def find_behavior_from_decision(self, decision: str) -> str:
        if decision == self.behavior_choices.withdraw:
            return "withdraw"
        elif decision == self.behavior_choices.escalate:
            return "escalate"
        else:
            raise ValueError(
                f"Decision must be one of {self.behavior_choices.get_choices()}"
            )

    def get_scenario_info(self) -> dict:
        return {"scenario": self.scenario, "description": self.description}

    def get_participants(self) -> list[str]:
        return self.participants

    def get_behavior_choices(self) -> ChickenBehaviorChoices:
        return self.behavior_choices

    @staticmethod
    def example():
        return {
            "scenario": "Border Standoff: The Territorial Dispute",
            "participants": [
                {
                    "name": "You",
                    "profile": "Military commander defending a strategic position",
                },
                {
                    "name": "Bob",
                    "profile": "Opposing military commander advancing on the same position",
                },
            ],
            "behavior_choices": ChickenBehaviorChoices.example(),
            "payoff_matrix_description": {
                "player 1: withdraw , player 2: withdraw": [
                    "player 1 gets 0: both forces withdraw, avoiding conflict but neither gains territory",
                    "player 2 gets 0: both forces withdraw, avoiding conflict but neither gains territory",
                ],
                "player 1: withdraw , player 2: escalate": [
                    "player 1 gets -1: loses face by backing down, cedes territory",
                    "player 2 gets 1: wins the standoff and gains strategic advantage",
                ],
                "player 1: escalate , player 2: withdraw": [
                    "player 1 gets 1: wins the standoff and gains strategic advantage",
                    "player 2 gets -1: loses face by backing down, cedes territory",
                ],
                "player 1: escalate , player 2: escalate": [
                    "player 1 gets -10: devastating conflict with major casualties and losses",
                    "player 2 gets -10: devastating conflict with major casualties and losses",
                ],
            },
            "description": "You and Bob are military commanders facing off over a strategic border position. Each of you must decide whether to withdraw your forces (avoiding conflict but potentially losing face) or escalate the confrontation (risking devastating conflict but potentially gaining strategic advantage). If both escalate, a devastating conflict is inevitable.",
        }


class ChickenDecision(GameDecision):
    scenario: ClassVar[Optional[ChickenScenario]] = None
    decision: str = Field(..., description="The decision made in the scenario")
    # rational: str = Field(..., description="The rationale for the decision")

    @classmethod
    def set_scenario(cls, scenario: GameScenario):
        if not isinstance(scenario, ChickenScenario):
            raise ValueError("Scenario must be a ChickenScenario")
        cls.scenario = scenario
        cls.model_fields["decision"].json_schema_extra = {
            "choices": scenario.get_behavior_choices().get_choices()
        }

    def validate_decision(self, decision: str) -> bool:
        if not self.scenario:
            raise ValueError(
                "Scenario must be set using Decision.set_scenario() before validating"
            )
        return self.scenario.get_behavior_choices().is_valid_choice(decision)

    @property
    def rational(self) -> str:
        return ""


if __name__ == "__main__":
    from autogen import config_list_from_json

    config_path = "config/OAI_CONFIG_LIST"
    config_list = config_list_from_json(config_path, filter_dict={"model": ["gpt-4o"]})
    cfg_ls_cp = copy.deepcopy(config_list)
    user = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        code_execution_config={"use_docker": False},
    )

    from constants import GameNames
    from games.payoff_matrices import ALL_GAME_PAYOFF

    payoff_matrix = ALL_GAME_PAYOFF[GameNames.CHICKEN_GAME]

    for file in Path("groupchat/scenarios/Chicken_Game").glob("*.json"):
        print(f" === begin: {file.name} ===\n")
        with open(file, "r") as f:
            data = json.load(f)
            data["payoff_matrix"] = payoff_matrix
            scenario = ChickenScenario(**data)

            ChickenDecision.set_scenario(scenario)

            for config in cfg_ls_cp:
                config["response_format"] = ChickenDecision

            assistant = AssistantAgent(
                name="Alice",
                llm_config={
                    "config_list": cfg_ls_cp,
                    "temperature": 0.9,
                },
                system_message="You are Alice, a military commander who values strategic advantage but also considers the costs of conflict. Make your decision carefully.",
            )

            message = f"Please analyze the following scenario: {scenario} and make your decision."
            while True:
                try:
                    res = user.initiate_chat(assistant, message=message, max_turns=1)
                    decision = ChickenDecision.model_validate_json(res.summary)
                    break
                except Exception as e:
                    print(f" === error: {e} ===")
                    message = (
                        f"Error: {e}\nPlease analyze the scenario again: {scenario}"
                    )

            behavior = scenario.find_behavior_from_decision(decision.decision)
            assert behavior is not None, f"Invalid decision: {decision.decision}"
            print(f" === behavior: {behavior} ===")
