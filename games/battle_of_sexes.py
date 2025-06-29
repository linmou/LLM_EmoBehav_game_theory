import copy
import json
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import Field

from games.game import BehaviorChoices, GameDecision, GameScenario
from games.payoff_matrices import PayoffMatrix


class BOSBehaviorChoices(BehaviorChoices):
    your_preference: str
    others_preference: str

    def get_choices(self):
        return [self.your_preference, self.others_preference]

    def is_valid_choice(self, choice: str) -> bool:
        return choice in self.get_choices()

    def __str__(self):
        return f"Behavior Choices: {self.get_choices()}"

    @staticmethod
    def example():
        return {
            "your_preference": "Organize a one-day 'Community Health Fair'",
            "others_preference": "Launch a series of 'Mental Health Workshops'",
        }


class BattleOfSexesScenario(GameScenario):
    scenario: str
    description: str
    participants: List[Dict[str, Any]]
    behavior_choices: BOSBehaviorChoices
    payoff_matrix: PayoffMatrix
    game_name: str = "Battle_Of_Sexes"

    def find_behavior_from_decision(self, decision: str) -> str:
        if decision == self.behavior_choices.your_preference:
            return "your_preference"
        elif decision == self.behavior_choices.others_preference:
            return "others_preference"
        else:
            raise ValueError(
                f"Decision must be one of {self.behavior_choices.get_choices()}"
            )

    def get_scenario_info(self) -> dict:
        return {"scenario": self.scenario, "description": self.description}

    def get_behavior_choices(self) -> BOSBehaviorChoices:
        return self.behavior_choices

    @staticmethod
    def example():
        return {
            "scenario": "The Community Wellness Grant: Broad Outreach vs. Deep Impact",
            "participants": [
                {"name": "You", "profile": "Community Health Advocate"},
                {"name": "Bob", "profile": "Community Health Advocate"},
            ],
            "behavior_choices": BOSBehaviorChoices.example(),
            "payoff_matrix_description": {
                "player 1: your_preference , player 2: your_preference": [
                    "player 1 gets 2: The Health Fair is a massive success, providing preventative screenings to thousands and raising your profile as an effective organizer.",
                    "player 2 gets 1: Bob is pleased the community benefited from the event, but knows a critical need for focused mental health support was left unaddressed.",
                ],
                "player 1: your_preference , player 2: others_preference": [
                    "player 1 gets 0: Resources are split. The Health Fair is underfunded and poorly attended, while the workshops struggle to find participants. The grant is wasted and community trust is damaged.",
                    "player 2 gets 0: Resources are split. The workshops lack the outreach to be effective, while the fair is understaffed. The grant is wasted and community trust is damaged.",
                ],
                "player 1: others_preference , player 2: your_preference": [
                    "player 1 gets 0: Resources are split. You try to launch workshops while Bob plans a fair. Both initiatives fail due to a lack of concentrated effort and funding.",
                    "player 2 gets 0: Resources are split. Bob's fair plans conflict with your workshops. Both initiatives fail due to a lack of concentrated effort and funding.",
                ],
                "player 1: others_preference , player 2: others_preference": [
                    "player 1 gets 1: You are glad the workshops provided deep, meaningful help, but you know that thousands in the community who needed basic screenings were missed.",
                    "player 2 gets 2: The Mental Health Workshops have a profound, lasting impact on participants. Bob's specialized approach is validated and receives acclaim.",
                ],
            },
            "description": "You and Bob, fellow Community Health Advocates, have secured a major grant that can fund one significant community initiative this year. You must pool your resources and volunteers to make it a success. However, you disagree on the best approach. You propose to host a large 'Community Health Fair' to provide broad, preventative care to the greatest number of people. Bob proposes to launch a series of focused 'Mental Health Workshops' to provide deep, sustained support for a critical issue. If you both commit to one plan, the community benefits,  though the advocate who proposed the chosen initiative will receive more of the credit. If you split your resources to pursue different initiatives, both will be underfunded and fail, wasting the grant and letting the community down.",
        }


class BattleOfSexesDecision(GameDecision):
    scenario: ClassVar[Optional[BattleOfSexesScenario]] = None
    decision: str = Field(..., description="The decision made in the scenario")
    rational: str = Field(..., description="The rationale for the decision")

    @classmethod
    def set_scenario(cls, scenario: GameScenario):
        if not isinstance(scenario, BattleOfSexesScenario):
            raise ValueError("Scenario must be a BattleOfSexesScenario")
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

    from games.payoff_matrix import battle_of_sexes as payoff_matrix

    for file in Path("groupchat/scenarios/Battle_Of_Sexes").glob("*.json"):
        print(f" === begin: {file.name} ===\n")
        with open(file, "r") as f:
            data = json.load(f)
            data["payoff_matrix"] = payoff_matrix
            scenario = BattleOfSexesScenario(**data)

            BattleOfSexesDecision.set_scenario(scenario)

            for config in cfg_ls_cp:
                config["response_format"] = BattleOfSexesDecision

            assistant = AssistantAgent(
                name="Alice",
                llm_config={
                    "config_list": cfg_ls_cp,
                    "temperature": 0.9,
                },
                system_message="You are Alice, trying to coordinate evening plans with your partner. Consider both your preferences and the desire to spend time together.",
            )

            message = f"Please analyze the following scenario: {scenario} and make your decision."
            while True:
                try:
                    res = user.initiate_chat(assistant, message=message, max_turns=1)
                    decision = BattleOfSexesDecision.model_validate_json(res.summary)
                    break
                except Exception as e:
                    print(f" === error: {e} ===")
                    message = (
                        f"Error: {e}\nPlease analyze the scenario again: {scenario}"
                    )

            behavior = scenario.find_behavior_from_decision(decision.decision)
            assert behavior is not None, f"Invalid decision: {decision.decision}"
            print(f" === behavior: {behavior} ===")
