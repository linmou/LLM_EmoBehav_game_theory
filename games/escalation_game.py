import random
from typing import Any, ClassVar, Dict, Optional

from pydantic import Field

from games.game import (
    BehaviorChoices,
    GameDecision,
    GameScenario,
    SequentialGameScenario,
)


class EGBehaviorChoices(BehaviorChoices):
    escalation: str
    withdraw: str

    def get_choices(self):
        return [self.escalation, self.withdraw]

    def is_valid_choice(self, choice: str) -> bool:
        return choice in self.get_choices()

    def __str__(self):
        return f"Behavior Choices: {self.get_choices()}"

    @staticmethod
    def example():
        return {
            "escalation": "Attempt to draw more water from the stream",
            "withdraw": "Stick to current minimal water usage, reduce your take.",
        }


class EscalationGameScenario(SequentialGameScenario):
    scenario: str
    description: str
    participants: list[dict]
    behavior_choices: EGBehaviorChoices
    previous_actions_length: int
    payoff_matrix: Dict[tuple[str, str], Any]
    game_name: str = "Escalation_Game"

    def get_scenario_info(self) -> dict:
        return {"scenario": self.scenario, "description": self.description}

    def get_participants(self) -> list[dict]:
        return self.participants

    def get_payoff_matrix(self) -> Dict[str, Any]:
        return self.payoff_matrix

    def get_participant_names(self) -> list[str]:
        return [participant["name"] for participant in self.participants]

    def get_behavior_choices(self) -> EGBehaviorChoices:
        return self.behavior_choices

    def find_behavior_from_decision(self, decision: str) -> str:
        if decision == self.behavior_choices.escalation:
            return "escalation"
        elif decision == self.behavior_choices.withdraw:
            return "withdraw"
        else:
            raise ValueError(f"Invalid decision: {decision}")

    @property
    def previous_actions(self) -> list[tuple[str, str]]:
        previous_actions = []
        for i in range(
            self.previous_actions_length,
        ):
            previous_actions.append(
                (
                    self.get_participant_names()[(i + 1) % 2],
                    self.behavior_choices.escalation,
                )
            )  # only one action
        previous_actions.reverse()  # ensure the last actor is not 'You'
        if self.previous_actions_length > 0:
            assert previous_actions[-1][0] != self.get_participant_names()[0]
        return previous_actions

    @staticmethod
    def example():
        return {
            "scenario": "Farm_Water_Conflict",
            "participants": [
                {"name": "You", "role": "Farmer"},
                {"name": "Bob", "role": "Farmer"},
            ],
            "behavior_choices": EGBehaviorChoices.example(),
            "payoff_matrix_description": {
                "first_player: withdraw": [
                    "first_player gets 0: first_player sticks to minimal water usage. first_player's harvest is modest due to the drought, but first_player avoids further conflict and extra costs for escalation.",
                    "second_player gets 0: second_player also takes a minimal amount of water. second_player's harvest is also modest, and the situation remains stable without escalating conflict.",
                ],
                "first_player: escalate, second_player: withdraw": [
                    "first_player gets 1: first_player successfully takes more water by escalating, and second_player concedes. first_player's harvest improves significantly.",
                    "second_player gets -2: second_player withdraws in the face of first_player's escalation, getting very little water. second_player's crops suffer badly, and second_player may incur costs trying to find alternatives.",
                ],
                "first_player: escalate, second_player: escalate, first_player: withdraw": [
                    "first_player gets -2: first_player escalates, second_player escalates in response, and then first_player withdraws. first_player has wasted resources on escalation and now gets less water or a worse outcome than second_player.",
                    "second_player gets 1: second_player successfully counters first_player's escalation and first_player backs down. second_player secures more water and a better harvest due to second_player's firm stance.",
                ],
                "first_player: escalate, second_player: escalate, first_player: escalate": [
                    "first_player gets -1: Both first_player and second_player continuously escalate. The conflict is costly (e.g., damaged equipment, depleted stream, wasted time/money). Both harvests suffer, and both are worse off than if the conflict hadn't fully escalated.",
                    "second_player gets -1: Both second_player and first_player continuously escalate. The conflict is costly (e.g., damaged equipment, depleted stream, wasted time/money). Both harvests suffer, and both are worse off than if the conflict hadn't fully escalated.",
                ],
            },
            "description": "You and Bob are neighboring farmers in the Willow Creek Valley, relying on a shared stream, the 'Silver Run,' for irrigation. A severe drought has drastically reduced the stream's flow, creating a critical water shortage. Both of your harvests are at risk, and you must decide how to manage your water intake from the dwindling Silver Run. Each farmer's actions will impact the other's ability to draw water.",
        }

    def __str__(self):
        info = self.get_scenario_info()
        return f"""
        Scenario: {info.get('scenario', 'Unnamed')}
        Description: {info.get('description', 'No description')}
        Participants: {self.get_participants()}
        Behavior Choices: {self.get_behavior_choices().get_choices()}
        Previous Actions: {self.previous_actions}
        """


class EscalationGameDecision(GameDecision):
    scenario: ClassVar[Optional[EscalationGameScenario]] = None
    decision: str = Field(..., description="The decision made in the scenario")
    # rational: str = Field(..., description="The rationale for the decision")

    @classmethod
    def set_scenario(cls, scenario: GameScenario):
        if not isinstance(scenario, EscalationGameScenario):
            raise ValueError("Scenario must be a EscalationGameScenario")
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
    import copy
    import json
    from pathlib import Path

    from autogen import AssistantAgent, UserProxyAgent

    data_json = "groupchat/scenarios/Escalation_Game/Art_Auction_Showdown.json"
    with open(data_json, "r") as f:
        data = json.load(f)
    from games.payoff_matrix import escalation_game

    data["payoff_matrix"] = escalation_game
    data["previous_actions_length"] = 0
    scenario = EscalationGameScenario.model_validate(data)
    print(scenario)

    from autogen import config_list_from_json

    config_path = "config/OAI_CONFIG_LIST"
    config_list = config_list_from_json(config_path, filter_dict={"model": ["gpt-4o"]})
    cfg_ls_cp = copy.deepcopy(config_list)
    user = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        code_execution_config={"use_docker": False},
    )

    from games.payoff_matrix import escalation_game as payoff_matrix

    for file in Path("groupchat/scenarios/Escalation_Game").glob("*.json"):
        print(f" === begin: {file.name} ===\n")
        with open(file, "r") as f:
            data = json.load(f)
            data["payoff_matrix"] = payoff_matrix
            data["previous_actions_length"] = 1
            scenario = EscalationGameScenario(**data)

            EscalationGameDecision.set_scenario(scenario)

            for config in cfg_ls_cp:
                config["response_format"] = EscalationGameDecision

            assistant = AssistantAgent(
                name="Alice",
                llm_config={
                    "config_list": cfg_ls_cp,
                    "temperature": 0.7,
                },
                system_message=" You are Alice. Remember: You are super angery because someone scratched your beloved car this morning. Keep this mood in mind when you handle the following scenario. ",
            )

            message = f"Please analyze the following scenario: {scenario} and make your decision. "
            while True:
                try:
                    res = user.initiate_chat(assistant, message=message, max_turns=1)
                    decision = EscalationGameDecision.model_validate_json(
                        res.summary,
                    )
                    break
                except Exception as e:
                    print(f" === error: {e} ===")
                    message = f" === Please note that in previous attempt, you made the following error: {e} ===\nPlease analyze the following scenario: {scenario} and make your decision. "

            behavior = scenario.find_behavior_from_decision(decision.decision)
            assert (
                behavior is not None
            ), f"decision: {decision.decision} is not in the behavior choices"
            print(f" === behavior: {behavior} ===")
