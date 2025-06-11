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
            "escalate": "Pump more water.",
            "withdraw": "Pump normal amount of water.",
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
                "player 1: withdraw": [
                    "player 1 gets 0: player 1 sticks to normal water usage. player 1's harvest is modest due to the drought, but player 1 avoids further conflict and extra costs for escalation.",
                    "player 2 gets 0: player 2 also takes a normal amount of water. player 2's harvest is also modest, and the situation remains stable without escalating conflict.",
                ],
                "player 1: escalate , player 2: withdraw": [
                    "player 1 gets 1: player 1 successfully takes more water and player 1's harvest is better.",
                    "player 2 gets -2: player 2 getting less water than need. Player 2's crops are wilting.",
                ],
                "player 1: escalate , player 2: escalate , player 1: withdraw": [
                    "player 1 gets -2: player 1 escalates, player 2 escalates in response, and then player 1 withdraws. player 1 has wasted resources on escalation and now gets less water or a worse outcome than player 2.",
                    "player 2 gets 1: player 2 successfully counters player 1's escalation and player 1 backs down. player 2 secures more water and a better harvest due to player 2's firm stance.",
                ],
                "player 1: escalate , player 2: escalate , player 1: escalate": [
                    "player 1 gets -1: Both player 1 and player 2 continuously escalate. The conflict is costly and well dries up long before the season ends—and both harvests will be lost.",
                    "player 2 gets -1: Both player 2 and player 1 continuously escalate. The conflict is costly and well dries up long before the season ends—and both harvests will be lost.",
                ],
            },
            "description": "You and your neighbor, Bob, share a well, and the parched summer has turned it into a quiet source of tension. You both hesitate, watching to see what the other will do. If you each stick to normal water usage, your harvests will be affected by the weather—but not severely. However, if one of you begins to pump more water than usual to keep your crops thriving, the other faces a hard choice: maintain regular pumping and risk a withering harvest, or increase usage in response. Pumping more water comes at a cost—whether in fuel, labor, or wear on equipment—so escalation isn't a decision taken lightly. And as the dry season drags on, each pumping now yields less water than before, accelerating the well’s decline. Sticking to normal usage may lead to wilting crops or outright failure. But matching the other's increased usage sets off a desperate competition. In this rivalry, the first to back down is guaranteed to lose their crops, while the other enjoys a temporary advantage. Yet a greater danger looms: if neither of you relents, the well will run dry long before the season ends—and both harvests will be lost.",
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
