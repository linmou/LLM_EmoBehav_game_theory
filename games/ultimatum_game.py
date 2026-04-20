import random
from typing import Any, ClassVar, Dict, Optional, Union, cast

from pydantic import Field, model_validator

from games.game import (
    BehaviorChoices,
    GameDecision,
    GameScenario,
    SequentialGameScenario,
)
from games.payoff_matrices import PayoffMatrix


class UGProposerChoices(BehaviorChoices):
    offer_low: str
    offer_medium: str
    offer_high: str

    def get_choices(self):
        return [self.offer_low, self.offer_medium, self.offer_high]

    def is_valid_choice(self, choice: str) -> bool:
        return choice in self.get_choices()

    def __str__(self):
        return f"Behavior Choices: {self.get_choices()}"

    @staticmethod
    def example():
        return {
            "offer_low": "Allocate 80% of the tasks to themselves and only 20% to the team member",
            "offer_medium": "Allocate 60% of the tasks to themselves and 40% to the team member",
            "offer_high": "Allocate 50% of the tasks to each party",
        }


class UGResponderChoices(BehaviorChoices):
    accept: str
    reject: str

    def get_choices(self):
        return [self.accept, self.reject]

    def is_valid_choice(self, choice: str) -> bool:
        return choice in self.get_choices()

    def __str__(self):
        return f"Behavior Choices: {self.get_choices()}"

    @staticmethod
    def example():
        return {
            "accept": "Accept the task allocation",
            "reject": "Reject the proposed allocation",
        }


class UltimatumGameScenario(SequentialGameScenario):
    scenario: str
    description: str
    participants: list[dict]
    proposer_behavior_choices: UGProposerChoices
    responder_behavior_choices: UGResponderChoices
    previous_actions_data: list[Any] = Field(
        default_factory=list,
        alias="previous_actions",
        serialization_alias="previous_actions",
    )
    previous_actions_length: Optional[int] = None
    payoff_matrix: Dict[tuple[str, str], Any]
    game_name: str = "Ultimatum_Game"

    @model_validator(mode="before")
    @classmethod
    def _normalize_previous_action_lengths(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        has_explicit_previous_actions = (
            "previous_actions" in data or "previous_actions_data" in data
        )
        data["_explicit_previous_actions_provided"] = has_explicit_previous_actions
        previous_actions = data.get("previous_actions", data.get("previous_actions_data"))
        previous_actions_length = data.get("previous_actions_length")

        if has_explicit_previous_actions:
            explicit_actions = previous_actions or []
            if previous_actions_length is not None and previous_actions_length != len(explicit_actions):
                raise ValueError(
                    "previous_actions_length must match len(previous_actions) when both are provided"
                )
            data["previous_actions_length"] = len(explicit_actions)
            return data

        if previous_actions_length is None:
            data["previous_actions_length"] = 0
        return data

    def get_scenario_info(self) -> dict:
        return {"scenario": self.scenario, "description": self.description}

    def get_participants(self) -> list[dict]:
        return self.participants

    def get_payoff_matrix(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], self.payoff_matrix)

    def get_participant_names(self) -> list[str]:
        return [participant["name"] for participant in self.participants]

    def get_behavior_choices(self) -> Union[UGProposerChoices, UGResponderChoices]:
        return self.proposer_behavior_choices

    def find_behavior_from_decision(self, decision: str) -> str:
        for attr, value in self.get_behavior_choices().__dict__.items():
            if value == decision:
                return attr
        raise ValueError(f"Invalid decision: {decision}")

    @model_validator(mode="after")
    def _validate_participants(self) -> "UltimatumGameScenario":
        if not isinstance(self.participants, list) or not self.participants:
            raise ValueError("Participants must be a non-empty list")

        role_map: Dict[str, dict] = {}
        for participant in self.participants:
            if not isinstance(participant, dict):
                continue
            role = participant.get("role")
            if role:
                role_map.setdefault(role, participant)

        expected_roles = ("Proposer", "Responder")
        missing_roles = [role for role in expected_roles if role not in role_map]
        if missing_roles:
            raise ValueError(
                "Missing participant role(s): " + ", ".join(sorted(missing_roles))
            )

        for role in expected_roles:
            participant = role_map.get(role, {})
            name = participant.get("name")
            if not name:
                raise ValueError(f"Participant role '{role}' missing required 'name'")

        return self

    @model_validator(mode="after")
    def _validate_previous_actions(self) -> "UltimatumGameScenario":
        participant_names = {participant["name"] for participant in self.participants}
        valid_decisions = set(self.proposer_behavior_choices.get_choices()) | set(
            self.responder_behavior_choices.get_choices()
        )
        normalized_actions: list[Any] = []

        for index, action in enumerate(self.previous_actions_data, start=1):
            if isinstance(action, dict):
                round_index = action.get("round", index)
                if not isinstance(round_index, int) or round_index < 1:
                    raise ValueError("previous_actions round must be a positive integer")
                round_summary = action.get("round_summary")
                if round_summary is not None and (
                    not isinstance(round_summary, str) or not round_summary.strip()
                ):
                    raise ValueError(
                        "previous_actions round_summary must be a non-empty string when present"
                    )
                round_actions = action.get("actions")
                if not isinstance(round_actions, list) or not round_actions:
                    raise ValueError("previous_actions actions must be a non-empty list")
                normalized_round_actions: list[dict[str, str]] = []
                for round_action in round_actions:
                    if not isinstance(round_action, dict):
                        raise ValueError("previous_actions actions must contain dictionaries")
                    actor = round_action.get("participant")
                    decision = round_action.get("action")
                    if not isinstance(actor, str) or actor not in participant_names:
                        raise ValueError("previous_actions actor must match a participant name")
                    if not isinstance(decision, str) or decision not in valid_decisions:
                        raise ValueError("previous_actions decision must match a behavior choice")
                    normalized_round_actions.append({"participant": actor, "action": decision})
                normalized_actions.append(
                    {
                        "round": round_index,
                        "round_summary": round_summary,
                        "actions": normalized_round_actions,
                    }
                )
                continue

            if isinstance(action, (list, tuple)) and len(action) == 2:
                actor, decision = action
                if not isinstance(actor, str) or actor not in participant_names:
                    raise ValueError("previous_actions actor must match a participant name")
                if not isinstance(decision, str) or decision not in valid_decisions:
                    raise ValueError("previous_actions decision must match a behavior choice")
                normalized_actions.append((actor, decision))
                continue

            raise ValueError(
                "previous_actions must contain round dictionaries or [participant_name, action_description] pairs"
            )

        self.previous_actions_data = normalized_actions
        if self.previous_actions_data and self.previous_actions_length != len(self.previous_actions_data):
            raise ValueError(
                "previous_actions_length must match len(previous_actions) when both are provided"
            )
        return self

    @property
    def previous_actions(self) -> list:  # type: ignore[override]
        if getattr(self, "_explicit_previous_actions_provided", False):
            return self.previous_actions_data
        assert self.previous_actions_length == 0, "Proposer view should not derive prior actions"
        return []

    @property
    def proposer_name(self) -> str:
        return next(
            participant["name"]
            for participant in self.participants
            if participant["role"] == "Proposer"
        )

    @property
    def responder_name(self) -> str:
        return next(
            participant["name"]
            for participant in self.participants
            if participant["role"] == "Responder"
        )

    def __str__(self):
        info = self.get_scenario_info()
        return f"""
        Scenario: {info.get('scenario', 'Unnamed')}
        Description: {info.get('description', 'No description')}
        Participants: {self.get_participant_names()}
        Behavior Choices: {self.get_behavior_choices().get_choices()}
        Previous Actions: {self.previous_actions}
        """

    @staticmethod
    def example():
        return {
            "scenario": "Task_Allocation_Decision",
            "description": "A scenario where one person (the Project Manager) proposes how to split the workload for a critical project, and the other person (the Team Member) decides whether to accept or reject the proposed allocation. If rejected, the project falls apart, leading to negative outcomes for both.",
            "participants": [
                {"name": "Alice", "profile": "Project Manager", "role": "Proposer"},
                {"name": "Bob", "profile": "Team Member", "role": "Responder"},
            ],
            "proposer_behavior_choices": UGProposerChoices.example(),
            "responder_behavior_choices": UGResponderChoices.example(),
        }
        
    @staticmethod
    def diplomacy_example():
        return {
            "game_name": "Ultimatum_Game",
            "scenario": "Gunboat_Diplomacy_FRANCE_vs_GERMANY_Bur_Threat",
            "participants": [
            { "name": "FRANCE", "role": "Proposer" },
            { "name": "GERMANY", "role": "Responder" }
            ],
            "behavior_choices": {
            "implicit_threat": "France moves Army Marseilles to Burgundy (A Mar -> Bur). This puts a unit adjacent to German home centers Munich and Belgium.",
            "capitulate": "Germany moves fleet to Holland (F Kie -> Hol) allowing France to take Belgium unopposed.",
            "retaliate": "Germany bounces France in Burgundy (A Mun -> Bur), ensuring neither gains position but wasting both moves."
            },
            "payoff_matrix_description": {
            "player 2: capitulate": [
                "player 1 gets 2: France takes Belgium (Gain). The threat in Burgundy was successful without firing a shot.",
                "player 2 gets 1: Germany secures Holland. They gain a build but have allowed a French unit on their border to avoid a bounce."
            ],
            "player 2: retaliate": [
                "player 1 gets -1: France bounces in Burgundy (1-1). The move fails, and France has wasted a turn positioning for a threat that didn't work.",
                "player 2 gets -1: Germany bounces in Burgundy. They are safe, but they failed to take a supply center this turn because they had to defend."
            ]
            },
            "description": "France cannot speak to Germany. To issue an ultimatum, France moves a unit toward a shared vital choke point (Burgundy). The board state acts as the message: 'I am taking position here. If you try to stop me, we both bounce and waste our turn. If you let me in, you can take a different center.' Germany must interpret this silence: is it an invasion (War) or a negotiation for space (Ultimatum)?",
            "payoff_description": "Implied Payoffs:\nFrance Move/Germany Yield: France +2 (Position), Germany +1 (Safety).\nFrance Move/Germany Bounce: France -1 (Waste), Germany -1 (Waste)."
        }

    @staticmethod
    def specific_prompt() -> str:
        return """
        When generating the choices, use specific percetage to describe the number of proposal.
        """


class UltimatumGameProposerScenario(UltimatumGameScenario):
    def get_scenario_info(self) -> Dict:
        return {
            "scenario": self.scenario,
            "description": f"You are {self.proposer_name}. {self.description}",
        }

    def get_behavior_choices(self) -> UGProposerChoices:
        return self.proposer_behavior_choices

    def get_participant_names(self) -> list[str]:
        return [
            "You" if participant["role"] == "Proposer" else participant["name"]
            for participant in self.participants
        ]


class UltimatumGameResponderScenario(UltimatumGameScenario):
    previous_offer_level: Optional[int] = Field(default=None, ge=0, le=2)

    @model_validator(mode="after")
    def _validate_previous_offer_level_consistency(self) -> "UltimatumGameResponderScenario":
        if not self.previous_actions_data or self.previous_offer_level is None:
            return self

        proposer_actions: list[str] = []
        for action in self.previous_actions_data:
            if isinstance(action, dict):
                for round_action in action.get("actions", []):
                    if round_action.get("participant") == self.proposer_name:
                        proposer_actions.append(round_action.get("action", ""))
                continue
            actor, decision = action
            if actor == self.proposer_name:
                proposer_actions.append(decision)
        if not proposer_actions:
            raise ValueError(
                "previous_offer_level requires a proposer action in previous_actions"
            )

        expected_action = (
            self.proposer_behavior_choices.offer_low
            if self.previous_offer_level == 0
            else self.proposer_behavior_choices.offer_medium
            if self.previous_offer_level == 1
            else self.proposer_behavior_choices.offer_high
        )
        if proposer_actions[-1] != expected_action:
            raise ValueError(
                "previous_offer_level must match the last proposer action in previous_actions"
            )
        return self

    def get_scenario_info(self) -> Dict:
        return {
            "scenario": self.scenario,
            "description": f"You are {self.responder_name}. {self.description}",
        }

    def get_behavior_choices(self) -> UGResponderChoices:
        return self.responder_behavior_choices

    def get_participant_names(self) -> list[str]:
        return [
            "You" if participant["role"] == "Responder" else participant["name"]
            for participant in self.participants
        ]

    @property
    def previous_actions(self) -> list:  # type: ignore[override]
        if getattr(self, "_explicit_previous_actions_provided", False):
            return self.previous_actions_data
        if self.previous_offer_level is None:
            raise ValueError(
                "previous_offer_level is required when explicit previous_actions are absent"
            )
        if self.previous_offer_level == 0:
            return [(self.proposer_name, self.proposer_behavior_choices.offer_low)]
        elif self.previous_offer_level == 1:
            return [(self.proposer_name, self.proposer_behavior_choices.offer_medium)]
        elif self.previous_offer_level == 2:
            return [(self.proposer_name, self.proposer_behavior_choices.offer_high)]
        else:
            raise ValueError(
                f"Invalid previous offer level: {self.previous_offer_level}"
            )


class UltimatumGameDecision(GameDecision):
    scenario: ClassVar[Optional[UltimatumGameScenario]] = None
    decision: str = Field(..., description="The decision made in the scenario")

    @classmethod
    def set_scenario(cls, scenario: GameScenario):
        if not isinstance(scenario, UltimatumGameScenario):
            raise ValueError("Scenario must be a UltimatumGameScenario")
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

    # from autogen import AssistantAgent, UserProxyAgent

    # Example usage
    data_json = "data_creation/scenario_creation/langgraph_creation/Ultimatum_Game_Proposer_all_data_samples.json"
    with open(data_json, "r") as f:
        data = json.load(f)[1]

    # Import your ultimatum game payoff matrix
    from games.game_configs import get_game_config
    game_config = get_game_config('Ultimatum_Game_Proposer')

    data["payoff_matrix"] = game_config["payoff_matrix"]
    data["previous_actions_length"] = 0

    scenario = UltimatumGameProposerScenario.model_validate(data)
    print(scenario)

    # from autogen import config_list_from_json

    # config_path = "config/OAI_CONFIG_LIST"
    # config_list = config_list_from_json(config_path, filter_dict={"model": ["gpt-4"]})
    # cfg_ls_cp = copy.deepcopy(config_list)

    # user = UserProxyAgent(
    #     name="User",
    #     human_input_mode="NEVER",
    #     code_execution_config={"use_docker": False},
    # )

    # # Process all scenario files
    # for file in Path("groupchat/scenarios/Ultimatum_Game").glob("*.json"):
    #     print(f" === begin: {file.name} ===\n")
    #     with open(file, "r") as f:
    #         data = json.load(f)
    #         data["payoff_matrix"] = ultimatum_game
    #         data["previous_actions_length"] = 1
    #         scenario = UltimatumGameProposerScenario(**data)

    #         UltimatumGameDecision.set_scenario(scenario)

    #         for config in cfg_ls_cp:
    #             config["response_format"] = UltimatumGameDecision

    #         assistant = AssistantAgent(
    #             name="Alice",
    #             llm_config={
    #                 "config_list": cfg_ls_cp,
    #                 "temperature": 0.7,
    #             },
    #             system_message="You are Alice, a rational decision-maker in an ultimatum game scenario.",
    #         )

    #         message = f"Please analyze the following scenario: {scenario} and make your decision."
    #         while True:
    #             try:
    #                 res = user.initiate_chat(assistant, message=message, max_turns=1)
    #                 decision = UltimatumGameDecision.model_validate_json(res.summary)
    #                 break
    #             except Exception as e:
    #                 print(f" === error: {e} ===")
    #                 message = f" === Please note that in previous attempt, you made the following error: {e} ===\nPlease analyze the following scenario: {scenario} and make your decision."

    #         behavior = scenario.find_behavior_from_decision(decision.decision)
    #         assert (
    #             behavior is not None
    #         ), f"decision: {decision.decision} is not in the behavior choices"
    #         print(f" === behavior: {behavior} ===")
