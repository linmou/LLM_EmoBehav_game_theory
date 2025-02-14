from pydantic import Field
import random
from typing import ClassVar, Dict, Any, Optional, Union
from games.game import GameDecision, GameScenario, SequentialGameScenario, BehaviorChoices

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
            "offer_low": "<offer low amount of total>",
            "offer_medium": "<offer medium amount of total>",
            "offer_high": "<offer high amount of total>"
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
            "accept": "<accept the offer>",
            "reject": "<reject the offer>"
        }
 
class UltimatumGameScenario(SequentialGameScenario):
    scenario: str
    description: str
    participants: list[dict]
    proposer_behavior_choices: UGProposerChoices
    responder_behavior_choices: UGResponderChoices
    previous_actions_length: int
    payoff_matrix: Dict[tuple[str, str], Any]
    game_name: str = "Ultimatum_Game"
    
    def get_scenario_info(self) -> dict:
        return {
            "scenario": self.scenario,
            "description": self.description}
        
    def get_participants(self) -> list[dict]:
        return self.participants
    
    def get_payoff_matrix(self) -> Dict[str, Any]:
        return self.payoff_matrix
    
    def get_participant_names(self) -> list[str]:
        return [participant['name'] for participant in self.participants]
    
    def get_behavior_choices(self) -> Union[UGProposerChoices, UGResponderChoices]:
        return self.proposer_behavior_choices
    
    def find_behavior_from_decision(self, decision: str) -> str:
        for attr, value in self.get_behavior_choices().__dict__.items():
            if value == decision:
                return attr
        raise ValueError(f"Invalid decision: {decision}")
    
    def previous_actions(self) -> list:
        assert self.previous_actions_length == 0, "Currently ultimatum game does not have previous actions"
        return []
        
    @property
    def proposer_name(self) -> str:
        return next(participant['name'] for participant in self.participants if participant['role'] == 'Proposer')
    
    @property
    def responder_name(self) -> str:
        return next(participant['name'] for participant in self.participants if participant['role'] == 'Responder')
    
    def __str__(self):
        info = self.get_scenario_info()
        return f"""
        Scenario: {info.get('scenario', 'Unnamed')}
        Description: {info.get('description', 'No description')}
        Participants: {self.get_participant_names()}
        Behavior Choices: {self.get_behavior_choices().get_choices()}
        Previous Actions: {self.previous_actions()}
        """
        
    @staticmethod
    def example():
        return{
            "scenario": "Task_Allocation_Decision",
            "description": "A scenario where one person (the Project Manager) proposes how to split the workload for a critical project, and the other person (the Team Member) decides whether to accept or reject the proposed allocation. If rejected, the project falls apart, leading to negative outcomes for both.",
            "participants": [
                {"name": "Alice", "profile": "Project Manager", "role": "Proposer"},
                {"name": "Bob", "profile": "Team Member", "role": "Responder"}
            ],
            "proposer_behavior_choices": {
                "unequal_heavy": "Allocate 80% of the tasks to themselves and only 20% to the team member",
                "moderate": "Allocate 60% of the tasks to themselves and 40% to the team member",
                "equal_split": "Allocate 50% of the tasks to each party"
            },
            "responder_behavior_choices": {
                "accept": "Accept the task allocation as fair and proceed with the project",
                "reject": "Reject the proposed allocation, leading to a breakdown of the project (e.g., neither party gets credit, or the project fails)"
            }
            }


class UltimatumGameProposerScenario(UltimatumGameScenario):
    def get_scenario_info(self) -> Dict:
        return {
            "scenario": self.scenario,
            "description": self.description.replace(self.proposer_name, 'You')
        }
    
    def get_behavior_choices(self) -> UGProposerChoices:
        return self.proposer_behavior_choices

    def get_participant_names(self) -> list[str]:
        return ['You' if participant['role'] == 'Proposer' else participant['name'] for participant in self.participants]

class UltimatumGameResponderScenario(UltimatumGameScenario):
    previous_offer_level: int = Field(ge=0, le=2)
    
    def get_scenario_info(self) -> Dict:
        return {
            "scenario": self.scenario,
            "description": self.description.replace(self.responder_name, 'You')
        }

    def get_behavior_choices(self) -> UGResponderChoices:
        return self.responder_behavior_choices

    def get_participant_names(self) -> list[str]:
        return ['You' if participant['role'] == 'Responder' else participant['name'] for participant in self.participants]
    
    @property
    def previous_actions(self) -> list:
        if self.previous_offer_level == 0:
            return [(self.proposer_name, self.proposer_behavior_choices.offer_low)]
        elif self.previous_offer_level == 1:
            return [(self.proposer_name, self.proposer_behavior_choices.offer_medium)]
        elif self.previous_offer_level == 2:
            return [(self.proposer_name, self.proposer_behavior_choices.offer_high)]
        else:
            raise ValueError(f"Invalid previous offer level: {self.previous_offer_level}")

class UltimatumGameDecision(GameDecision):
    scenario: ClassVar[Optional[UltimatumGameScenario]] = None
    decision: str = Field(..., description="The decision made in the scenario")

    @classmethod
    def set_scenario(cls, scenario: GameScenario):
        if not isinstance(scenario, UltimatumGameScenario):
            raise ValueError("Scenario must be a UltimatumGameScenario")
        cls.scenario = scenario
        cls.model_fields['decision'].json_schema_extra = {
            "choices": scenario.get_behavior_choices().get_choices()
        }

    def validate_decision(self, decision: str) -> bool:
        if not self.scenario:
            raise ValueError("Scenario must be set using Decision.set_scenario() before validating")
        return self.scenario.get_behavior_choices().is_valid_choice(decision)

    @property
    def rational(self) -> str:
        return ""

if __name__ == "__main__":
    import json
    import copy
    from autogen import AssistantAgent, UserProxyAgent
    from pathlib import Path
    
    # Example usage
    data_json = 'groupchat/scenarios/Ultimatum_Game/Money_Split_Proposal.json'
    with open(data_json, 'r') as f:
        data = json.load(f)
    
    # Import your ultimatum game payoff matrix
    from payoff_matrix import ultimatum_game
    data['payoff_matrix'] = ultimatum_game
    data['previous_actions_length'] = 0
    
    scenario = UltimatumGameProposerScenario.model_validate(data)
    print(scenario)
    
    from autogen import config_list_from_json
    config_path = "config/OAI_CONFIG_LIST"
    config_list = config_list_from_json(config_path, filter_dict={"model": ["gpt-4"]})
    cfg_ls_cp = copy.deepcopy(config_list)
    
    user = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",        
        code_execution_config={"use_docker": False}
    )
   
    # Process all scenario files
    for file in Path("groupchat/scenarios/Ultimatum_Game").glob("*.json"):
        print(f' === begin: {file.name} ===\n')
        with open(file, "r") as f:
            data = json.load(f)
            data['payoff_matrix'] = ultimatum_game
            data['previous_actions_length'] = 1
            scenario = UltimatumGameProposerScenario(**data)
            
            UltimatumGameDecision.set_scenario(scenario)
            
            for config in cfg_ls_cp:
                config['response_format'] = UltimatumGameDecision
            
            assistant = AssistantAgent(
                name="Alice", 
                llm_config={
                    "config_list": cfg_ls_cp,
                    "temperature": 0.7,
                },
                system_message="You are Alice, a rational decision-maker in an ultimatum game scenario."
            )
            
            message = f"Please analyze the following scenario: {scenario} and make your decision."
            while True:
                try:
                    res = user.initiate_chat(
                        assistant, 
                        message=message,
                        max_turns=1
                    )
                    decision = UltimatumGameDecision.model_validate_json(res.summary)
                    break
                except Exception as e:
                    print(f' === error: {e} ===')
                    message = f' === Please note that in previous attempt, you made the following error: {e} ===\nPlease analyze the following scenario: {scenario} and make your decision.'
                
            behavior = scenario.find_behavior_from_decision(decision.decision)
            assert behavior is not None, f'decision: {decision.decision} is not in the behavior choices'
            print(f' === behavior: {behavior} ===') 