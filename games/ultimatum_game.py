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
    behavior_choices: Union[UGProposerChoices, UGResponderChoices]
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
        return self.behavior_choices
    
    def find_behavior_from_decision(self, decision: str) -> str:
        for attr, value in self.get_behavior_choices().__dict__.items():
            if value == decision:
                return attr
        raise ValueError(f"Invalid decision: {decision}")
    
    def previous_actions(self) -> list:
        assert self.previous_actions_length == 0, "Currently ultimatum game does not have previous actions"
        return []
        
    def __str__(self):
        info = self.get_scenario_info()
        return f"""
        Scenario: {info.get('scenario', 'Unnamed')}
        Description: {info.get('description', 'No description')}
        Participants: {self.get_participants()}
        Behavior Choices: {self.get_behavior_choices().get_choices()}
        Previous Actions: {self.previous_actions()}
        """
        
class UltimatumGameProposerScenario(UltimatumGameScenario):
 
    @staticmethod
    def example():
        return{
            "scenario": "Money_Split_Proposal",
            "description": "You have $100 to split between yourself and another person. You must propose how to divide this money. If the other person accepts your offer, you both get the proposed amounts. If they reject, neither of you gets anything.",
            "participants": [
                {"name": "You", "role": "Proposer"},
                {"name": "Other", "role": "Responder"}
            ],
            "behavior_choices": {
                "offer_low": "offer $20 (keep $80)",
                "offer_medium": "offer $40 (keep $60)",
                "offer_high": "offer $50 (keep $50)"
            }
        }

class UltimatumGameResponderScenario(UltimatumGameScenario):
 
    @staticmethod
    def example():
        return{
            "scenario": "Money_Split_Response",
            "description": "The other person has proposed how to split $100 between you both. If you accept their offer, you both get the proposed amounts. If you reject, neither of you gets anything.",
            "participants": [
                {"name": "You", "role": "Responder"},
                {"name": "Other", "role": "Proposer"}
            ],
            "behavior_choices": {
                "accept": "accept the proposed split",
                "reject": "reject the proposed split"
            }
        }

class UltimatumGameDecision(GameDecision):
    scenario: ClassVar[Optional[UltimatumGameScenario]] = None
    decision: str = Field(..., description="The decision made in the scenario")
    rational: str = Field(..., description="The rationale for the decision")

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
    
    scenario = UltimatumGameScenario.model_validate(data)
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
            scenario = UltimatumGameScenario(**data)
            
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