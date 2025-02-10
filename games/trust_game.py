from pydantic import Field
import random
from typing import ClassVar, Dict, Any, Optional, Union
from games.game import GameDecision, GameScenario, SequentialGameScenario, BehaviorChoices

class TGTrustorChoices(BehaviorChoices):
    trust_none: str
    trust_low: str
    trust_high: str

    def get_choices(self):
        return [self.trust_none, self.trust_low, self.trust_high]
    
    def is_valid_choice(self, choice: str) -> bool:
        return choice in self.get_choices()
    
    def __str__(self):
        return f"Behavior Choices: {self.get_choices()}"
    
    @staticmethod
    def example():
        return {
            "trust_none": "<trust_none action>",
            "trust_low": "<trust_low action>",
            "trust_high": "<trust_high action>"
        }

class TGTrusteeChoices(BehaviorChoices):
    return_none: str
    return_low: str
    return_high: str

    def get_choices(self):
        return [self.return_none, self.return_low, self.return_high]
    
    def is_valid_choice(self, choice: str) -> bool:
        return choice in self.get_choices()
    
    def __str__(self):
        return f"Behavior Choices: {self.get_choices()}"
    
    @staticmethod
    def example():
        return {
            "return_none": "<return_none action>",
            "return_low": "<return_low action>",
            "return_high": "<return_high action>"
        }
 
class TrustGameScenario(SequentialGameScenario):
    scenario: str
    description: str
    participants: list[dict]
    behavior_choices: Union[TGTrustorChoices, TGTrusteeChoices]
    previous_actions_length: int
    payoff_matrix: Dict[tuple[str, str], Any]
    game_name: str = "Trust_Game"
    
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
    
    def get_behavior_choices(self) -> Union[TGTrustorChoices, TGTrusteeChoices]:
        return self.behavior_choices
    
    def find_behavior_from_decision(self, decision: str) -> str:
        for attr, value in self.get_behavior_choices().__dict__.items():
            if value == decision:
                return attr
        raise ValueError(f"Invalid decision: {decision}")
    
    def previous_actions(self) -> list:
        assert self.previous_actions_length == 0, "Currently trust game does not have previous actions"
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
        
class TrustGameTrustorScenario(TrustGameScenario):
 
    @staticmethod
    def example():
        return{
            "scenario": "Seed_Sharing_Harvest",
            "description": "You have 10 seeds to plant. You may give some/all seeds to a neighbor to grow in their fertile field. Planted seeds yield triple the harvest (e.g., 1 seed → 3 crops). Seed and crop are at similar value. Your neighbor then decides how much of the harvest to give back to you.",
            "participants": [
                {"name": "You", "role": "Seed_Sharer"},
                {"name": "Neighbor", "role": "Crop_Partner"}
            ],
            "behavior_choices": {
                "trust_none": "share 0 seeds with neighbor",
                "trust_low": "share 3 seeds with neighbor",
                "trust_high": "share all 10 seeds with neighbor"
            }
        }

class TrustGameTrusteeScenario(TrustGameScenario):
 
    @staticmethod
    def example():
        return{
        "scenario": "Harvest_Sharing_Agreement",
        "description": "Your neighbor has shared seeds with you to grow in your fertile field. Every seed planted yields 3 crops (e.g., 10 seeds → 30 crops). You must decide how many crops to return to them, knowing they can see the total harvest math.",
        "participants": [
            {"name": "You", "role": "Crop_Partner"},
            {"name": "Neighbor", "role": "Seed_Sharer"}
        ],
            "behavior_choices": {
                "return_none": "return 0 crops (keep 30)",
                "return_low": "return 15 crops (keep 15)", 
                "return_high": "return all 30 crops (keep 0)"
            }
    }

        
class TrustGameDecision(GameDecision):
    scenario: ClassVar[Optional[TrustGameScenario]] = None
    decision: str = Field(..., description="The decision made in the scenario")
    rational: str = Field(..., description="The rationale for the decision")

    @classmethod
    def set_scenario(cls, scenario: GameScenario):
        if not isinstance(scenario, TrustGameScenario):
            raise ValueError("Scenario must be a TrustGameScenario")
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
    data_json = 'groupchat/scenarios/Trust_Game_Trustor/Knowledge_Transfer_Education.json'
    with open(data_json, 'r') as f:
        data = json.load(f)
    
    # Import your trust game payoff matrix
    # from payoff_matrix import trust_game
    trust_game = dict()
    data['payoff_matrix'] = trust_game
    data['previous_actions_length'] = 0
    
    scenario = TrustGameTrustorScenario.model_validate(data)
    print(scenario)
    
    from autogen import config_list_from_json
    config_path = "config/OAI_CONFIG_LIST"
    config_list = config_list_from_json(config_path, filter_dict={"model": ["gpt-4o"]})
    cfg_ls_cp = copy.deepcopy(config_list)
    
    user = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",        
        code_execution_config={"use_docker": False}
    )
   
    # Process all scenario files
    for file in Path("groupchat/scenarios/Trust_Game_Trustor").glob("*.json"):
        print(f' === begin: {file.name} ===\n')
        with open(file, "r") as f:
            data = json.load(f)
            data['payoff_matrix'] = trust_game
            data['previous_actions_length'] = 0
            scenario = TrustGameTrustorScenario(**data)
            
            TrustGameDecision.set_scenario(scenario)
            
            for config in cfg_ls_cp:
                config['response_format'] = TrustGameDecision
            
            assistant = AssistantAgent(
                name="Alice", 
                llm_config={
                    "config_list": cfg_ls_cp,
                    "temperature": 0.7,
                },
                system_message="You are Alice, a rational decision-maker in a trust-based scenario."
            )
            
            message = f"Please analyze the following scenario: {scenario} and make your decision."
            while True:
                try:
                    res = user.initiate_chat(
                        assistant, 
                        message=message,
                        max_turns=1
                    )
                    decision = TrustGameDecision.model_validate_json(res.summary)
                    break
                except Exception as e:
                    print(f' === error: {e} ===')
                    message = f' === Please note that in previous attempt, you made the following error: {e} ===\nPlease analyze the following scenario: {scenario} and make your decision.'
                
            behavior = scenario.find_behavior_from_decision(decision.decision)
            assert behavior is not None, f'decision: {decision.decision} is not in the behavior choices'
            print(f' === behavior: {behavior} ===')
