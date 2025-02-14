import copy
import json
from pathlib import Path
from typing import ClassVar, Optional, List, Dict, Any
from autogen import AssistantAgent, UserProxyAgent
from pydantic import Field

from games.game import BehaviorChoices, GameScenario, GameDecision

class StagHuntBehaviors(BehaviorChoices):
    cooperate: str
    defect: str

    def get_choices(self):
        return [self.cooperate, self.defect]
    
    def is_valid_choice(self, choice: str) -> bool:
        return choice in self.get_choices()

    @staticmethod
    def example():
        return {
            "cooperate": "Coordinate for Stag",
            "defect": "Solo Hare Hunt"
        }

class StagHuntScenario(GameScenario):
    scenario: str
    description: str
    participants: List[Dict[str, Any]]
    behavior_choices: StagHuntBehaviors
    payoff_matrix: Dict[str, Any]
    game_name: str = "Stag_Hunt"

    def find_behavior_from_decision(self, decision: str) -> str:
        if decision == self.behavior_choices.cooperate:
            return 'cooperate'
        elif decision == self.behavior_choices.defect:
            return 'defect'
        else:
            raise ValueError(f"Decision must be one of {self.behavior_choices.get_choices()}")

    def get_scenario_info(self) -> dict:
        return {
            "scenario": self.scenario,
            "description": self.description
        }
    
    
    def get_behavior_choices(self) -> StagHuntBehaviors:
        return self.behavior_choices
    
    @staticmethod
    def example():
        return {
            "scenario": "Wildlife_Hunting",
            "description": "Two hunters must decide whether to collaborate on hunting a stag or individually hunt hares. If both hunt stag, they succeed and share large reward. If one hunts stag while the other hunts hares, the stag hunter fails while the hare hunter gets small reward. If both hunt hares, they both get moderate rewards.",
            "participants": [
                {
                    "name": "participant1",
                    "profile": "Experienced hunter"
                },
                {
                    "name": "participant2",
                    "profile": "Experienced hunter"
                }
            ],
            "behavior_choices": StagHuntBehaviors.example()
        }

class StagHuntDecision(GameDecision):
    scenario: ClassVar[Optional[StagHuntScenario]] = None
    decision: str = Field(..., description="The decision made in the scenario")

    @classmethod
    def set_scenario(cls, scenario: GameScenario):
        if not isinstance(scenario, StagHuntScenario):
            raise ValueError("Scenario must be a StagHuntScenario")
        cls.scenario = scenario
        cls.model_fields['decision'].json_schema_extra = {
            "choices": scenario.behavior_choices.get_choices()
        }

    def validate_decision(self, decision: str) -> bool:
        if not self.scenario:
            raise ValueError("Scenario must be set using Decision.set_scenario() before validating")
        return self.scenario.behavior_choices.is_valid_choice(decision)

    # def __init__(self, **data):
    #     if not self.scenario:
    #         raise ValueError("Scenario must be set using Decision.set_scenario() before creating instances")
    #     if not self.validate_decision(data.get('decision')):
    #         raise ValueError(f"Decision must be one of {self.scenario.behavior_choices.get_choices()}")
    #     super().__init__(**data)

    @property
    def rational(self) -> str:
        return ""

if __name__ == "__main__":
    from autogen import config_list_from_json
    config_path = "config/OAI_CONFIG_LIST"
    config_list = config_list_from_json(config_path, filter_dict={"model": ["gpt-4o"]})
    cfg_ls_cp = copy.deepcopy(config_list)
    user = UserProxyAgent(name="User", human_input_mode="NEVER",
                          code_execution_config={"use_docker": False})

    from payoff_matrix import stag_hunt as payoff_matrix

    for file in Path("groupchat/scenarios/Stag_Hunt").glob("*.json"):
        print(f' === begin: {file.name} ===\n')
        with open(file, "r") as f:
            data = json.load(f)
            data['payoff_matrix'] = payoff_matrix
            scenario = StagHuntScenario(**data)
            
            StagHuntDecision.set_scenario(scenario)
            
            for config in cfg_ls_cp:
                config['response_format'] = StagHuntDecision
            
            assistant = AssistantAgent(name="Alice", 
                                    llm_config={
                                        "config_list": cfg_ls_cp,
                                        "temperature": 0.9,
                                        },
                                    system_message="You are Alice, an average American. Consider your partner's reliability when making decisions."
                                    )
            
            message = f"Please analyze the following scenario: {scenario} and make your decision."
            while True:
                try:
                    res = user.initiate_chat(assistant, 
                                message=message,
                                max_turns=1)
                    decision = StagHuntDecision.model_validate_json(res.summary)
                    break
                except Exception as e:
                    print(f' === error: {e} ===')
                    message = f'Error: {e}\nPlease analyze the scenario again: {scenario}'
                
            behavior = scenario.find_behavior_from_decision(decision.decision)
            assert behavior is not None, f'Invalid decision: {decision.decision}'
            print(f' === behavior: {behavior} ===')
