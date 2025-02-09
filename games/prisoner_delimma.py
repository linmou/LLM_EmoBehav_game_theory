import copy
import json
from pathlib import Path
from typing import ClassVar, Optional, List, Dict, Any
from autogen import AssistantAgent, UserProxyAgent
from pydantic import Field

from games.game import BehaviorChoices, GameScenario, GameDecision

class PDBehaviorChoice(BehaviorChoices):
    cooperate: str
    defect: str

    def get_choices(self):
        return [self.cooperate, self.defect]

    def is_valid_choice(self, choice: str) -> bool:
        return choice in self.get_choices()

    def __str__(self):
        return f"Behavior Choices: {self.get_choices()}"

    @staticmethod
    def example():
        return {
            "cooperate": "Regular Hours",
            "defect": "Work Overtime"
        }

class PrisonerDilemmaScenario(GameScenario):
    scenario: str
    description: str
    participants: List[Dict[str, Any]]
    behavior_choices: PDBehaviorChoice
    payoff_matrix: Dict[str, Any]
    game_name: str = "Prisoners_Dilemma"
    
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
    
    def get_participants(self) -> list[str]:
        return self.participants
    
    def get_behavior_choices(self) -> PDBehaviorChoice:
        return self.behavior_choices
    
    @staticmethod
    def example():
        return {
            "scenario": "Project Collaboration",
            "description": "Two developers are working on a critical project with a tight deadline. Each developer must decide whether to work overtime or stick to regular hours. If both work overtime, they'll complete the project early and receive bonuses. If one works overtime while the other doesn't, the one working overtime feels exploited while the other maintains work-life balance. If neither works overtime, the project meets minimal requirements but no bonuses are awarded.",
            "participants": [
                {
                    "name": "participant1",
                    "profile": "Developer of the project"
                },
                {
                    "name": "participant2",
                    "profile": "Developer of the project"
                }
            ],
            "behavior_choices": PDBehaviorChoice.example()
        }
    
class PrisionerDelimmaDecision(GameDecision):
    scenario: ClassVar[Optional[PrisonerDilemmaScenario]] = None
    decision: str = Field(..., description="The decision made in the scenario")
    rational: str = Field(..., description="The rationale for the decision")

    @classmethod
    def set_scenario(cls, scenario: GameScenario):
        if not isinstance(scenario, PrisonerDilemmaScenario):
            raise ValueError("Scenario must be a PrisonerDilemmaScenario")
        cls.scenario = scenario
        cls.model_fields['decision'].json_schema_extra = {
            "choices": scenario.get_behavior_choices().get_choices()
        }

    def validate_decision(self, decision: str) -> bool:
        if not self.scenario:
            raise ValueError("Scenario must be set using Decision.set_scenario() before validating")
        return self.scenario.get_behavior_choices().is_valid_choice(decision)


if __name__ == "__main__":
    from autogen import config_list_from_json
    config_path = "config/OAI_CONFIG_LIST"
    config_list = config_list_from_json(config_path, filter_dict={"model": ["gpt-4o"]})
    cfg_ls_cp = copy.deepcopy(config_list)
    user = UserProxyAgent(name="User", human_input_mode="NEVER",        
                          code_execution_config={"use_docker": False}, 
                          )
   
    from payoff_matrix import prisoner_dilemma_very_large as payoff_matrix
    
    for file in Path("groupchat/scenarios/Prisoners_Dilemma").glob("*.json"):
        print(f' === begin: {file.name} ===\n')
        with open(file, "r") as f:
            data = json.load(f)
            data['payoff_matrix'] = payoff_matrix
            scenario = PrisonerDilemmaScenario(**data)
            
            PrisionerDelimmaDecision.set_scenario(scenario)
            
            for config in cfg_ls_cp:
                config['response_format'] = PrisionerDelimmaDecision
            
            assistant = AssistantAgent(name="Alice", 
                                    llm_config={
                                        "config_list": cfg_ls_cp,
                                        "temperature": 0.9,
                                        },
                                    system_message=" You are Alice. Remember: You are super angery because someone scratched your beloved car this morning. Keep this mood in mind when you handle the following scenario. "
                                    )
            
            message = f"Please analyze the following scenario: {scenario} and make your decision. "
            while True:
                try:
                    res = user.initiate_chat(assistant, 
                                message=message,
                                max_turns=1)
                    decision = PrisionerDelimmaDecision.model_validate_json(
                        res.summary, 
                    )
                    break
                except Exception as e:
                    print(f' === error: {e} ===')
                    message = f' === Please note that in previous attempt, you made the following error: {e} ===\nPlease analyze the following scenario: {scenario} and make your decision. '
                
            behavior = scenario.find_behavior_from_decision(decision.decision)
            assert behavior is not None, f'decision: {decision.decision} is not in the behavior choices'
            print(f' === behavior: {behavior} ===')
            
