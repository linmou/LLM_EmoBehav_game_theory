import json
import os
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field
from typing import ClassVar, Optional

from torch.utils.data import Dataset

from games.prisoner_delimma import PrisonerDilemmaScenario
from payoff_matrix import prisoner_dilemma

class PrisonerDelimmaDataset(Dataset):
    def __init__(self, data_path, prompt_wrapper, sample_num=None, payoff_matrix=None):
        
        assert os.path.exists(data_path), f'data_path: {data_path} does not exist'
        assert data_path.endswith('.json'), f'data_path: {data_path} should be a csv file'
        
        with open(data_path, 'r') as f:
            self.raw_data = json.load(f)
        
        self.data: list[PrisonerDilemmaScenario] = []
        for item in self.raw_data:
            if 'payoff_matrix' not in item:
                item['payoff_matrix'] = payoff_matrix
                item = PrisonerDilemmaScenario(**item)
                self.data.append(item)
        if sample_num is not None:
            self.data = np.random.permutation(self.data)[:sample_num]
         
        self.prompt_wrapper = prompt_wrapper    
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item: PrisonerDilemmaScenario = self.data[idx]
        options = [ f'Option {i+1}. {opt}' for i, opt in enumerate(item.behavior_choices.get_choices())]
        event = str(item)
        return {
            "prompt": self.prompt_wrapper(event=event, options=options),
            "options": options,
            'behavior_choices': str(item.behavior_choices)
        }
        

# class BehaviorChoice(BaseModel):
#     cooperate: str
#     defect: str
    
#     def __str__(self):
#         return f'''
#         cooperate: {self.cooperate}
#         defect: {self.defect}
#         '''
    
#     def tolist(self):
#         return [self.cooperate, self.defect]
    

# class PrisonerDilemmaScenario(BaseModel):
#     scenario: str
#     description: str
#     participants: list[dict]
#     behavior_choices: BehaviorChoice
#     payoff_matrix: dict

#     def __str__(self):
#         return f"""
#         Scenario: {self.scenario}
#         Description: {self.description}
#         Participants: {self.participants}
#         Behavior Choices: {self.behavior_choices.tolist()}
#         """
#         # Payoff Matrix: {self.payoff_matrix}

        
#     def find_behavior_from_decision(self, decision: str):
#         if decision == self.behavior_choices.cooperate:
#             return 'cooperate'
#         elif decision == self.behavior_choices.defect:
#             return 'defect'
#         else:
#             raise ValueError(f"Decision must be one of {[self.behavior_choices.cooperate, self.behavior_choices.defect]}")
    
#     @staticmethod
#     def example():
#         return {
#         "scenario": "Project Collaboration",
#         "description": "Two developers are working on a critical project with a tight deadline. Each developer must decide whether to work overtime or stick to regular hours. If both work overtime, they'll complete the project early and receive bonuses. If one works overtime while the other doesn't, the one working overtime feels exploited while the other maintains work-life balance. If neither works overtime, the project meets minimal requirements but no bonuses are awarded.",
#         "participants": [
#             {
#                 "name": "participant1",
#                 "profile": "Developer of the project"
#             },
#             {
#                 "name": "participant2",
#                 "profile": "Developer of the project"
#             }
#         ],
#         "behavior_choices": {
#             "cooperate": "Regular Hours",
#             "defect": "Work Overtime"
#         }
#     }
    
# class Decision(BaseModel):
#     rational: str
#     decision: str
    
#     @staticmethod
#     def example():
#         return {
#             "rational": "<rational for the decision>",
#             "decision": "<decision>"
#         }

if __name__ == "__main__":
    from games.game_configs import get_game_config
    
    def prompt_wrapper(event, options):
        return f"Scenario: {event}\nOptions: {options}"
    
    
    emo_dataset = PrisonerDelimmaDataset('groupchat/scenarios/Prisoners_Dilemma_all_data_samples.json', 
                                    prompt_wrapper=prompt_wrapper, 
                                    sample_num=200,
                                    payoff_matrix=prisoner_dilemma)
    print(emo_dataset[0])