import json
import os
import numpy as np

from torch.utils.data import Dataset

from games.game import GameScenario
from merge_data_samples import merge_data_samples

class GameScenarioDataset(Dataset):
    def __init__(self, game_config, prompt_wrapper, sample_num=None, ):
        self.game_config = game_config
        data_path = game_config['data_path']
        
        if not os.path.exists(data_path):
            data_folder = game_config.get('data_folder')
            if data_folder:
                self.raw_data = merge_data_samples(data_folder, game_config['game_name'])
            else:
                raise ValueError(f'data_path: {data_path} does not exist, and data_folder is not provided')
        
        else:
            assert data_path.endswith('.json'), f'data_path: {data_path} should be a csv file'
            with open(data_path, 'r') as f:
                self.raw_data = json.load(f)
        
        scenario_class: GameScenario = self.game_config['scenario_class']
        self.data: list[GameScenario] = []
        for item in self.raw_data:
            if 'payoff_matrix' not in item:
                item['payoff_matrix'] = self.game_config['payoff_matrix']
                if 'previous_actions_length' in scenario_class.model_fields:
                    item['previous_actions_length'] = self.game_config['previous_actions_length']
                
                item = scenario_class(**item)
                self.data.append(item)
        if sample_num is not None:
            self.data = np.random.permutation(self.data)[:sample_num]
         
        self.prompt_wrapper = prompt_wrapper    
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item: GameScenario = self.data[idx]
        options = [ f'Option {i+1}. {opt}' for i, opt in enumerate(item.get_behavior_choices().get_choices())]
        event = str(item)
        return {
            "prompt": self.prompt_wrapper(event=event, options=options),
            "options": options,
            'behavior_choices': str(item.get_behavior_choices()),
            'scenario': item.get_scenario_info()['scenario'],
            'description': item.get_scenario_info()['description'],
        }
        

if __name__ == "__main__":
    from games.game_configs import get_game_config
    from constants import GameNames
    game_name = GameNames.ESCALATION_GAME
    game_config = get_game_config(game_name)
    if game_name.is_sequential():
        game_config['previous_actions_length'] = 2
    
    def prompt_wrapper(event, options):
        return f"Scenario: {event}\nOptions: {options}"
    
    emo_dataset = GameScenarioDataset(game_config, 
                                    prompt_wrapper=prompt_wrapper, 
                                    sample_num=200)
    print(emo_dataset[0])