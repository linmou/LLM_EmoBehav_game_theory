'''
This script is used to generate the reactions of different LLMs on the synthesis data.
'''
import time
from constants import GameNames
from neuro_manipulation.repe import repe_pipeline_registry
from neuro_manipulation.configs.experiment_config import get_exp_config, get_repe_eng_config, get_model_config
from neuro_manipulation.experiments.emotion_game_experiment import EmotionGameExperiment
from games.game_configs import get_game_config

def setup_experiment():
    repe_pipeline_registry()
    
    exp_config = get_exp_config('config/priDeli_repEng_experiment_config.yaml') 
    game_name = GameNames.from_string(exp_config['experiment']['game']['name'])
    model_name = exp_config['experiment']['llm']['model_name'] #  "meta-llama/Meta-Llama-3.1-8B-Instruct"
    repe_eng_config = get_repe_eng_config(model_name)
    game_config = get_game_config(game_name)
    if game_name.is_sequential():
        game_config['previous_actions_length'] = exp_config['experiment']['game']['previous_actions_length']
     
    experiment = EmotionGameExperiment(repe_eng_config, exp_config, game_config, batch_size=128, repeat=exp_config['experiment']['repeat'])
    return experiment

def main():
    experiment = setup_experiment()
    time_start = time.time()
    experiment.run_experiment()
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")

def run_sanity_check():
    experiment = setup_experiment()
    time_start = time.time()
    experiment.run_sanity_check()
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")

if __name__ == "__main__":
    # main()
    run_sanity_check()
