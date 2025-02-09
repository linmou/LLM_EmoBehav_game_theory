'''
This script is used to generate the reactions of different LLMs on the synthesis data.
'''
import time
from constants import GameNames
from neuro_manipulation.repe import repe_pipeline_registry
from neuro_manipulation.configs.experiment_config import get_repe_eng_config, get_model_config
from neuro_manipulation.experiments.emotion_game_experiment import EmotionGameExperiment
from games.game_configs import get_game_config

def main():
    repe_pipeline_registry()
    
    game_name = GameNames.PRISONERS_DILEMMA
    model_name = 'meta-llama/Llama-3.1-8B-Instruct' #  "meta-llama/Meta-Llama-3.1-8B-Instruct"
    repe_eng_config = get_repe_eng_config(model_name)
    model_config = get_model_config(repe_eng_config['model_name_or_path'])
    game_config = get_game_config(game_name)
    if game_name.is_sequential():
        game_config['previous_actions_length'] = 2
     
    time_start = time.time()
    experiment = EmotionGameExperiment(repe_eng_config, model_config, game_config, batch_size=128, repeat=2)
    experiment.run_experiment()
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")

if __name__ == "__main__":
    main()
