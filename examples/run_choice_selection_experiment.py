import argparse
import yaml
import logging
from pathlib import Path
import sys

# Add project root to path for imports, so we can find the neuro_manipulation module
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from neuro_manipulation.configs.experiment_config import get_repe_eng_config
from neuro_manipulation.experiments.choice_selection_experiment import (
    ChoiceSelectionExperiment,
)
from games.game_configs import get_game_config

def prepare_configs(config: dict) -> tuple:
    """Extract and prepare the three configuration dictionaries."""
    
    # 1. Representation Engineering Config
    customized_repe_eng_config = config['repe_config'].copy()
    repe_eng_config = get_repe_eng_config(customized_repe_eng_config['model_name_or_path'])
    repe_eng_config.update(customized_repe_eng_config)
    
    # 2. Experiment Config
    exp_config = config['experiment'].copy()
     
    # 3. Game Config - use the built-in game configs or load custom
    game_name = config['game_config']['game_name']
    game_config = get_game_config(game_name)
    game_config.update(config['game_config']) 

    
    return repe_eng_config, exp_config, game_config
def main():
    """Main function to run the choice selection experiment."""
    parser = argparse.ArgumentParser(description="Run a choice selection experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/choice_selection_experiment_config.yaml",
        help="Path to the experiment configuration YAML file.",
    )
    args = parser.parse_args()

    # Load configuration from YAML file
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        sys.exit(1)

    # The ChoiceSelectionExperiment class handles its own logging setup.
    # We can configure a basic logger here for messages from this script itself.
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Extract configurations from the loaded YAML
    # repe_config = config.get("repe_config")
    # exp_config = config.get("experiment")
    # game_config_from_yaml = config.get("game_config")

    # if not all([repe_config, exp_config, game_config_from_yaml]):
    #     logger.error(
    #         "Configuration file must contain 'repe_config', 'experiment', and 'game_config' sections."
    #     )
    #     sys.exit(1)

    # # Get the full game configuration from game_configs.py based on the name
    # game_name = game_config_from_yaml.get("game_name")
    # game_config = get_game_config(game_name)

    # game_config.update(game_config_from_yaml)

    repe_eng_config, exp_config, game_config = prepare_configs(config)


    # Extract batch_size and sample_num for the experiment class
    batch_size = exp_config.get("batch_size", 4)
    sample_num = exp_config.get("sample_num")

    # Initialize and run the experiment
    logger.info("Initializing ChoiceSelectionExperiment...")
    try:
        experiment = ChoiceSelectionExperiment(
            repe_eng_config=repe_eng_config,
            exp_config=exp_config,
            game_config=game_config,
            batch_size=batch_size,
            sample_num=sample_num,
        )

        logger.info("Running experiment...")
        results_file = experiment.run_experiment()
        logger.info(f"Experiment finished successfully. Results are at: {results_file}")

    except Exception as e:
        logger.error(f"An error occurred during the experiment: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
