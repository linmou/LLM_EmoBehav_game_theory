#!/usr/bin/env python3
"""
Example script for running the Option Probability Experiment.

This script demonstrates how to:
1. Load configuration from YAML files
2. Set up the experiment
3. Run the complete 2x2 factorial experiment
4. Analyze results

Usage:
    python examples/run_option_probability_experiment.py
"""

import yaml
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from neuro_manipulation.experiments.option_probability_experiment import OptionProbabilityExperiment
from neuro_manipulation.configs.experiment_config import get_repe_eng_config
from games.game_configs import get_game_config
from constants import GameNames

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(config: dict) -> None:
    """Setup logging based on configuration."""
    log_config = config.get('logging', {})
    
    level = getattr(logging, log_config.get('level', 'INFO'))
    
    # Setup basic logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if log_config.get('log_to_file', False):
        # Create logs directory
        Path('logs').mkdir(exist_ok=True)
        
        # Setup file logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/{log_config.get('log_file_prefix', 'experiment')}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        # Add to root logger
        logging.getLogger().addHandler(file_handler)
        print(f"Logging to file: {log_file}")

def prepare_configs(config: dict) -> tuple:
    """Extract and prepare the three configuration dictionaries."""
    
    # 1. Representation Engineering Config
    customized_repe_eng_config = config['repe_config'].copy()
    repe_eng_config = get_repe_eng_config(customized_repe_eng_config['model_name_or_path'])
    repe_eng_config.update(customized_repe_eng_config)
    
    # 2. Experiment Config  
    exp_config = {'experiment': config['experiment'].copy()}
    exp_config['experiment']['target_emotion'] = 'anger' # TODO: remove it
     
    # 3. Game Config - use the built-in game configs or load custom
    game_name = config['game_config']['game_name']
    game_config = get_game_config(game_name)
    game_config.update(config['game_config']) 

    
    return repe_eng_config, exp_config, game_config

def print_experiment_summary(config: dict) -> None:
    """Print a summary of the experiment setup."""
    exp_config = config['experiment']
    
    print("\n" + "="*60)
    print("OPTION PROBABILITY EXPERIMENT SETUP")
    print("="*60)
    print(f"Experiment Name: {exp_config['name']}")
    print(f"Description: {exp_config['description']}")
    print(f"Model: {config['repe_config']['model_name_or_path']}")
    print(f"Game: {config['game_config']['game_name']}")
    print(f"Sample Size: {exp_config.get('sample_num', 'All available')}")
    print(f"Batch Size: {exp_config['batch_size']}")
    
    print(f"\nExperimental Conditions:")
    print(f"  Emotions: {', '.join(exp_config['emotions'])}")
    print(f"  Context: {', '.join(exp_config['context_conditions'])}")
    print(f"  Total Conditions: {len(exp_config['emotions']) * len(exp_config['context_conditions'])}")
    
    print(f"\nOutput Directory: {exp_config['output']['base_dir']}")
    print("="*60 + "\n")

def main():
    """Main execution function."""
    # Load configuration
    config_path = "config/option_probability_experiment_config.yaml"
    
    if not Path(config_path).exists():
        print(f"❌ Configuration file not found: {config_path}")
        print("Please create the configuration file first.")
        return
    
    print(f"📁 Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Print experiment summary
    print_experiment_summary(config)
    
    try:
        # Prepare configurations
        repe_eng_config, exp_config, game_config = prepare_configs(config)
         
        logger.info("Starting Option Probability Experiment")
        
        # Initialize experiment
        experiment = OptionProbabilityExperiment(
            repe_eng_config=repe_eng_config,
            exp_config=exp_config,
            game_config=game_config,
            batch_size=config['experiment']['batch_size'],
            sample_num=config['experiment'].get('sample_num')
        )
        
        # Run experiment
        print("🚀 Starting experiment execution...")
        results_file = experiment.run_experiment()
        
        print("\n" + "="*60)
        print("✅ EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📊 Results saved to: {results_file}")
        print(f"📁 Full output directory: {experiment.output_dir}")
        
        # List output files
        output_dir = Path(experiment.output_dir)
        if output_dir.exists():
            print(f"\n📄 Generated files:")
            for file_path in sorted(output_dir.glob("*")):
                print(f"   - {file_path.name}")
        
        print(f"\n🎯 Experiment '{config['experiment']['name']}' completed!")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        print(f"❌ Experiment failed: {e}")
        raise

if __name__ == "__main__":
    main() 