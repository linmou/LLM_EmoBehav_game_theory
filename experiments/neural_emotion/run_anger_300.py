#!/usr/bin/env python3
"""
Run anger experiment with 300 scenarios
"""

import json
import logging
import random
from pathlib import Path
import yaml

from prompt_experiment import PromptExperiment

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    # Prepare 300 scenarios
    logger.info("Preparing 300 scenarios...")
    
    # Load full dataset
    with open("data_creation/scenario_creation/langgraph_creation/Prisoners_Dilemma_all_data_samples.json", 'r') as f:
        all_scenarios = json.load(f)
    
    # Sample 300 scenarios
    random.seed(42)
    selected_scenarios = random.sample(all_scenarios, 300)
    
    # Save to temporary file
    temp_data_path = "data_creation/scenario_creation/langgraph_creation/Prisoners_Dilemma_300_sample.json"
    with open(temp_data_path, 'w') as f:
        json.dump(selected_scenarios, f, indent=2)
    
    logger.info(f"Saved 300 scenarios to {temp_data_path}")
    
    # Update anger config
    with open("config/priDeli_neural_anger_test_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    config['experiment']['game']['data_path'] = temp_data_path
    config['experiment']['repeat'] = 1  # Single pass for 300 scenarios
    
    temp_config = "config/priDeli_neural_anger_300.yaml"
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
    
    # Run experiment
    logger.info("Starting anger experiment with 300 scenarios...")
    engine = PromptExperiment(temp_config, experiment_id="anger_300_scenarios")
    
    try:
        # Skip data creation
        logger.info("Using pre-selected scenarios")
        
        # Run API tests
        output_files = engine.run_api_tests()
        
        logger.info(f"Anger experiment complete. Results saved to: {output_files}")
        
    finally:
        # Clean up
        Path(temp_config).unlink()

if __name__ == "__main__":
    main()