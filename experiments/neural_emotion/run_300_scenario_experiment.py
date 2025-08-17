#!/usr/bin/env python3
"""
Run neural emotion activation experiment with 300 scenarios
Compares happiness (baseline) vs anger activation
"""

import json
import logging
import random
from pathlib import Path
import pandas as pd
from datetime import datetime
import time

from prompt_experiment import PromptExperiment

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def prepare_limited_data(num_scenarios=300):
    """Extract 300 random scenarios from the full dataset"""
    logger.info(f"Preparing {num_scenarios} scenarios for experiment...")
    
    # Load full dataset
    full_data_path = "data_creation/scenario_creation/langgraph_creation/Prisoners_Dilemma_all_data_samples.json"
    with open(full_data_path, 'r') as f:
        all_scenarios = json.load(f)
    
    logger.info(f"Total available scenarios: {len(all_scenarios)}")
    
    # Randomly sample scenarios
    random.seed(42)  # For reproducibility
    selected_scenarios = random.sample(all_scenarios, min(num_scenarios, len(all_scenarios)))
    
    # Save to temporary file
    temp_data_path = Path("data_creation/scenario_creation/langgraph_creation/Prisoners_Dilemma_300_sample.json")
    with open(temp_data_path, 'w') as f:
        json.dump(selected_scenarios, f, indent=2)
    
    logger.info(f"Saved {len(selected_scenarios)} scenarios to {temp_data_path}")
    return str(temp_data_path)

def run_neural_experiment(emotion_type, config_file, experiment_name):
    """Run experiment with specified emotion configuration"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running {emotion_type.upper()} Experiment")
    logger.info(f"{'='*60}")
    
    engine = PromptExperiment(config_file, experiment_id=experiment_name)
    
    try:
        # Skip data creation, use existing data
        logger.info("Using pre-selected 300 scenarios")
        
        # Run API tests
        output_files = engine.run_api_tests()
        
        # Don't run full statistical analysis yet (wait for both conditions)
        return output_files
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise

def analyze_results(happiness_files, anger_files):
    """Compare results between happiness and anger conditions"""
    logger.info("\n" + "="*60)
    logger.info("ANALYZING RESULTS")
    logger.info("="*60)
    
    # Load all results
    happiness_results = []
    for file in happiness_files:
        with open(file, 'r') as f:
            happiness_results.extend(json.load(f))
    
    anger_results = []
    for file in anger_files:
        with open(file, 'r') as f:
            anger_results.extend(json.load(f))
    
    # Create DataFrames
    df_happiness = pd.DataFrame(happiness_results)
    df_anger = pd.DataFrame(anger_results)
    
    # Calculate cooperation rates
    happiness_coop_rate = (df_happiness['category'] == 'cooperate').mean() * 100
    anger_coop_rate = (df_anger['category'] == 'cooperate').mean() * 100
    
    logger.info(f"Happiness Cooperation Rate: {happiness_coop_rate:.2f}%")
    logger.info(f"Anger Cooperation Rate: {anger_coop_rate:.2f}%")
    logger.info(f"Difference: {happiness_coop_rate - anger_coop_rate:.2f} percentage points")
    
    # Save detailed comparison
    comparison = {
        "experiment_date": datetime.now().isoformat(),
        "num_scenarios": 300,
        "conditions": {
            "happiness": {
                "total_responses": len(happiness_results),
                "cooperation_rate": happiness_coop_rate,
                "defection_rate": 100 - happiness_coop_rate,
                "cooperate_count": (df_happiness['category'] == 'cooperate').sum(),
                "defect_count": (df_happiness['category'] == 'defect').sum()
            },
            "anger": {
                "total_responses": len(anger_results),
                "cooperation_rate": anger_coop_rate,
                "defection_rate": 100 - anger_coop_rate,
                "cooperate_count": (df_anger['category'] == 'cooperate').sum(),
                "defect_count": (df_anger['category'] == 'defect').sum()
            }
        },
        "effect_size": {
            "cooperation_rate_change": anger_coop_rate - happiness_coop_rate,
            "relative_change": ((anger_coop_rate - happiness_coop_rate) / happiness_coop_rate) * 100
        }
    }
    
    results_dir = Path("results/neural_test/300_scenario_comparison")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "comparison_summary.json", 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Save combined results
    df_happiness['condition'] = 'happiness'
    df_anger['condition'] = 'anger'
    df_combined = pd.concat([df_happiness, df_anger])
    df_combined.to_csv(results_dir / "all_results.csv", index=False)
    
    logger.info(f"\nResults saved to {results_dir}")
    
    return comparison

def main():
    """Main experiment runner"""
    logger.info("Starting 300-Scenario Neural Emotion Activation Experiment")
    logger.info("Model: Qwen2.5-0.5B-Instruct")
    logger.info("Conditions: Happiness (baseline) vs Anger")
    
    # Prepare data
    temp_data_path = prepare_limited_data(300)
    
    # Update config files to use the temporary data
    import yaml
    
    # Update neutral config
    with open("config/priDeli_neural_test_config.yaml", 'r') as f:
        neutral_config = yaml.safe_load(f)
    neutral_config['experiment']['game']['data_path'] = temp_data_path
    
    temp_neutral_config = "config/priDeli_neural_test_300.yaml"
    with open(temp_neutral_config, 'w') as f:
        yaml.dump(neutral_config, f)
    
    # Update anger config
    with open("config/priDeli_neural_anger_test_config.yaml", 'r') as f:
        anger_config = yaml.safe_load(f)
    anger_config['experiment']['game']['data_path'] = temp_data_path
    
    temp_anger_config = "config/priDeli_neural_anger_test_300.yaml"
    with open(temp_anger_config, 'w') as f:
        yaml.dump(anger_config, f)
    
    try:
        # Check server status
        logger.info("\nChecking current server status...")
        import requests
        try:
            response = requests.get("http://localhost:8000/v1/models", timeout=5)
            logger.info("Server is running and responsive")
        except:
            logger.error("No server detected! Please start appropriate server before running.")
            return
        
        # Note: User should manually switch between servers due to VRAM constraints
        logger.info("\nIMPORTANT: This experiment requires manually switching between servers")
        logger.info("Due to VRAM constraints, we cannot run both servers simultaneously")
        
        input("\nPress Enter when HAPPINESS server is running on port 8000...")
        
        # Run happiness experiment
        happiness_files = run_neural_experiment(
            "happiness", 
            temp_neutral_config,
            f"happiness_300_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        logger.info("\n" + "="*60)
        logger.info("HAPPINESS EXPERIMENT COMPLETE")
        logger.info("Please stop the happiness server and start the anger server")
        logger.info("Command: python -m openai_server --model <path> --emotion anger --port 8000")
        logger.info("="*60)
        
        input("\nPress Enter when ANGER server is running on port 8000...")
        
        # Run anger experiment
        anger_files = run_neural_experiment(
            "anger",
            temp_anger_config, 
            f"anger_300_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Analyze combined results
        comparison = analyze_results(happiness_files, anger_files)
        
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT COMPLETE")
        logger.info(f"Happiness cooperation: {comparison['conditions']['happiness']['cooperation_rate']:.2f}%")
        logger.info(f"Anger cooperation: {comparison['conditions']['anger']['cooperation_rate']:.2f}%")
        logger.info(f"Effect size: {comparison['effect_size']['cooperation_rate_change']:.2f} percentage points")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise
    finally:
        # Clean up temporary files
        for temp_file in [temp_neutral_config, temp_anger_config]:
            if Path(temp_file).exists():
                Path(temp_file).unlink()

if __name__ == "__main__":
    main()