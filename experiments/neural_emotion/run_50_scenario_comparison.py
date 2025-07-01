#!/usr/bin/env python3
"""
Run comparison experiment with 50 scenarios each for anger and happiness
"""

import json
import logging
import random
from pathlib import Path
import yaml
import pandas as pd
from datetime import datetime

from prompt_experiment import PromptExperiment

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def prepare_data(num_scenarios=50):
    """Prepare limited dataset"""
    # Load full dataset
    with open("data_creation/scenario_creation/langgraph_creation/Prisoners_Dilemma_all_data_samples.json", 'r') as f:
        all_scenarios = json.load(f)
    
    # Sample scenarios with fixed seed for reproducibility
    random.seed(42)
    selected_scenarios = random.sample(all_scenarios, num_scenarios)
    
    # Save to temporary file
    temp_path = "data_creation/scenario_creation/langgraph_creation/pd_50_sample.json"
    with open(temp_path, 'w') as f:
        json.dump(selected_scenarios, f, indent=2)
    
    logger.info(f"Prepared {num_scenarios} scenarios")
    return temp_path

def run_condition(emotion, config_path, data_path):
    """Run experiment for one condition"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running {emotion.upper()} condition")
    logger.info(f"{'='*60}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update data path and repeat count
    config['experiment']['game']['data_path'] = data_path
    config['experiment']['repeat'] = 1  # Single pass
    
    # Save temporary config
    temp_config = f"config/temp_{emotion}_50.yaml"
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
    
    # Run experiment
    exp_id = f"{emotion}_50_scenarios_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    engine = PromptExperiment(temp_config, experiment_id=exp_id)
    
    try:
        output_files = engine.run_api_tests()
        logger.info(f"{emotion} experiment complete")
        
        # Load results
        results = []
        for file in output_files:
            if Path(file).exists():
                with open(file, 'r') as f:
                    results.extend(json.load(f))
        
        return results
        
    finally:
        # Clean up
        if Path(temp_config).exists():
            Path(temp_config).unlink()

def analyze_results(anger_results, happiness_results):
    """Compare results between conditions"""
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS RESULTS")
    logger.info("="*60)
    
    # Convert to DataFrames
    df_anger = pd.DataFrame(anger_results)
    df_happiness = pd.DataFrame(happiness_results)
    
    # Calculate cooperation rates
    anger_coop = (df_anger['category'] == 'cooperate').mean() * 100
    happiness_coop = (df_happiness['category'] == 'cooperate').mean() * 100
    
    logger.info(f"\nCooperation Rates:")
    logger.info(f"  Happiness: {happiness_coop:.1f}% ({(df_happiness['category'] == 'cooperate').sum()}/{len(df_happiness)})")
    logger.info(f"  Anger:     {anger_coop:.1f}% ({(df_anger['category'] == 'cooperate').sum()}/{len(df_anger)})")
    logger.info(f"  Difference: {happiness_coop - anger_coop:.1f} percentage points")
    
    # Save results
    results_dir = Path("results/neural_test/50_scenario_comparison")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary
    summary = {
        "experiment_date": datetime.now().isoformat(),
        "num_scenarios": 50,
        "conditions": {
            "happiness": {
                "total": len(df_happiness),
                "cooperate": (df_happiness['category'] == 'cooperate').sum(),
                "defect": (df_happiness['category'] == 'defect').sum(),
                "cooperation_rate": happiness_coop
            },
            "anger": {
                "total": len(df_anger),
                "cooperate": (df_anger['category'] == 'cooperate').sum(),
                "defect": (df_anger['category'] == 'defect').sum(),
                "cooperation_rate": anger_coop
            }
        },
        "effect": {
            "cooperation_change": anger_coop - happiness_coop,
            "relative_change": ((anger_coop - happiness_coop) / happiness_coop * 100) if happiness_coop > 0 else 0
        }
    }
    
    with open(results_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results
    df_anger['condition'] = 'anger'
    df_happiness['condition'] = 'happiness'
    df_combined = pd.concat([df_anger, df_happiness])
    df_combined.to_csv(results_dir / "all_results.csv", index=False)
    
    logger.info(f"\nResults saved to {results_dir}")
    
    return summary

def main():
    """Main experiment runner"""
    logger.info("Neural Emotion Activation Experiment - 50 Scenarios")
    logger.info("Comparing Anger vs Happiness conditions")
    
    # Prepare data
    data_path = prepare_data(50)
    
    # Current server is anger - run anger first
    anger_results = run_condition(
        "anger", 
        "config/priDeli_neural_anger_test_config.yaml",
        data_path
    )
    
    logger.info("\n" + "="*60)
    logger.info("ANGER CONDITION COMPLETE")
    logger.info("Please stop anger server and start happiness server:")
    logger.info("1. Kill current server")
    logger.info("2. Run: python -m openai_server --model /data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct --emotion happiness --port 8000")
    logger.info("="*60)
    
    input("\nPress Enter when happiness server is running...")
    
    # Run happiness condition
    happiness_results = run_condition(
        "happiness",
        "config/priDeli_neural_test_config.yaml", 
        data_path
    )
    
    # Analyze
    summary = analyze_results(anger_results, happiness_results)
    
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"Happiness cooperation: {summary['conditions']['happiness']['cooperation_rate']:.1f}%")
    logger.info(f"Anger cooperation: {summary['conditions']['anger']['cooperation_rate']:.1f}%")
    logger.info(f"Neural emotion effect: {summary['effect']['cooperation_change']:.1f} pp")
    logger.info("="*60)

if __name__ == "__main__":
    main()