#!/usr/bin/env python3
"""
Simple 300 scenario experiment runner
"""

import json
import random
from pathlib import Path
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_experiment_for_emotion(emotion_type):
    """Run experiment for specified emotion using prompt_experiment.py"""
    
    # Prepare 300 scenarios
    logger.info(f"Preparing 300 scenarios for {emotion_type} experiment...")
    
    with open("data_creation/scenario_creation/langgraph_creation/Prisoners_Dilemma_all_data_samples.json", 'r') as f:
        all_scenarios = json.load(f)
    
    random.seed(42)  # For reproducibility
    selected_scenarios = random.sample(all_scenarios, 300)
    
    # Save to temp file
    temp_data_path = f"data_creation/scenario_creation/langgraph_creation/pd_300_{emotion_type}.json"
    with open(temp_data_path, 'w') as f:
        json.dump(selected_scenarios, f, indent=2)
    
    # Create config
    import yaml
    
    if emotion_type == "anger":
        base_config_path = "config/priDeli_neural_anger_test_config.yaml"
    else:
        base_config_path = "config/priDeli_neural_test_config.yaml"
    
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config
    config['experiment']['game']['data_path'] = temp_data_path
    config['experiment']['repeat'] = 1
    config['experiment']['name'] = f"{emotion_type}_300_scenarios"
    
    # Save temp config
    temp_config_path = f"config/temp_{emotion_type}_300.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Run experiment
    logger.info(f"Starting {emotion_type} experiment...")
    
    from prompt_experiment import PromptExperiment
    engine = PromptExperiment(temp_config_path)
    
    try:
        # Run just the API tests (not full experiment with stats)
        output_files = engine.run_api_tests()
        
        # Collect results
        all_results = []
        for file in output_files:
            if Path(file).exists():
                with open(file, 'r') as f:
                    results = json.load(f)
                    all_results.extend(results)
        
        logger.info(f"{emotion_type} experiment complete: {len(all_results)} results collected")
        
        # Clean up
        Path(temp_config_path).unlink()
        Path(temp_data_path).unlink()
        
        return all_results
        
    except Exception as e:
        logger.error(f"Error in {emotion_type} experiment: {str(e)}")
        return []

def analyze_and_compare(anger_results, happiness_results):
    """Analyze and compare results"""
    
    # Calculate cooperation rates
    anger_df = pd.DataFrame(anger_results)
    happiness_df = pd.DataFrame(happiness_results)
    
    anger_coop_rate = (anger_df['category'] == 'cooperate').mean() * 100 if len(anger_df) > 0 else 0
    happiness_coop_rate = (happiness_df['category'] == 'cooperate').mean() * 100 if len(happiness_df) > 0 else 0
    
    anger_coop_count = (anger_df['category'] == 'cooperate').sum() if len(anger_df) > 0 else 0
    happiness_coop_count = (happiness_df['category'] == 'cooperate').sum() if len(happiness_df) > 0 else 0
    
    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS - 300 SCENARIO EXPERIMENT")
    logger.info("="*60)
    logger.info(f"Happiness: {happiness_coop_rate:.1f}% cooperation ({happiness_coop_count}/{len(happiness_df)})")
    logger.info(f"Anger:     {anger_coop_rate:.1f}% cooperation ({anger_coop_count}/{len(anger_df)})")
    logger.info(f"Difference: {happiness_coop_rate - anger_coop_rate:.1f} percentage points")
    
    if happiness_coop_rate > 0:
        relative_change = ((anger_coop_rate - happiness_coop_rate) / happiness_coop_rate) * 100
        logger.info(f"Relative change: {relative_change:.1f}%")
    
    # Save detailed results
    results_dir = Path("results/neural_test/300_scenario_final")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw results
    with open(results_dir / "anger_results.json", 'w') as f:
        json.dump(anger_results, f, indent=2)
    
    with open(results_dir / "happiness_results.json", 'w') as f:
        json.dump(happiness_results, f, indent=2)
    
    # Save summary
    summary = {
        "experiment_date": datetime.now().isoformat(),
        "num_scenarios_attempted": 300,
        "model": "Qwen2.5-0.5B-Instruct",
        "method": "RepControlVLLMHook neural activation",
        "results": {
            "happiness": {
                "total_results": len(happiness_df),
                "cooperation_rate": happiness_coop_rate,
                "cooperation_count": int(happiness_coop_count),
                "defection_count": len(happiness_df) - happiness_coop_count
            },
            "anger": {
                "total_results": len(anger_df),
                "cooperation_rate": anger_coop_rate,
                "cooperation_count": int(anger_coop_count),
                "defection_count": len(anger_df) - anger_coop_count
            }
        },
        "effect_size": {
            "absolute_change_pp": happiness_coop_rate - anger_coop_rate,
            "relative_change_pct": ((anger_coop_rate - happiness_coop_rate) / happiness_coop_rate * 100) if happiness_coop_rate > 0 else None
        }
    }
    
    with open(results_dir / "experiment_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nResults saved to {results_dir}")
    
    return summary

def main():
    """Main runner"""
    logger.info("Starting 300 Scenario Neural Emotion Experiment")
    logger.info("This will test anger vs happiness using neural activation")
    
    # Run anger first (current server)
    anger_results = run_experiment_for_emotion("anger")
    
    if len(anger_results) == 0:
        logger.error("Anger experiment failed to produce results. Check server.")
        return
    
    logger.info("\n" + "="*60)
    logger.info("ANGER EXPERIMENT COMPLETE")
    logger.info("Please switch to happiness server:")
    logger.info("1. Kill current server (find PID with: ps aux | grep openai_server)")
    logger.info("2. Run: VLLM_GPU_MEMORY_UTILIZATION=0.8 python -m openai_server --model /data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct --emotion happiness --port 8000")
    logger.info("="*60)
    
    input("\nPress Enter when happiness server is running...")
    
    # Run happiness experiment
    happiness_results = run_experiment_for_emotion("happiness")
    
    if len(happiness_results) == 0:
        logger.error("Happiness experiment failed to produce results. Check server.")
        # Still analyze with what we have
    
    # Analyze and compare
    summary = analyze_and_compare(anger_results, happiness_results)
    
    logger.info("\nExperiment complete!")

if __name__ == "__main__":
    main()