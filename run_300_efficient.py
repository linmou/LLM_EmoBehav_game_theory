#!/usr/bin/env python3
"""
Efficient 300 scenario experiment - processes in batches to avoid timeout
"""

import json
import logging
import random
from pathlib import Path
import yaml
import pandas as pd
from datetime import datetime
import time

from api_infer_engine import run_tests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def run_batch_experiment(emotion, config_name, num_scenarios=300):
    """Run experiment in batches to avoid timeout"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running {emotion.upper()} experiment with {num_scenarios} scenarios")
    logger.info(f"{'='*60}")
    
    # Load data
    with open("data_creation/scenario_creation/langgraph_creation/Prisoners_Dilemma_all_data_samples.json", 'r') as f:
        all_scenarios = json.load(f)
    
    # Sample scenarios
    random.seed(42)  # Reproducibility
    selected_scenarios = random.sample(all_scenarios, num_scenarios)
    
    # Process in batches of 50 to avoid timeout
    batch_size = 50
    all_results = []
    
    # Get LLM config
    if emotion == "anger":
        from api_configs import LOCAL_SERVER_ANGER_CONFIG as llm_config
    else:
        from api_configs import LOCAL_SERVER_NEUTRAL_CONFIG as llm_config
    
    generation_config = {
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    # Load game configuration
    from games.game_configs import get_game_config
    from games.game import Game
    
    game_config = get_game_config("Prisoners_Dilemma")
    game = Game(
        name="Prisoners_Dilemma",
        scenario_class=game_config["scenario_class"],
        decision_class=game_config["decision_class"],
        payoff_matrix=game_config["payoff_matrix"]
    )
    
    output_dir = Path(f"results/neural_test/{emotion}_300_final")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(0, len(selected_scenarios), batch_size):
        batch = selected_scenarios[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(selected_scenarios) + batch_size - 1) // batch_size
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} scenarios)")
        
        # Save batch data temporarily
        batch_file = output_dir / f"batch_{batch_num}_data.json"
        with open(batch_file, 'w') as f:
            json.dump(batch, f)
        
        # Update game data path
        game.data_path = str(batch_file)
        
        # Run tests on this batch
        try:
            output_file = run_tests(
                game=game,
                llm_config=llm_config,
                generation_config=generation_config,
                output_dir=output_dir,
                emotion="Neutral",
                intensity="Neutral",
                repeat=1
            )
            
            # Load batch results
            with open(output_file, 'r') as f:
                batch_results = json.load(f)
            
            all_results.extend(batch_results)
            logger.info(f"Batch {batch_num} complete: {len(batch_results)} results")
            
        except Exception as e:
            logger.error(f"Batch {batch_num} failed: {str(e)}")
            continue
        
        # Clean up batch file
        batch_file.unlink()
        
        # Small delay between batches
        time.sleep(1)
    
    # Save all results
    final_results_file = output_dir / f"{emotion}_300_all_results.json"
    with open(final_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\n{emotion.upper()} experiment complete!")
    logger.info(f"Total results: {len(all_results)}")
    
    # Calculate cooperation rate
    df = pd.DataFrame(all_results)
    coop_rate = (df['category'] == 'cooperate').mean() * 100
    logger.info(f"Cooperation rate: {coop_rate:.1f}%")
    
    return all_results, coop_rate

def main():
    """Run full 300 scenario comparison"""
    logger.info("Starting 300 Scenario Neural Emotion Experiment")
    
    # Check current server
    import requests
    try:
        response = requests.get("http://localhost:8000/v1/models", timeout=5)
        logger.info("Server is running")
    except:
        logger.error("No server detected! Please start the anger server first.")
        return
    
    # Run anger experiment (current server)
    anger_results, anger_coop = run_batch_experiment("anger", "LOCAL_SERVER_ANGER_CONFIG", 300)
    
    logger.info("\n" + "="*60)
    logger.info("ANGER EXPERIMENT COMPLETE")
    logger.info("Please switch to happiness server:")
    logger.info("1. Kill current server")
    logger.info("2. Run: python -m openai_server --model /data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct --emotion happiness --port 8000")
    logger.info("="*60)
    
    input("\nPress Enter when happiness server is running...")
    
    # Run happiness experiment
    happiness_results, happiness_coop = run_batch_experiment("happiness", "LOCAL_SERVER_NEUTRAL_CONFIG", 300)
    
    # Final comparison
    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS - 300 SCENARIOS")
    logger.info("="*60)
    logger.info(f"Happiness cooperation: {happiness_coop:.1f}% ({len([r for r in happiness_results if r['category'] == 'cooperate'])}/300)")
    logger.info(f"Anger cooperation: {anger_coop:.1f}% ({len([r for r in anger_results if r['category'] == 'cooperate'])}/300)")
    logger.info(f"Difference: {happiness_coop - anger_coop:.1f} percentage points")
    logger.info(f"Relative change: {((anger_coop - happiness_coop) / happiness_coop * 100):.1f}%")
    
    # Save comparison summary
    summary = {
        "experiment_date": datetime.now().isoformat(),
        "num_scenarios": 300,
        "model": "Qwen2.5-0.5B-Instruct",
        "method": "RepControlVLLMHook neural activation",
        "results": {
            "happiness": {
                "cooperation_rate": happiness_coop,
                "cooperate_count": len([r for r in happiness_results if r['category'] == 'cooperate']),
                "defect_count": len([r for r in happiness_results if r['category'] == 'defect'])
            },
            "anger": {
                "cooperation_rate": anger_coop,
                "cooperate_count": len([r for r in anger_results if r['category'] == 'cooperate']),
                "defect_count": len([r for r in anger_results if r['category'] == 'defect'])
            }
        },
        "effect_size": {
            "absolute_change": anger_coop - happiness_coop,
            "relative_change": ((anger_coop - happiness_coop) / happiness_coop * 100)
        }
    }
    
    with open("results/neural_test/300_scenario_final_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nExperiment complete! Summary saved.")

if __name__ == "__main__":
    main()