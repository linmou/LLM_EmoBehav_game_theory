#!/usr/bin/env python3
"""
Simple code example for Option Probability Experiment configuration.

This shows how to define configurations directly in Python code
without using YAML files.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from neuro_manipulation.experiments.option_probability_experiment import OptionProbabilityExperiment
from games.game_configs import get_game_config
from constants import GameNames

def create_configs():
    """Create the three configuration dictionaries needed for the experiment."""
    
    # 1. Representation Engineering Configuration
    repe_eng_config = {
        'model_name_or_path': '/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct',
        'tensor_parallel_size': 1,
        'gpu_memory_utilization': 0.85,
        'max_num_seqs': 16,
        'enforce_eager': True,
        'trust_remote_code': True,
        'coeffs': [0.0, 1.0],
        'block_name': 'model.layers.{}.self_attn',
        'control_method': 'reading_vec'
    }
    
    # 2. Experiment Configuration
    exp_config = {
        'experiment': {
            'name': 'python_config_test',
            'emotions': ['neutral', 'angry'],
            'emotion_intensities': [0.0, 1.0],
            'output': {
                'base_dir': 'experiments/python_config_results'
            }
        }
    }
    
    # 3. Game Configuration - use built-in config
    game_config = get_game_config(GameNames.PRISONERS_DILEMMA)
    
    return repe_eng_config, exp_config, game_config

def run_experiment():
    """Run a simple option probability experiment."""
    print("🚀 Setting up Option Probability Experiment from Python config...")
    
    # Create configurations
    repe_eng_config, exp_config, game_config = create_configs()
    
    print(f"Model: {repe_eng_config['model_name_or_path']}")
    print(f"Game: {game_config['game_name']}")
    print(f"Emotions: {exp_config['experiment']['emotions']}")
    
    # Initialize experiment
    experiment = OptionProbabilityExperiment(
        repe_eng_config=repe_eng_config,
        exp_config=exp_config,
        game_config=game_config,
        batch_size=2,          # Small batch for quick test
        sample_num=3           # Only 3 scenarios per condition
    )
    
    print("📊 Running experiment...")
    
    # Run the experiment
    results_file = experiment.run_experiment()
    
    print(f"✅ Experiment completed!")
    print(f"📁 Results saved to: {results_file}")
    print(f"📂 Output directory: {experiment.output_dir}")
    
    return results_file

def create_custom_game_config():
    """Example of creating a custom game configuration."""
    
    custom_game_config = {
        'game_name': 'custom_prisoners_dilemma',
        'data_path': 'data_creation/scenario_creation/langgraph_creation/Prisoners_Dilemma_all_data_samples_4.1_distinct_choices.json',
        'decision_class': 'binary_choice',
        'payoff_matrix': {
            'both_cooperate': [3, 3],
            'cooperate_defect': [0, 5], 
            'defect_cooperate': [5, 0],
            'both_defect': [1, 1]
        },
        'options': ['Cooperate', 'Defect'],
        'description': 'Classic Prisoner\'s Dilemma with custom settings'
    }
    
    return custom_game_config

def example_with_different_emotions():
    """Example with different emotion configurations."""
    
    repe_eng_config, exp_config, game_config = create_configs()
    
    # Modify to test different emotions
    exp_config['experiment']['emotions'] = ['neutral', 'angry', 'happy', 'sad']
    exp_config['experiment']['emotion_intensities'] = [0.0, 1.0, 1.0, 1.0]
    exp_config['experiment']['name'] = 'multi_emotion_test'
    
    print("🎭 Running multi-emotion experiment...")
    print(f"Testing emotions: {exp_config['experiment']['emotions']}")
    
    experiment = OptionProbabilityExperiment(
        repe_eng_config=repe_eng_config,
        exp_config=exp_config,
        game_config=game_config,
        batch_size=2,
        sample_num=2
    )
    
    results_file = experiment.run_experiment()
    print(f"✅ Multi-emotion experiment completed: {results_file}")
    
    return results_file

if __name__ == "__main__":
    print("=" * 60)
    print("OPTION PROBABILITY EXPERIMENT - PYTHON CONFIG EXAMPLES")
    print("=" * 60)
    
    # Example 1: Basic experiment
    print("\n1. 📊 Basic Experiment")
    try:
        run_experiment()
    except Exception as e:
        print(f"❌ Basic experiment failed: {e}")
    
    # Example 2: Multi-emotion experiment
    print("\n2. 🎭 Multi-Emotion Experiment")
    try:
        example_with_different_emotions()
    except Exception as e:
        print(f"❌ Multi-emotion experiment failed: {e}")
    
    # Example 3: Show custom game config
    print("\n3. 🎮 Custom Game Configuration Example")
    custom_config = create_custom_game_config()
    print("Custom game config structure:")
    for key, value in custom_config.items():
        print(f"  {key}: {value}")
    
    print("\n🎯 All examples completed!") 