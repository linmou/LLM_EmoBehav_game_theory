"""
Example usage of the emotion memory experiment framework.

This script demonstrates how to run emotion experiments on memory benchmarks
using the framework. It includes examples for different benchmark types and
configurations.
"""
import json
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from emotion_memory_experiments.experiment import EmotionMemoryExperiment
from emotion_memory_experiments.data_models import ExperimentConfig, BenchmarkConfig
from emotion_memory_experiments.config_loader import load_emotion_memory_config, EmotionMemoryConfigLoader
from emotion_memory_experiments.tests.test_utils import (
    create_mock_passkey_data, create_mock_kv_retrieval_data,
    create_temp_data_file
)


def create_sample_passkey_data(output_path: str, num_items: int = 20):
    """Create sample passkey data for testing"""
    print(f"Creating {num_items} sample passkey items...")
    
    data = create_mock_passkey_data(num_items)
    
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Sample data saved to {output_path}")
    return output_path


def create_sample_kv_data(output_path: str, num_items: int = 15):
    """Create sample key-value retrieval data for testing"""
    print(f"Creating {num_items} sample key-value retrieval items...")
    
    data = create_mock_kv_retrieval_data(num_items)
    
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Sample data saved to {output_path}")
    return output_path


def run_passkey_experiment():
    """Example: Run emotion experiment on passkey retrieval task"""
    print("\n" + "="*60)
    print("PASSKEY RETRIEVAL EXPERIMENT")
    print("="*60)
    
    # Create sample data
    data_path = "sample_passkey_data.jsonl"
    create_sample_passkey_data(data_path, 10)
    
    # Configure benchmark
    benchmark_config = BenchmarkConfig(
        name="infinitebench",
        data_path=Path(data_path),
        task_type="passkey",
        evaluation_method="get_score_one_passkey",
        sample_limit=5  # Limit for quick demo
    )
    
    # Configure experiment - use get_repe_eng_config for proper setup
    from neuro_manipulation.configs.experiment_config import get_repe_eng_config
    
    # Get proper RepE config with all required parameters
    model_path = "/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct"
    repe_config = get_repe_eng_config(model_path)
    
    exp_config = ExperimentConfig(
        model_path=model_path,
        emotions=["anger", "happiness"],
        intensities=[0.5, 1.0],
        benchmark=benchmark_config,
        output_dir="results/emotion_memory",
        batch_size=2,
        generation_config={
            "temperature": 0.1,
            "max_new_tokens": 50,
            "do_sample": False,
            "top_p": 0.9
        }
    )
    
    try:
        # Run experiment
        experiment = EmotionMemoryExperiment(exp_config)
        print("Running passkey experiment...")
        results_df = experiment.run_experiment()
        
        print(f"\nExperiment completed! Results saved to {experiment.output_dir}")
        print(f"Total results: {len(results_df)}")
        
        # Show summary
        summary = results_df.groupby(['emotion', 'intensity'])['score'].agg(['mean', 'std', 'count'])
        print("\nSummary Results:")
        print(summary)
        
        return results_df
        
    except Exception as e:
        print(f"Error running experiment: {e}")
        print("This is expected if you don't have the model or dependencies installed.")
        return None
    
    finally:
        # Clean up
        try:
            Path(data_path).unlink()
        except FileNotFoundError:
            pass


def run_kv_retrieval_experiment():
    """Example: Run emotion experiment on key-value retrieval task"""
    print("\n" + "="*60)
    print("KEY-VALUE RETRIEVAL EXPERIMENT")
    print("="*60)
    
    # Create sample data
    data_path = "sample_kv_data.jsonl"
    create_sample_kv_data(data_path, 8)
    
    # Configure benchmark
    benchmark_config = BenchmarkConfig(
        name="infinitebench",
        data_path=Path(data_path),
        task_type="kv_retrieval",
        evaluation_method="get_score_one_kv_retrieval",
        sample_limit=4  # Small demo
    )
    
    # Configure experiment
    exp_config = ExperimentConfig(
        model_path="/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct",
        emotions=["anger", "sadness"],
        intensities=[1.0],
        benchmark=benchmark_config,
        output_dir="results/emotion_memory",
        batch_size=2
    )
    
    try:
        # Run experiment
        experiment = EmotionMemoryExperiment(exp_config)
        print("Running key-value retrieval experiment...")
        results_df = experiment.run_experiment()
        
        print(f"\nExperiment completed! Results saved to {experiment.output_dir}")
        print(f"Total results: {len(results_df)}")
        
        # Show summary
        summary = results_df.groupby(['emotion', 'intensity'])['score'].agg(['mean', 'std', 'count'])
        print("\nSummary Results:")
        print(summary)
        
        return results_df
        
    except Exception as e:
        print(f"Error running experiment: {e}")
        print("This is expected if you don't have the model or dependencies installed.")
        return None
    
    finally:
        # Clean up
        try:
            Path(data_path).unlink()
        except FileNotFoundError:
            pass


def run_yaml_config_experiment():
    """Example: Run experiment using YAML configuration"""
    print("\n" + "="*60)
    print("YAML CONFIGURATION EXPERIMENT")
    print("="*60)
    
    # Create a sample YAML config
    config_path = "sample_emotion_memory_config.yaml"
    print(f"Creating sample YAML config: {config_path}")
    
    EmotionMemoryConfigLoader.create_sample_config(config_path)
    
    # Create matching sample data
    data_path = "yaml_experiment_data.jsonl"
    create_sample_passkey_data(data_path, 8)
    
    # Update config to use our sample data
    import yaml
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    config_data['experiment']['benchmark']['data_path'] = data_path
    config_data['experiment']['benchmark']['sample_limit'] = 4
    config_data['experiment']['emotions'] = ["anger", "happiness"]
    config_data['experiment']['intensities'] = [1.0]
    config_data['experiment']['batch_size'] = 2
    
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    try:
        # Load configuration from YAML
        print("Loading experiment configuration from YAML...")
        exp_config = load_emotion_memory_config(config_path)
        
        print(f"Loaded config:")
        print(f"  Model: {exp_config.model_path}")
        print(f"  Emotions: {exp_config.emotions}")
        print(f"  Intensities: {exp_config.intensities}")
        print(f"  Benchmark: {exp_config.benchmark.name} - {exp_config.benchmark.task_type}")
        print(f"  Output: {exp_config.output_dir}")
        
        # Run experiment
        experiment = EmotionMemoryExperiment(exp_config)
        print("Running YAML-configured experiment...")
        results_df = experiment.run_experiment()
        
        print(f"\nYAML experiment completed! Results saved to {experiment.output_dir}")
        print(f"Total results: {len(results_df)}")
        
        # Show summary
        summary = results_df.groupby(['emotion', 'intensity'])['score'].agg(['mean', 'std', 'count'])
        print("\nSummary Results:")
        print(summary)
        
        return results_df
        
    except Exception as e:
        print(f"Error running YAML experiment: {e}")
        print("This is expected if you don't have the model or dependencies installed.")
        return None
    
    finally:
        # Clean up
        try:
            Path(config_path).unlink()
            Path(data_path).unlink()
        except FileNotFoundError:
            pass


def run_sanity_check_demo():
    """Example: Run a quick sanity check"""
    print("\n" + "="*60)
    print("SANITY CHECK DEMO")
    print("="*60)
    
    # Create minimal data
    data_path = "sanity_check_data.jsonl"
    create_sample_passkey_data(data_path, 3)
    
    # Configure benchmark
    benchmark_config = BenchmarkConfig(
        name="infinitebench",
        data_path=Path(data_path),
        task_type="passkey",
        evaluation_method="get_score_one_passkey"
    )
    
    # Configure experiment
    exp_config = ExperimentConfig(
        model_path="/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct",
        emotions=["anger"],
        intensities=[1.0],
        benchmark=benchmark_config,
        output_dir="results/sanity_check",
        batch_size=1
    )
    
    try:
        # Run sanity check
        experiment = EmotionMemoryExperiment(exp_config)
        print("Running sanity check with 2 samples...")
        results_df = experiment.run_sanity_check(sample_limit=2)
        
        print(f"\nSanity check completed! Results saved to {experiment.output_dir}")
        print(f"Total results: {len(results_df)}")
        
        # Show all results for small demo
        print("\nDetailed Results:")
        for _, row in results_df.iterrows():
            print(f"  {row['emotion']} (intensity {row['intensity']}): score = {row['score']:.3f}")
        
        return results_df
        
    except Exception as e:
        print(f"Error running sanity check: {e}")
        print("This is expected if you don't have the model or dependencies installed.")
        return None
    
    finally:
        # Clean up
        try:
            Path(data_path).unlink()
        except FileNotFoundError:
            pass


def show_configuration_example():
    """Show example configuration options"""
    print("\n" + "="*60)
    print("CONFIGURATION EXAMPLES")
    print("="*60)
    
    print("0. YAML Configuration (Recommended):")
    print("""
# Create and use YAML config
from emotion_memory_experiments.config_loader import load_emotion_memory_config

# Load from YAML file
exp_config = load_emotion_memory_config("config/emotion_memory_passkey.yaml")
experiment = EmotionMemoryExperiment(exp_config)
results = experiment.run_experiment()

# Sample YAML structure:
experiment:
  model_path: "/path/to/model"
  emotions: ["anger", "happiness", "sadness"]
  intensities: [0.5, 1.0, 1.5]
  benchmark:
    name: "infinitebench"
    data_path: "/path/to/data.jsonl"
    task_type: "passkey"
    evaluation_method: "get_score_one_passkey"
    sample_limit: 100
  generation_config:
    temperature: 0.1
    max_new_tokens: 50
  batch_size: 4
  output:
    base_dir: "results/emotion_memory"
""")
    
    print("1. Basic Passkey Configuration (Programmatic):")
    print("""
benchmark_config = BenchmarkConfig(
    name="infinitebench",
    data_path=Path("passkey_data.jsonl"),
    task_type="passkey",
    evaluation_method="get_score_one_passkey",
    sample_limit=100
)

exp_config = ExperimentConfig(
    model_path="/path/to/qwen/model",
    emotions=["anger", "happiness", "sadness"],
    intensities=[0.5, 1.0, 1.5],
    benchmark=benchmark_config,
    output_dir="results/emotion_memory",
    batch_size=4
)
""")
    
    print("2. Advanced Generation Configuration:")
    print("""
exp_config = ExperimentConfig(
    model_path="/path/to/model",
    emotions=["anger", "fear"],
    intensities=[0.3, 0.7, 1.0],
    benchmark=benchmark_config,
    output_dir="results/custom_experiment",
    batch_size=8,
    generation_config={
        "temperature": 0.2,
        "max_new_tokens": 200,
        "do_sample": True,
        "top_p": 0.95
    }
)
""")
    
    print("3. Different Benchmark Types:")
    print("""
# Key-Value Retrieval
benchmark_config = BenchmarkConfig(
    name="infinitebench",
    data_path=Path("kv_data.jsonl"),
    task_type="kv_retrieval",
    evaluation_method="get_score_one_kv_retrieval"
)

# Reading Comprehension
benchmark_config = BenchmarkConfig(
    name="infinitebench",
    data_path=Path("longbook_qa.jsonl"),
    task_type="longbook_qa_eng",
    evaluation_method="get_score_one_longbook_qa_eng"
)

# LoCoMo Conversational QA
benchmark_config = BenchmarkConfig(
    name="locomo",
    data_path=Path("locomo_data.json"),
    task_type="conversational_qa",
    evaluation_method="custom"
)
""")


def main():
    """Main example runner"""
    print("Emotion Memory Experiment Framework - Example Usage")
    print("="*60)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Show configuration examples
    show_configuration_example()
    
    # Run examples (these will likely fail without proper model setup)
    print("\nRunning demonstration experiments...")
    print("Note: These examples use mock data and may fail without proper model installation.")
    
    # Run YAML config example first (recommended approach)
    run_yaml_config_experiment()
    
    # Run sanity check
    run_sanity_check_demo()
    
    # Run full experiments (programmatic approach)
    run_passkey_experiment()
    run_kv_retrieval_experiment()
    
    print("\n" + "="*60)
    print("EXAMPLES COMPLETED")
    print("="*60)
    print("To run with real data:")
    print("1. Download InfiniteBench data: https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench")
    print("2. Modify the data_path in BenchmarkConfig to point to real data")
    print("3. Ensure you have the model and dependencies installed")
    print("4. Run the experiment!")


if __name__ == "__main__":
    main()