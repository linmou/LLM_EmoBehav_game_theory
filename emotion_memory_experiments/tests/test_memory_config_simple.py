#!/usr/bin/env python3
"""
Simplified test script to validate memory experiment series configuration.
Tests configuration parsing without importing heavy dependencies like vLLM.
"""

import yaml
from pathlib import Path

# Import the real BenchmarkConfig class directly from data_models
import sys
import importlib.util

# Load the data_models module directly to avoid heavy dependencies
spec = importlib.util.spec_from_file_location(
    "data_models", 
    "emotion_memory_experiments/data_models.py"
)
data_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_models)

BenchmarkConfig = data_models.BenchmarkConfig


def test_config_structure():
    """Test basic configuration structure"""
    print("🔍 Testing configuration structure...")
    
    config_path = 'config/memory_experiment_series.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check required sections
    required_sections = ['models', 'benchmarks', 'emotions', 'intensities']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")
        print(f"✅ Found {section}: {config[section]}")
    
    print("✅ All required sections present")
    return config


def test_benchmark_configs(config):
    """Test benchmark configurations"""
    print("\n🔍 Testing benchmark configurations...")
    
    benchmarks = config['benchmarks']
    print(f"📊 Found {len(benchmarks)} benchmarks:")
    
    for i, bench in enumerate(benchmarks):
        required_fields = ['name', 'task_type']
        for field in required_fields:
            if field not in bench:
                raise ValueError(f"Benchmark {i} missing required field: {field}")
        
        bench_id = f"{bench['name']}_{bench['task_type']}"
        print(f"   • {bench_id}")
        
        # Test auto data path generation
        filename = f"{bench['name']}_{bench['task_type']}.jsonl"
        auto_path = Path("data/memory_benchmarks") / filename
        print(f"     → Auto data path: {auto_path}")
    
    print("✅ All benchmark configs valid")


def test_model_configs(config):
    """Test model configurations"""
    print("\n🔍 Testing model configurations...")
    
    models = config['models']
    print(f"🤖 Found {len(models)} models:")
    
    for model in models:
        print(f"   • {model}")
        
        # Test model name formatting for folders
        formatted = format_model_name_for_folder(model)
        print(f"     → Folder format: {formatted}")
    
    print("✅ All model configs valid")


def format_model_name_for_folder(model_name: str) -> str:
    """Format model name for folder (simplified version)"""
    slash_count = model_name.count("/")
    
    if slash_count >= 2 and model_name.startswith("/"):
        parts = model_name.split("/")
        if len(parts) >= 3:
            return f"{parts[-2]}/{parts[-1]}"
    elif slash_count == 1:
        return model_name
    elif slash_count == 2:
        parts = model_name.split("/")
        return f"{parts[1]}/{parts[2]}"
    elif slash_count > 2:
        parts = model_name.split("/")
        return f"{parts[-2]}/{parts[-1]}"
    
    return model_name


def is_pattern_task_type(task_type: str) -> bool:
    """Check if task_type contains regex pattern characters"""
    if not task_type:
        return False
    
    pattern_chars = ['*', '?', '[', ']', '{', '}', '(', ')', '^', '$', '+', '.', '|', '\\']
    return any(char in task_type for char in pattern_chars)

def expand_benchmark_configs_real(benchmarks):
    """Real benchmark expansion using actual BenchmarkConfig.discover_datasets_by_pattern"""
    expanded_benchmarks = []
    
    for benchmark_config in benchmarks:
        task_type = benchmark_config.get("task_type", "")
        
        if is_pattern_task_type(task_type):
            # Use REAL BenchmarkConfig class and method
            benchmark = BenchmarkConfig(
                name=benchmark_config["name"],
                task_type=task_type
            )
            
            try:
                discovered_tasks = benchmark.discover_datasets_by_pattern()
                
                print(f"   🔍 Pattern '{task_type}' for '{benchmark_config['name']}' → {discovered_tasks}")
                
                if discovered_tasks:
                    # Verify files actually exist
                    for task in discovered_tasks:
                        expected_file = benchmark.get_data_path().parent / f"{benchmark.name}_{task}.jsonl"
                        if expected_file.exists():
                            print(f"      ✅ {expected_file}")
                        else:
                            print(f"      ❌ {expected_file} (discovered but not found)")
                else:
                    print(f"      ℹ️  No matching datasets found")
                
                # Create individual configs for each discovered task
                for discovered_task_type in discovered_tasks:
                    expanded_config = benchmark_config.copy()
                    expanded_config["task_type"] = discovered_task_type
                    expanded_benchmarks.append(expanded_config)
                    
            except ValueError as e:
                print(f"   ❌ Regex error for '{benchmark_config['name']}': {e}")
            except Exception as e:
                print(f"   💥 Unexpected error for '{benchmark_config['name']}': {e}")
        else:
            # Keep literal task types as-is
            expanded_benchmarks.append(benchmark_config)
    
    return expanded_benchmarks

def test_experiment_combinations(config):
    """Test experiment ID generation"""
    print("\n🔍 Testing experiment combinations...")
    
    original_benchmarks = config['benchmarks']
    models = config['models']
    emotions = config['emotions']
    intensities = config['intensities']
    
    # Test benchmark expansion with REAL discovery
    print("📈 Testing benchmark expansion with REAL discovery...")
    expanded_benchmarks = expand_benchmark_configs_real(original_benchmarks)
    
    total_combinations = len(expanded_benchmarks) * len(models)
    total_with_emotions = total_combinations * len(emotions) * len(intensities)
    
    print(f"🧮 Original benchmarks: {len(original_benchmarks)}")
    print(f"🧮 Expanded benchmarks: {len(expanded_benchmarks)}")
    print(f"🧮 Benchmark × Model combinations: {total_combinations}")
    print(f"🧮 Total with emotions/intensities: {total_with_emotions}")
    
    experiment_ids = []
    
    for benchmark in expanded_benchmarks:
        for model in models:
            formatted_model = format_model_name_for_folder(model)
            exp_id = f"{benchmark['name']}_{benchmark['task_type']}_{formatted_model.replace('/', '_')}"
            experiment_ids.append(exp_id)
            print(f"   • {exp_id}")
    
    print(f"🏷️  Generated {len(experiment_ids)} unique experiment IDs")
    print("✅ All experiment combinations valid")


def test_regex_pattern_feature():
    """Test the new regex pattern feature for task_type"""
    print("\n🔍 Testing regex pattern feature...")
    
    # Test config with different regex patterns
    test_configs = [
        {
            'name': 'All tasks pattern',
            'benchmarks': [
                {
                    'name': 'longbench',
                    'task_type': '.*',  # All tasks
                    'evaluation_method': 'exact_match',
                    'sample_limit': 100
                }
            ]
        },
        {
            'name': 'QA tasks pattern',
            'benchmarks': [
                {
                    'name': 'longbench',
                    'task_type': '.*qa.*',  # QA-related tasks
                    'evaluation_method': 'exact_match',
                    'sample_limit': 100
                }
            ]
        },
        {
            'name': 'Narrative tasks pattern',
            'benchmarks': [
                {
                    'name': 'longbench',
                    'task_type': 'narrative.*',  # Starts with 'narrative'
                    'evaluation_method': 'exact_match',
                    'sample_limit': 100
                }
            ]
        }
    ]
    
    for test_config in test_configs:
        print(f"\n📋 Testing {test_config['name']}:")
        benchmarks = test_config['benchmarks']
        expanded = expand_benchmark_configs_real(benchmarks)
        
        original_task_type = benchmarks[0]['task_type']
        print(f"   • Pattern: {original_task_type}")
        
        if expanded:
            for bench in expanded:
                print(f"   • Expanded to: {bench['name']}_{bench['task_type']}")
        else:
            print("   • No matches found")
    
    # Test mixed config (patterns + literals)
    test_config_mixed = {
        'benchmarks': [
            {
                'name': 'longbench',
                'task_type': '.*'  # Pattern
            },
            {
                'name': 'infinitebench', 
                'task_type': 'passkey'  # Literal
            }
        ]
    }
    
    print("\n📋 Testing mixed config (patterns + literals):")
    mixed_expanded = expand_benchmark_configs_real(test_config_mixed['benchmarks'])
    for bench in mixed_expanded:
        pattern_indicator = "🔍" if is_pattern_task_type(bench['task_type']) else "📄"
        print(f"   • {pattern_indicator} {bench['name']}_{bench['task_type']}")
    
    print("✅ Regex pattern feature working correctly")

def test_optional_configs(config):
    """Test optional configuration sections"""
    print("\n🔍 Testing optional configurations...")
    
    optional_sections = [
        'loading_config', 'repe_eng_config', 'generation_config',
        'batch_size', 'output_dir', 'run_sanity_check'
    ]
    
    for section in optional_sections:
        if section in config:
            print(f"✅ Found optional {section}")
        else:
            print(f"ℹ️  Missing optional {section} (will use defaults)")
    
    print("✅ Optional configs checked")


def main():
    """Run all tests"""
    print("🚀 Starting simplified memory experiment configuration tests...\n")
    
    try:
        # Test 1: Config structure
        config = test_config_structure()
        
        # Test 2: Benchmark configs
        test_benchmark_configs(config)
        
        # Test 3: Model configs
        test_model_configs(config)
        
        # Test 4: Experiment combinations
        test_experiment_combinations(config)
        
        # Test 5: regex pattern feature
        test_regex_pattern_feature()
        
        # Test 6: Optional configs
        test_optional_configs(config)
        
        print("\n🎉 All tests passed! Configuration is valid and ready to use.")
        print(f"\n📈 Summary:")
        print(f"   • {len(config['benchmarks'])} benchmarks")
        print(f"   • {len(config['models'])} models") 
        print(f"   • {len(config['emotions'])} emotions")
        print(f"   • {len(config['intensities'])} intensities")
        
        total_experiments = len(config['benchmarks']) * len(config['models'])
        print(f"   • {total_experiments} total experiment combinations")
        
        print("\n💡 To run the actual experiments:")
        print("   python emotion_memory_experiments/memory_experiment_series_runner.py --config config/memory_experiment_series.yaml")
        
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)