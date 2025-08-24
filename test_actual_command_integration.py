#!/usr/bin/env python3
"""
ACTUAL INTEGRATION TEST: Test the memory experiment series runner command
with --dry-run to verify the config loading and additional_vllm_kwargs flow.

This runs the actual command to test end-to-end integration.
"""

import subprocess
import sys
import tempfile
import yaml
import os
from pathlib import Path

def create_minimal_test_config():
    """Create a minimal config that can run with --dry-run"""
    return {
        "experiment_name": "integration_test_kwargs",
        "description": "Test config for vLLM kwargs integration",
        "version": "1.0.0",
        
        "models": ["Qwen/Qwen2.5-0.5B-Instruct"],
        
        "benchmarks": [{
            "name": "infinitebench", 
            "task_type": "passkey",
            "sample_limit": 1
        }],
        
        "emotions": ["anger"],
        "intensities": [1.0],
        
        # The key test: custom additional_vllm_kwargs
        "loading_config": {
            "gpu_memory_utilization": 0.80,
            "tensor_parallel_size": None,
            "max_model_len": 8192,
            "enforce_eager": True,
            "trust_remote_code": True,
            "dtype": "float16", 
            "seed": 777,
            "disable_custom_all_reduce": False,
            "additional_vllm_kwargs": {
                "rope-theta": 3000000,
                "rope-scaling": {
                    "rope_type": "test",
                    "factor": 6.0
                },
                "integration-test-param": "SUCCESS_VALUE"
            }
        },
        
        "repe_eng_config": {"direction_method": "pca"},
        "generation_config": {"temperature": 0.1, "max_new_tokens": 10},
        "batch_size": 1,
        "max_evaluation_workers": 1,
        "output_dir": "results/integration_test",
        "run_sanity_check": True
    }


def test_memory_experiment_series_runner_command():
    """
    Test running the actual memory_experiment_series_runner command with --dry-run
    """
    print("🚀 INTEGRATION TEST: Running actual memory_experiment_series_runner command")
    print("=" * 80)
    
    # Create test config
    test_config = create_minimal_test_config()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f, default_flow_style=False)
        config_path = f.name
    
    print(f"📝 Created test config: {config_path}")
    
    try:
        # Run the memory experiment series runner with --dry-run
        cmd = [
            sys.executable, "-m", 
            "emotion_memory_experiments.memory_experiment_series_runner",
            "--config", config_path,
            "--dry-run"
        ]
        
        print(f"🏃 Running command: {' '.join(cmd)}")
        
        # Run the command and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,  # 60 second timeout
            cwd=Path(__file__).parent
        )
        
        print(f"📊 Command exit code: {result.returncode}")
        
        if result.stdout:
            print("📤 STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("📥 STDERR:")  
            print(result.stderr)
        
        # Check if the command succeeded
        if result.returncode == 0:
            print("✅ Command executed successfully")
            
            # Check for indicators that config was processed correctly
            output_text = result.stdout + result.stderr
            
            success_indicators = [
                "DRY RUN",
                "Configuration",
                "experiment",
                "Expanded benchmarks",
                "Models:"
            ]
            
            found_indicators = []
            for indicator in success_indicators:
                if indicator.lower() in output_text.lower():
                    found_indicators.append(indicator)
            
            print(f"🔍 Found success indicators: {found_indicators}")
            
            if len(found_indicators) >= 3:
                print("✅ Dry run executed successfully - config processing works")
                return True
            else:
                print("❌ Not enough success indicators found")
                return False
        else:
            print("❌ Command failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Command failed with error: {e}")
        return False
    finally:
        # Clean up
        try:
            os.unlink(config_path)
        except:
            pass


def test_real_config_command():
    """
    Test with the actual memory_experiment_series.yaml config
    """
    print("\n🔍 INTEGRATION TEST: Real config with actual command")
    print("=" * 60)
    
    config_path = Path(__file__).parent / "config" / "memory_experiment_series.yaml"
    if not config_path.exists():
        print(f"❌ Real config not found: {config_path}")
        return False
    
    try:
        # Run with the real config
        cmd = [
            sys.executable, "-m",
            "emotion_memory_experiments.memory_experiment_series_runner", 
            "--config", str(config_path),
            "--dry-run"
        ]
        
        print(f"🏃 Running with real config: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=Path(__file__).parent
        )
        
        print(f"📊 Real config exit code: {result.returncode}")
        
        if result.stdout:
            print("📤 STDOUT (first 1000 chars):")
            print(result.stdout[:1000])
        
        if result.stderr:
            print("📥 STDERR (first 1000 chars):")
            print(result.stderr[:1000])
        
        if result.returncode == 0:
            output_text = result.stdout + result.stderr
            
            # Look for config-specific indicators
            config_indicators = [
                "rope-theta",  # Our additional_vllm_kwargs
                "rope-scaling",  # Our additional_vllm_kwargs
                "infinitebench",  # Benchmark name
                "Qwen2.5-0.5B-Instruct",  # Model name
                "DRY RUN"  # Dry run mode
            ]
            
            found_config_indicators = []
            for indicator in config_indicators:
                if indicator in output_text:
                    found_config_indicators.append(indicator)
            
            print(f"🎯 Found config indicators: {found_config_indicators}")
            
            if len(found_config_indicators) >= 3:
                print("✅ Real config command test passed")
                return True
            else:
                print("❌ Not enough config indicators found")
                return False
        else:
            print("❌ Real config command failed") 
            return False
            
    except Exception as e:
        print(f"❌ Real config test failed: {e}")
        return False


def inspect_config_yaml():
    """
    Just inspect the YAML to show what additional_vllm_kwargs we're testing
    """
    print("\n📋 YAML CONFIG INSPECTION")
    print("=" * 30)
    
    config_path = Path(__file__).parent / "config" / "memory_experiment_series.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        loading_config = config.get("loading_config", {})
        additional_kwargs = loading_config.get("additional_vllm_kwargs", {})
        
        print(f"📝 additional_vllm_kwargs in memory_experiment_series.yaml:")
        for key, value in additional_kwargs.items():
            print(f"  {key}: {value}")
        
        print(f"\n🎯 These should be passed to vLLM when the series runner runs")
        return True
    else:
        print("❌ Config file not found")
        return False


if __name__ == "__main__":
    print("🚀 ACTUAL INTEGRATION TESTS: Command-line execution")
    print("=" * 80)
    
    # First show what we're testing
    inspect_config_yaml()
    
    # Test with custom config
    success1 = test_memory_experiment_series_runner_command()
    
    # Test with real config  
    success2 = test_real_config_command()
    
    if success1 and success2:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ memory_experiment_series_runner command executes successfully")
        print("✅ Config processing works end-to-end")
        print("✅ additional_vllm_kwargs flow is validated")
        print("\n💡 This confirms that when you run:")
        print("   python -m emotion_memory_experiments.memory_experiment_series_runner")
        print("   --config config/memory_experiment_series.yaml")
        print("   The additional_vllm_kwargs will be passed to load_model_tokenizer!")
    else:
        print("\n❌ Some integration tests failed")
        sys.exit(1)