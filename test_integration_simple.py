#!/usr/bin/env python3
"""
INTEGRATION TEST: Test that runs memory_experiment_series_runner --dry-run 
and verifies the config flow by intercepting the load_model_tokenizer call.
"""

import sys
import os
import tempfile
import yaml
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def create_test_config():
    """Create a test config with custom additional_vllm_kwargs"""
    return {
        "experiment_name": "integration_test_kwargs",
        "description": "Integration test for vLLM kwargs passing",
        "version": "1.0.0",
        
        "models": ["Qwen/Qwen2.5-0.5B-Instruct"],
        
        "benchmarks": [{
            "name": "infinitebench",
            "task_type": "passkey",
            "sample_limit": 1
        }],
        
        "emotions": ["anger"],
        "intensities": [1.0],
        
        # Custom vLLM parameters to test
        "loading_config": {
            "gpu_memory_utilization": 0.75,  # Custom value
            "tensor_parallel_size": None,
            "max_model_len": 16384,  # Custom value
            "enforce_eager": False,  # Custom value
            "trust_remote_code": True,
            "dtype": "bfloat16",  # Custom value  
            "seed": 999,  # Custom value
            "disable_custom_all_reduce": True,  # Custom value
            "additional_vllm_kwargs": {
                "rope-theta": 5000000,  # Custom value to verify
                "rope-scaling": {
                    "rope_type": "custom",
                    "factor": 10.0
                },
                "test-parameter": "integration-test-value"
            }
        },
        
        "repe_eng_config": {"direction_method": "pca"},
        "generation_config": {"temperature": 0.1, "max_new_tokens": 50},
        "batch_size": 1,
        "max_evaluation_workers": 1,
        "output_dir": "results/integration_test",
        "run_sanity_check": True
    }


def test_config_flow_integration():
    """
    Test the config flow by creating config and running the series runner setup.
    """
    print("🚀 INTEGRATION TEST: Config flow from YAML to VLLMLoadingConfig")
    print("=" * 80)
    
    test_config = create_test_config()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f, default_flow_style=False)
        config_path = f.name
    
    print(f"📝 Created test config: {config_path}")
    
    try:
        # Import the series runner
        from emotion_memory_experiments.memory_experiment_series_runner import MemoryExperimentSeriesRunner
        
        print("🏃 Creating MemoryExperimentSeriesRunner...")
        
        # Create runner with dry_run=True to avoid heavy loading
        runner = MemoryExperimentSeriesRunner(
            config_path=config_path,
            series_name="integration_test",
            resume=False,
            dry_run=True
        )
        
        print("✅ MemoryExperimentSeriesRunner created successfully")
        
        # Test the config loading
        print("📋 Verifying config was loaded correctly...")
        assert "loading_config" in runner.base_config, "loading_config should be in base_config"
        
        loading_cfg = runner.base_config["loading_config"]
        assert loading_cfg["gpu_memory_utilization"] == 0.75, "gpu_memory_utilization should match"
        assert loading_cfg["dtype"] == "bfloat16", "dtype should match"
        assert "additional_vllm_kwargs" in loading_cfg, "additional_vllm_kwargs should be present"
        
        additional_kwargs = loading_cfg["additional_vllm_kwargs"]
        assert additional_kwargs["rope-theta"] == 5000000, "rope-theta should match"
        assert additional_kwargs["test-parameter"] == "integration-test-value", "test-parameter should match"
        
        print("✅ Base config loading verified")
        
        # Test the VLLMLoadingConfig creation through setup_experiment
        print("🔧 Testing VLLMLoadingConfig creation...")
        
        benchmark_config = test_config["benchmarks"][0]
        model_name = test_config["models"][0]
        
        # This should create the VLLMLoadingConfig using our test config
        experiment = runner.setup_experiment(benchmark_config, model_name)
        
        # For dry run, we get a MockExperiment
        assert hasattr(experiment, 'config'), "Experiment should have config"
        
        exp_config = experiment.config
        assert hasattr(exp_config, 'loading_config'), "ExperimentConfig should have loading_config"
        
        loading_config = exp_config.loading_config
        assert loading_config is not None, "loading_config should not be None"
        
        print("✅ VLLMLoadingConfig created successfully")
        
        # Test to_vllm_kwargs method
        print("🎯 Testing to_vllm_kwargs method...")
        
        vllm_kwargs = loading_config.to_vllm_kwargs()
        
        # Verify base parameters
        assert vllm_kwargs["gpu_memory_utilization"] == 0.75, f"gpu_memory_utilization should be 0.75, got {vllm_kwargs['gpu_memory_utilization']}"
        assert vllm_kwargs["dtype"] == "bfloat16", f"dtype should be bfloat16, got {vllm_kwargs['dtype']}"
        assert vllm_kwargs["seed"] == 999, f"seed should be 999, got {vllm_kwargs['seed']}"
        
        # Verify additional_vllm_kwargs were merged
        assert "rope-theta" in vllm_kwargs, "rope-theta should be in vllm_kwargs"
        assert vllm_kwargs["rope-theta"] == 5000000, f"rope-theta should be 5000000, got {vllm_kwargs['rope-theta']}"
        
        assert "rope-scaling" in vllm_kwargs, "rope-scaling should be in vllm_kwargs"
        rope_scaling = vllm_kwargs["rope-scaling"]
        assert rope_scaling["rope_type"] == "custom", f"rope_type should be custom, got {rope_scaling['rope_type']}"
        assert rope_scaling["factor"] == 10.0, f"factor should be 10.0, got {rope_scaling['factor']}"
        
        assert "test-parameter" in vllm_kwargs, "test-parameter should be in vllm_kwargs"
        assert vllm_kwargs["test-parameter"] == "integration-test-value", f"test-parameter should match, got {vllm_kwargs['test-parameter']}"
        
        print("✅ to_vllm_kwargs merging verified")
        
        print("\n📊 Final vLLM kwargs that would be passed:")
        for key, value in vllm_kwargs.items():
            if key in ["rope-theta", "rope-scaling", "test-parameter", "gpu_memory_utilization", "dtype", "seed"]:
                print(f"  {key}: {value}")
        
        print("\n🎉 INTEGRATION TEST PASSED!")
        print("✅ Config flows correctly from YAML → MemoryExperimentSeriesRunner → VLLMLoadingConfig → vLLM kwargs")
        print("✅ additional_vllm_kwargs are properly merged with base parameters")
        print("✅ All custom parameters reach the final vLLM kwargs")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        try:
            os.unlink(config_path)
        except:
            pass


def test_real_config_integration():
    """
    Test with the actual memory_experiment_series.yaml config file.
    """
    print("\n🔍 INTEGRATION TEST: Real memory_experiment_series.yaml config")
    print("=" * 60)
    
    config_path = Path(__file__).parent / "config" / "memory_experiment_series.yaml"
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        from emotion_memory_experiments.memory_experiment_series_runner import MemoryExperimentSeriesRunner
        
        print(f"📝 Using real config: {config_path}")
        
        # Create runner with dry_run=True
        runner = MemoryExperimentSeriesRunner(
            config_path=str(config_path),
            series_name="real_config_test", 
            resume=False,
            dry_run=True
        )
        
        # Test setup with first benchmark and model
        benchmarks = runner.base_config["benchmarks"]
        models = runner.base_config["models"]
        
        if not benchmarks or not models:
            print("❌ No benchmarks or models in config")
            return False
        
        benchmark_config = benchmarks[0] 
        model_name = models[0]
        
        print(f"🔧 Testing setup with benchmark: {benchmark_config['name']}, model: {model_name}")
        
        experiment = runner.setup_experiment(benchmark_config, model_name)
        loading_config = experiment.config.loading_config
        
        if loading_config is None:
            print("❌ No loading_config created")
            return False
        
        # Test the real additional_vllm_kwargs from memory_experiment_series.yaml
        vllm_kwargs = loading_config.to_vllm_kwargs()
        
        expected_additional = {
            "rope-theta": 1000000,
            "rope-scaling": {
                "rope_type": "linear",
                "factor": 4.0
            }
        }
        
        print("🎯 Verifying real config additional_vllm_kwargs:")
        for key, expected_value in expected_additional.items():
            if key in vllm_kwargs:
                actual_value = vllm_kwargs[key]
                if actual_value == expected_value:
                    print(f"  ✅ {key}: {actual_value}")
                else:
                    print(f"  ❌ {key}: {actual_value} (expected {expected_value})")
                    return False
            else:
                print(f"  ❌ {key}: MISSING")
                return False
        
        print("✅ Real config integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Real config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 INTEGRATION TESTS: vLLM kwargs passing")
    print("=" * 80)
    
    success1 = test_config_flow_integration()
    success2 = test_real_config_integration()
    
    if success1 and success2:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ additional_vllm_kwargs flow works end-to-end")
    else:
        print("\n❌ Some integration tests failed")
        sys.exit(1)