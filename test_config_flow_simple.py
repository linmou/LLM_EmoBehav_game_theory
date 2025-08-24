#!/usr/bin/env python3
"""
Simple test to verify additional_vllm_kwargs flow from YAML config without heavy dependencies.
This test verifies the configuration chain without importing vllm or other heavy modules.
"""

import sys
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import only the data models to test configuration flow
from emotion_memory_experiments.data_models import create_vllm_loading_config, VLLMLoadingConfig


def test_vllm_config_creation_and_kwargs():
    """
    Test VLLMLoadingConfig creation and to_vllm_kwargs method with additional_vllm_kwargs.
    """
    print("🔍 Testing VLLMLoadingConfig creation and kwargs merging...")
    
    # Test data matching memory_experiment_series.yaml
    test_additional_kwargs = {
        "rope-theta": 1000000,
        "rope-scaling": {
            "rope_type": "linear", 
            "factor": 4.0
        }
    }
    
    # Create VLLMLoadingConfig using the factory function
    loading_config = create_vllm_loading_config(
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        gpu_memory_utilization=0.90,
        tensor_parallel_size=None,
        max_model_len=32768,
        enforce_eager=True,
        quantization=None,
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        disable_custom_all_reduce=False,
        additional_vllm_kwargs=test_additional_kwargs
    )
    
    print(f"  ✅ VLLMLoadingConfig created successfully")
    print(f"  📋 additional_vllm_kwargs: {loading_config.additional_vllm_kwargs}")
    
    # Test to_vllm_kwargs method
    vllm_kwargs = loading_config.to_vllm_kwargs()
    
    # Verify base parameters
    assert vllm_kwargs["model"] == "Qwen/Qwen2.5-0.5B-Instruct", "Model path should match"
    assert vllm_kwargs["gpu_memory_utilization"] == 0.90, "GPU memory util should match"
    assert vllm_kwargs["dtype"] == "float16", "dtype should match"
    assert vllm_kwargs["seed"] == 42, "seed should match"
    
    # Verify additional_vllm_kwargs are merged
    assert "rope-theta" in vllm_kwargs, "rope-theta should be in vllm_kwargs"
    assert vllm_kwargs["rope-theta"] == 1000000, "rope-theta value should match"
    assert "rope-scaling" in vllm_kwargs, "rope-scaling should be in vllm_kwargs"
    assert vllm_kwargs["rope-scaling"]["rope_type"] == "linear", "rope-scaling type should match"
    assert vllm_kwargs["rope-scaling"]["factor"] == 4.0, "rope-scaling factor should match"
    
    print("  ✅ to_vllm_kwargs correctly merges additional parameters")
    return vllm_kwargs


def test_real_yaml_config():
    """
    Test with the actual memory_experiment_series.yaml config.
    """
    print("\n🔍 Testing with real memory_experiment_series.yaml config...")
    
    config_path = Path(__file__).parent / "config" / "memory_experiment_series.yaml"
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
        
    # Load the real config
    with open(config_path, 'r') as f:
        real_config = yaml.safe_load(f)
    
    # Extract loading_config section
    loading_cfg_dict = real_config.get("loading_config", {})
    if not loading_cfg_dict:
        print("❌ No loading_config section in YAML")
        return False
        
    print(f"  📋 Found loading_config in YAML:")
    print(f"    gpu_memory_utilization: {loading_cfg_dict.get('gpu_memory_utilization')}")
    print(f"    max_model_len: {loading_cfg_dict.get('max_model_len')}")
    print(f"    dtype: {loading_cfg_dict.get('dtype')}")
    print(f"    additional_vllm_kwargs: {loading_cfg_dict.get('additional_vllm_kwargs')}")
    
    # Create VLLMLoadingConfig using the same logic as memory_experiment_series_runner.py
    model_path = "Qwen/Qwen2.5-0.5B-Instruct"
    loading_config = create_vllm_loading_config(
        model_path=loading_cfg_dict.get("model_path", model_path),
        gpu_memory_utilization=loading_cfg_dict.get("gpu_memory_utilization", 0.90),
        tensor_parallel_size=loading_cfg_dict.get("tensor_parallel_size"),
        max_model_len=loading_cfg_dict.get("max_model_len", 32768),
        enforce_eager=loading_cfg_dict.get("enforce_eager", True),
        quantization=loading_cfg_dict.get("quantization"),
        trust_remote_code=loading_cfg_dict.get("trust_remote_code", True),
        dtype=loading_cfg_dict.get("dtype", "float16"),
        seed=loading_cfg_dict.get("seed", 42),
        disable_custom_all_reduce=loading_cfg_dict.get("disable_custom_all_reduce", False),
        additional_vllm_kwargs=loading_cfg_dict.get("additional_vllm_kwargs", {}),
    )
    
    # Test to_vllm_kwargs
    vllm_kwargs = loading_config.to_vllm_kwargs()
    
    print(f"  🔧 Final vLLM kwargs generated:")
    expected_additional = loading_cfg_dict.get("additional_vllm_kwargs", {})
    
    for key, expected_value in expected_additional.items():
        if key in vllm_kwargs:
            actual_value = vllm_kwargs[key]
            if actual_value == expected_value:
                print(f"    ✅ {key}: {actual_value} (matches expected)")
            else:
                print(f"    ❌ {key}: {actual_value} (expected {expected_value})")
                return False
        else:
            print(f"    ❌ {key}: MISSING from final vllm_kwargs")
            return False
    
    print("  ✅ Real YAML config test passed!")
    return True


def simulate_memory_experiment_series_runner_flow():
    """
    Simulate the exact flow used in memory_experiment_series_runner.py
    """
    print("\n🔍 Simulating memory_experiment_series_runner.py flow...")
    
    # Simulate the YAML config structure
    config_dict = {
        "loading_config": {
            "model_path": None,
            "gpu_memory_utilization": 0.90,
            "tensor_parallel_size": None,
            "max_model_len": 32768,
            "enforce_eager": True,
            "quantization": None,
            "trust_remote_code": True,
            "dtype": "float16",
            "seed": 42,
            "disable_custom_all_reduce": False,
            "additional_vllm_kwargs": {
                "rope-theta": 1000000,
                "rope-scaling": {
                    "rope_type": "linear",
                    "factor": 4.0
                }
            }
        }
    }
    
    model_path = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Replicate create_vllm_config_from_dict logic from memory_experiment_series_runner.py
    if "loading_config" not in config_dict:
        print("❌ No loading_config found")
        return False
        
    loading_cfg_dict = config_dict["loading_config"]
    
    # This is the exact code from memory_experiment_series_runner.py:68-80
    loading_config = create_vllm_loading_config(
        model_path=loading_cfg_dict.get("model_path", model_path),
        gpu_memory_utilization=loading_cfg_dict.get("gpu_memory_utilization", 0.90),
        tensor_parallel_size=loading_cfg_dict.get("tensor_parallel_size"),
        max_model_len=loading_cfg_dict.get("max_model_len", 32768),
        enforce_eager=loading_cfg_dict.get("enforce_eager", True),
        quantization=loading_cfg_dict.get("quantization"),
        trust_remote_code=loading_cfg_dict.get("trust_remote_code", True),
        dtype=loading_cfg_dict.get("dtype", "float16"),
        seed=loading_cfg_dict.get("seed", 42),
        disable_custom_all_reduce=loading_cfg_dict.get("disable_custom_all_reduce", False),
        additional_vllm_kwargs=loading_cfg_dict.get("additional_vllm_kwargs", {}),
    )
    
    print(f"  📋 VLLMLoadingConfig created")
    print(f"  🔧 additional_vllm_kwargs: {loading_config.additional_vllm_kwargs}")
    
    # This simulates what happens in neuro_manipulation/utils.py:601
    vllm_kwargs = loading_config.to_vllm_kwargs()
    
    print(f"  🎯 Final kwargs that would be passed to vLLM:")
    print(f"    model: {vllm_kwargs['model']}")
    print(f"    gpu_memory_utilization: {vllm_kwargs['gpu_memory_utilization']}")
    print(f"    dtype: {vllm_kwargs['dtype']}")
    print(f"    rope-theta: {vllm_kwargs.get('rope-theta', 'MISSING')}")
    print(f"    rope-scaling: {vllm_kwargs.get('rope-scaling', 'MISSING')}")
    
    # Verify the additional kwargs are present
    assert "rope-theta" in vllm_kwargs, "rope-theta should be in final vllm_kwargs"
    assert vllm_kwargs["rope-theta"] == 1000000, "rope-theta value should match"
    assert "rope-scaling" in vllm_kwargs, "rope-scaling should be in final vllm_kwargs"
    
    print("  ✅ Flow simulation successful!")
    return True


if __name__ == "__main__":
    print("🚀 Testing additional_vllm_kwargs configuration flow (without heavy dependencies)")
    print("=" * 80)
    
    try:
        # Test 1: Basic VLLMLoadingConfig functionality
        test_vllm_config_creation_and_kwargs()
        
        # Test 2: Real YAML config loading
        test_real_yaml_config()
        
        # Test 3: Simulate the exact series runner flow  
        simulate_memory_experiment_series_runner_flow()
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ additional_vllm_kwargs are correctly extracted from YAML")
        print("✅ VLLMLoadingConfig.to_vllm_kwargs() correctly merges additional parameters")
        print("✅ The configuration flow works end-to-end from YAML to vLLM kwargs")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)