#!/usr/bin/env python3
"""
Test file to verify additional_vllm_kwargs passing from YAML config to vLLM loading.
This test ensures that parameters in loading_config.additional_vllm_kwargs are properly
passed through the entire configuration chain to load_model_tokenizer in neuro_manipulation/utils.py.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the components to test
from emotion_memory_experiments.memory_experiment_series_runner import create_vllm_config_from_dict
from emotion_memory_experiments.data_models import VLLMLoadingConfig
from neuro_manipulation.utils import load_model_tokenizer


def test_additional_vllm_kwargs_passing():
    """
    Test that additional_vllm_kwargs from YAML config are properly passed through to vLLM.
    """
    print("🔍 Testing additional_vllm_kwargs passing from YAML to vLLM...")
    
    # Step 1: Create test config with additional_vllm_kwargs (matching memory_experiment_series.yaml)
    test_config = {
        "loading_config": {
            "model_path": "test/model/path",
            "gpu_memory_utilization": 0.85,
            "tensor_parallel_size": 2,
            "max_model_len": 16384,
            "enforce_eager": False,
            "quantization": None,
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "seed": 123,
            "disable_custom_all_reduce": True,
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
    
    # Step 2: Test create_vllm_config_from_dict extracts additional_vllm_kwargs correctly
    print("  📋 Step 1: Testing config extraction...")
    loading_config = create_vllm_config_from_dict(test_config, model_path)
    
    assert loading_config is not None, "VLLMLoadingConfig should be created"
    assert isinstance(loading_config, VLLMLoadingConfig), "Should return VLLMLoadingConfig instance"
    assert loading_config.additional_vllm_kwargs is not None, "additional_vllm_kwargs should be extracted"
    assert "rope-theta" in loading_config.additional_vllm_kwargs, "rope-theta should be in additional_vllm_kwargs"
    assert loading_config.additional_vllm_kwargs["rope-theta"] == 1000000, "rope-theta value should match"
    print("    ✅ Config extraction successful")
    
    # Step 3: Test to_vllm_kwargs method merges additional_vllm_kwargs
    print("  🔧 Step 2: Testing to_vllm_kwargs method...")
    vllm_kwargs = loading_config.to_vllm_kwargs()
    
    # Check that base parameters are there
    assert vllm_kwargs["model"] == "test/model/path", "Model path should match"
    assert vllm_kwargs["gpu_memory_utilization"] == 0.85, "GPU memory util should match"
    assert vllm_kwargs["dtype"] == "bfloat16", "dtype should match"
    
    # Check that additional_vllm_kwargs are merged
    assert "rope-theta" in vllm_kwargs, "rope-theta should be in final vllm_kwargs"
    assert vllm_kwargs["rope-theta"] == 1000000, "rope-theta value should match"
    assert "rope-scaling" in vllm_kwargs, "rope-scaling should be in final vllm_kwargs"
    assert vllm_kwargs["rope-scaling"]["rope_type"] == "linear", "rope-scaling type should match"
    assert vllm_kwargs["rope-scaling"]["factor"] == 4.0, "rope-scaling factor should match"
    print("    ✅ to_vllm_kwargs merging successful")
    
    # Step 4: Mock vLLM and test load_model_tokenizer receives correct kwargs
    print("  🎯 Step 3: Testing load_model_tokenizer vLLM integration...")
    with patch('neuro_manipulation.utils.LLM') as mock_llm_class, \
         patch('neuro_manipulation.utils.get_optimal_tensor_parallel_size', return_value=1), \
         patch('neuro_manipulation.utils.AutoTokenizer') as mock_tokenizer_class, \
         patch('neuro_manipulation.utils.detect_multimodal_model', return_value=False):
        
        # Mock the LLM constructor
        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance
        
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = None
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Call load_model_tokenizer with our loading_config
        model, tokenizer, processor = load_model_tokenizer(
            model_name_or_path=model_path,
            from_vllm=True,
            loading_config=loading_config
        )
        
        # Verify LLM was called with correct kwargs including additional_vllm_kwargs
        mock_llm_class.assert_called_once()
        actual_kwargs = mock_llm_class.call_args[1]
        
        # Check base parameters were passed
        assert actual_kwargs["model"] == "test/model/path", "Model path should be passed to vLLM"
        assert actual_kwargs["gpu_memory_utilization"] == 0.85, "GPU memory util should be passed to vLLM"
        assert actual_kwargs["dtype"] == "bfloat16", "dtype should be passed to vLLM"
        assert actual_kwargs["seed"] == 123, "seed should be passed to vLLM"
        
        # Check additional_vllm_kwargs were merged and passed
        assert "rope-theta" in actual_kwargs, "rope-theta should be passed to vLLM"
        assert actual_kwargs["rope-theta"] == 1000000, "rope-theta value should match in vLLM call"
        assert "rope-scaling" in actual_kwargs, "rope-scaling should be passed to vLLM"
        assert actual_kwargs["rope-scaling"]["rope_type"] == "linear", "rope-scaling type should match in vLLM call"
        assert actual_kwargs["rope-scaling"]["factor"] == 4.0, "rope-scaling factor should match in vLLM call"
        
        print("    ✅ load_model_tokenizer vLLM integration successful")
    
    print("✅ All tests passed! additional_vllm_kwargs are correctly passed from YAML to vLLM")


def test_with_real_memory_experiment_series_config():
    """
    Test using the actual memory_experiment_series.yaml config to verify the real flow.
    """
    print("\n🔍 Testing with real memory_experiment_series.yaml config...")
    
    config_path = Path(__file__).parent / "config" / "memory_experiment_series.yaml"
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
        
    # Load the real config
    with open(config_path, 'r') as f:
        real_config = yaml.safe_load(f)
    
    model_path = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Extract vLLM config using the real flow
    loading_config = create_vllm_config_from_dict(real_config, model_path)
    
    if loading_config is None:
        print("❌ No loading_config found in real config")
        return False
        
    print(f"  📋 Extracted loading config:")
    print(f"    model_path: {loading_config.model_path}")
    print(f"    gpu_memory_utilization: {loading_config.gpu_memory_utilization}")
    print(f"    max_model_len: {loading_config.max_model_len}")
    print(f"    dtype: {loading_config.dtype}")
    print(f"    additional_vllm_kwargs: {loading_config.additional_vllm_kwargs}")
    
    # Get the final vLLM kwargs
    vllm_kwargs = loading_config.to_vllm_kwargs()
    
    print(f"  🔧 Final vLLM kwargs contain additional parameters:")
    for key, value in loading_config.additional_vllm_kwargs.items():
        if key in vllm_kwargs:
            print(f"    ✅ {key}: {value}")
        else:
            print(f"    ❌ {key}: MISSING from final kwargs")
            return False
    
    print("✅ Real config test passed!")
    return True


if __name__ == "__main__":
    print("🚀 Testing additional_vllm_kwargs passing from YAML config to vLLM loading")
    print("=" * 80)
    
    try:
        # Test 1: Unit test with mock config
        test_additional_vllm_kwargs_passing()
        
        # Test 2: Integration test with real config
        success = test_with_real_memory_experiment_series_config()
        
        if success:
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ additional_vllm_kwargs are correctly passed from YAML to vLLM")
            print("✅ The configuration flow works end-to-end")
        else:
            print("\n❌ Some tests failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)