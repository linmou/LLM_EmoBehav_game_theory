#!/usr/bin/env python3
"""
Test for vLLM loading_config integration in emotion_memory_experiments.

This test verifies that VLLMLoadingConfig.additional_vllm_kwargs is properly passed through
to the load_model_tokenizer function in neuro_manipulation/utils.py, specifically testing
the configuration pipeline from YAML config -> VLLMLoadingConfig -> vLLM kwargs.

RED phase: This test should expose if additional_vllm_kwargs is not passed through properly.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from emotion_memory_experiments.data_models import VLLMLoadingConfig, create_vllm_loading_config
from emotion_memory_experiments.dataset_factory import create_vllm_config_from_dict


class TestVLLMLoadingConfigIntegration:
    """Test VLLMLoadingConfig integration with vLLM loading"""

    def test_vllm_loading_config_to_kwargs_includes_additional_params(self):
        """Test that to_vllm_kwargs() includes additional_vllm_kwargs parameters"""
        # Test additional kwargs are properly merged
        additional_kwargs = {
            "rope-theta": 1000000,
            "rope-scaling": {
                "rope_type": "linear", 
                "factor": 4.0
            },
            "custom_param": "test_value"
        }
        
        config = create_vllm_loading_config(
            model_path="Qwen/Qwen2.5-0.5B-Instruct",
            additional_vllm_kwargs=additional_kwargs
        )
        
        vllm_kwargs = config.to_vllm_kwargs()
        
        # Check that additional kwargs are present
        assert "rope-theta" in vllm_kwargs
        assert vllm_kwargs["rope-theta"] == 1000000
        assert "rope-scaling" in vllm_kwargs
        assert vllm_kwargs["rope-scaling"]["rope_type"] == "linear"
        assert vllm_kwargs["custom_param"] == "test_value"
        
        # Check that base kwargs are also present
        assert vllm_kwargs["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
        assert vllm_kwargs["gpu_memory_utilization"] == 0.90

    def test_create_vllm_config_from_dict_preserves_additional_kwargs(self):
        """Test that factory function preserves additional_vllm_kwargs from config dict"""
        config_dict = {
            "loading_config": {
                "model_path": "Qwen/Qwen2.5-0.5B-Instruct",
                "gpu_memory_utilization": 0.90,
                "max_model_len": 32768,
                "additional_vllm_kwargs": {
                    "rope-theta": 1000000,
                    "rope-scaling": {
                        "rope_type": "linear",
                        "factor": 4.0
                    }
                }
            }
        }
        
        vllm_config = create_vllm_config_from_dict(config_dict, "Qwen/Qwen2.5-0.5B-Instruct")
        vllm_kwargs = vllm_config.to_vllm_kwargs()
        
        # Verify additional kwargs are preserved
        assert "rope-theta" in vllm_kwargs
        assert vllm_kwargs["rope-theta"] == 1000000
        assert "rope-scaling" in vllm_kwargs
        assert vllm_kwargs["rope-scaling"]["factor"] == 4.0

    @patch('neuro_manipulation.utils.LLM')
    @patch('neuro_manipulation.utils.get_optimal_tensor_parallel_size')
    def test_load_model_tokenizer_uses_vllm_loading_config_kwargs(self, mock_tensor_size, mock_llm):
        """Test that load_model_tokenizer properly uses VLLMLoadingConfig.to_vllm_kwargs()"""
        # Import here to avoid dependency issues during test discovery
        from neuro_manipulation.utils import load_model_tokenizer
        
        # Setup mocks
        mock_tensor_size.return_value = 1
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance
        
        # Create loading config with additional kwargs
        additional_kwargs = {
            "rope-theta": 1000000,
            "rope-scaling": {
                "rope_type": "linear",
                "factor": 4.0
            },
            "custom_vllm_param": "test_value"
        }
        
        loading_config = create_vllm_loading_config(
            model_path="Qwen/Qwen2.5-0.5B-Instruct",
            gpu_memory_utilization=0.85,
            max_model_len=16384,
            additional_vllm_kwargs=additional_kwargs
        )
        
        # Call load_model_tokenizer with the config
        with patch('neuro_manipulation.utils.AutoTokenizer') as mock_tokenizer_cls:
            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token_id = None
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
            
            load_model_tokenizer(
                model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
                from_vllm=True,
                loading_config=loading_config
            )
        
        # Verify LLM was called with the correct kwargs
        mock_llm.assert_called_once()
        call_kwargs = mock_llm.call_args[1]
        
        # Check that additional kwargs were passed through
        assert "rope-theta" in call_kwargs
        assert call_kwargs["rope-theta"] == 1000000
        assert "rope-scaling" in call_kwargs
        assert call_kwargs["rope-scaling"]["rope_type"] == "linear"
        assert call_kwargs["custom_vllm_param"] == "test_value"
        
        # Check that config values were used (not defaults)
        assert call_kwargs["gpu_memory_utilization"] == 0.85
        assert call_kwargs["max_model_len"] == 16384
        assert call_kwargs["model"] == "Qwen/Qwen2.5-0.5B-Instruct"

    def test_memory_experiment_series_config_additional_kwargs(self):
        """Test that memory_experiment_series.yaml config format works correctly"""
        # This tests the exact format from memory_experiment_series.yaml
        config_dict = {
            "loading_config": {
                "model_path": None,  # Will be set by runner
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
        vllm_config = create_vllm_config_from_dict(config_dict, model_path)
        
        # Verify the config was created correctly
        assert vllm_config.model_path == model_path
        assert vllm_config.additional_vllm_kwargs["rope-theta"] == 1000000
        
        # Test to_vllm_kwargs includes everything
        vllm_kwargs = vllm_config.to_vllm_kwargs()
        
        # Should include additional kwargs
        assert vllm_kwargs["rope-theta"] == 1000000
        assert vllm_kwargs["rope-scaling"]["rope_type"] == "linear"
        assert vllm_kwargs["rope-scaling"]["factor"] == 4.0
        
        # Should include base kwargs
        assert vllm_kwargs["model"] == model_path
        assert vllm_kwargs["gpu_memory_utilization"] == 0.90
        assert vllm_kwargs["max_model_len"] == 32768
        assert vllm_kwargs["enforce_eager"] is True
        assert vllm_kwargs["trust_remote_code"] is True
        assert vllm_kwargs["dtype"] == "float16"
        assert vllm_kwargs["seed"] == 42
        assert vllm_kwargs["disable_custom_all_reduce"] is False


def test_additional_vllm_kwargs_override_base_kwargs():
    """Test that additional_vllm_kwargs can override base parameters"""
    # Create config where additional kwargs override a base parameter
    loading_config = create_vllm_loading_config(
        model_path="test/model", 
        seed=42,
        additional_vllm_kwargs={"seed": 999, "custom_param": "test"}
    )
    
    vllm_kwargs = loading_config.to_vllm_kwargs()
    
    # additional_vllm_kwargs should override base kwargs (dict merge order)
    assert vllm_kwargs["seed"] == 999  # Override worked
    assert vllm_kwargs["custom_param"] == "test"  # Additional param present


if __name__ == "__main__":
    # Run individual tests for debugging
    test_instance = TestVLLMLoadingConfigIntegration()
    
    print("Testing VLLMLoadingConfig.to_vllm_kwargs()...")
    test_instance.test_vllm_loading_config_to_kwargs_includes_additional_params()
    print("✅ PASSED")
    
    print("Testing create_vllm_config_from_dict()...")
    test_instance.test_create_vllm_config_from_dict_preserves_additional_kwargs()
    print("✅ PASSED")
    
    print("Testing memory_experiment_series.yaml format...")
    test_instance.test_memory_experiment_series_config_additional_kwargs()
    print("✅ PASSED")
    
    print("Testing additional_vllm_kwargs override...")
    test_additional_vllm_kwargs_override_base_kwargs()
    print("✅ PASSED")
    
    print("All tests PASSED! ✅")
    print("\nTo test actual vLLM integration, run with vLLM available:")
    print("pytest test_vllm_loading_config_integration.py -v")