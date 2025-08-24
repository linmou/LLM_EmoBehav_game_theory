#!/usr/bin/env python3
"""
INTEGRATION TEST: Test that additional_vllm_kwargs from memory_experiment_series.yaml
actually reach the vLLM constructor when running the memory experiment series runner.

This test actually runs the memory_experiment_series_runner with --dry-run and 
intercepts the vLLM loading to verify the kwargs are passed correctly.
"""

import sys
import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def create_test_config():
    """Create a minimal test config with additional_vllm_kwargs"""
    return {
        "experiment_name": "integration_test",
        "description": "Integration test for additional_vllm_kwargs passing",
        "version": "1.0.0",
        
        "models": ["Qwen/Qwen2.5-0.5B-Instruct"],
        
        "benchmarks": [{
            "name": "infinitebench",
            "task_type": "passkey",
            "sample_limit": 1,
            "augmentation_config": None,
            "enable_auto_truncation": True,
            "truncation_strategy": "right",
            "preserve_ratio": 0.95
        }],
        
        "emotions": ["anger"],
        "intensities": [1.0],
        
        # This is what we're testing - these should reach vLLM
        "loading_config": {
            "model_path": None,
            "gpu_memory_utilization": 0.85,  # Different from default to verify
            "tensor_parallel_size": None,
            "max_model_len": 16384,  # Different from default to verify
            "enforce_eager": False,  # Different from default to verify  
            "quantization": None,
            "trust_remote_code": True,
            "dtype": "bfloat16",  # Different from default to verify
            "seed": 123,  # Different from default to verify
            "disable_custom_all_reduce": True,  # Different from default to verify
            "additional_vllm_kwargs": {
                "rope-theta": 2000000,  # Custom value to verify
                "rope-scaling": {
                    "rope_type": "dynamic",  # Custom value to verify
                    "factor": 8.0  # Custom value to verify
                },
                "custom-param": "test-value"  # Custom parameter to verify
            }
        },
        
        "repe_eng_config": {
            "direction_method": "pca"
        },
        
        "generation_config": {
            "temperature": 0.1,
            "max_new_tokens": 50,
            "do_sample": False
        },
        
        "batch_size": 2,
        "max_evaluation_workers": 1,
        "pipeline_queue_size": 1,
        "output_dir": "results/integration_test",
        "run_sanity_check": True
    }


def test_integration_vllm_kwargs_passing():
    """
    Integration test: Run memory_experiment_series_runner and verify
    additional_vllm_kwargs are passed to vLLM constructor.
    """
    print("🚀 INTEGRATION TEST: Testing additional_vllm_kwargs passing to vLLM")
    print("=" * 80)
    
    # Create temporary config file
    test_config = create_test_config()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f, default_flow_style=False)
        config_path = f.name
    
    print(f"📝 Created test config: {config_path}")
    
    try:
        # Import the memory experiment series runner
        # We need to patch modules to avoid heavy dependencies
        captured_vllm_kwargs = {}
        
        def mock_llm_constructor(**kwargs):
            """Mock LLM constructor that captures kwargs"""
            captured_vllm_kwargs.update(kwargs)
            mock_llm = MagicMock()
            return mock_llm
        
        def mock_get_optimal_tensor_parallel_size(model_path):
            return 1
        
        def mock_autoconfig_from_pretrained(*args, **kwargs):
            mock_config = MagicMock()
            mock_config.architectures = ["QwenForCausalLM"]
            return mock_config
        
        def mock_automodel_from_pretrained(*args, **kwargs):
            mock_model = MagicMock()
            mock_model.config = mock_config_from_pretrained()
            return mock_model
        
        def mock_config_from_pretrained(*args, **kwargs):
            mock_config = MagicMock()
            mock_config.architectures = ["QwenForCausalLM"] 
            return mock_config
        
        def mock_autotokenizer_from_pretrained(*args, **kwargs):
            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token_id = None
            return mock_tokenizer
        
        def mock_detect_multimodal_model(model_path):
            return False
            
        # Mock all the heavy imports and dependencies
        patches = [
            patch('neuro_manipulation.utils.LLM', side_effect=mock_llm_constructor),
            patch('neuro_manipulation.utils.get_optimal_tensor_parallel_size', side_effect=mock_get_optimal_tensor_parallel_size),
            patch('neuro_manipulation.utils.AutoConfig.from_pretrained', side_effect=mock_autoconfig_from_pretrained),
            patch('neuro_manipulation.utils.AutoModelForCausalLM.from_pretrained', side_effect=mock_automodel_from_pretrained), 
            patch('neuro_manipulation.utils.AutoTokenizer.from_pretrained', side_effect=mock_autotokenizer_from_pretrained),
            patch('neuro_manipulation.utils.detect_multimodal_model', side_effect=mock_detect_multimodal_model),
            patch('emotion_memory_experiments.experiment.EmotionMemoryExperiment'),
            patch('neuro_manipulation.repe.pipelines.repe_pipeline_registry'),
        ]
        
        print("🔧 Setting up mocks...")
        
        with patch.multiple(
            'neuro_manipulation.utils',
            LLM=mock_llm_constructor,
            get_optimal_tensor_parallel_size=mock_get_optimal_tensor_parallel_size,
            AutoConfig=MagicMock(),
            AutoModelForCausalLM=MagicMock(),
            AutoTokenizer=MagicMock(),
            detect_multimodal_model=mock_detect_multimodal_model,
        ), patch('emotion_memory_experiments.experiment.EmotionMemoryExperiment') as mock_experiment:
            
            # Import and run the series runner
            from emotion_memory_experiments.memory_experiment_series_runner import MemoryExperimentSeriesRunner
            
            print("🏃 Running MemoryExperimentSeriesRunner with dry_run=False...")
            
            # Create mock experiment instance
            mock_exp_instance = MagicMock()
            mock_experiment.return_value = mock_exp_instance
            
            # Run the series runner (this should trigger vLLM loading)
            runner = MemoryExperimentSeriesRunner(
                config_path=config_path,
                series_name="integration_test", 
                resume=False,
                dry_run=False  # We want actual experiment setup, not just dry run
            )
            
            # This should trigger the vLLM loading via setup_experiment -> load_model_tokenizer
            runner.run_experiment_series()
            
        print("📊 Analyzing captured vLLM kwargs...")
        
        if not captured_vllm_kwargs:
            print("❌ FAILED: No vLLM kwargs were captured!")
            print("   This means vLLM constructor was never called.")
            return False
        
        print(f"✅ vLLM constructor was called with {len(captured_vllm_kwargs)} parameters")
        
        # Verify base parameters from loading_config were passed
        expected_base = {
            "gpu_memory_utilization": 0.85,
            "max_model_len": 16384, 
            "enforce_eager": False,
            "dtype": "bfloat16",
            "seed": 123,
            "disable_custom_all_reduce": True
        }
        
        print("🔍 Verifying base loading_config parameters...")
        for key, expected_value in expected_base.items():
            if key in captured_vllm_kwargs:
                actual_value = captured_vllm_kwargs[key]
                if actual_value == expected_value:
                    print(f"  ✅ {key}: {actual_value}")
                else:
                    print(f"  ❌ {key}: {actual_value} (expected {expected_value})")
                    return False
            else:
                print(f"  ❌ {key}: MISSING")
                return False
        
        # Verify additional_vllm_kwargs were merged
        expected_additional = {
            "rope-theta": 2000000,
            "rope-scaling": {"rope_type": "dynamic", "factor": 8.0},
            "custom-param": "test-value"
        }
        
        print("🎯 Verifying additional_vllm_kwargs were merged...")
        for key, expected_value in expected_additional.items():
            if key in captured_vllm_kwargs:
                actual_value = captured_vllm_kwargs[key]
                if actual_value == expected_value:
                    print(f"  ✅ {key}: {actual_value}")
                else:
                    print(f"  ❌ {key}: {actual_value} (expected {expected_value})")
                    return False
            else:
                print(f"  ❌ {key}: MISSING from vLLM kwargs")
                return False
        
        print("\n🎉 INTEGRATION TEST PASSED!")
        print("✅ additional_vllm_kwargs from YAML config successfully reached vLLM constructor")
        print("✅ Base loading_config parameters were correctly applied")
        print("✅ Custom additional parameters were correctly merged")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up temp file
        try:
            os.unlink(config_path)
        except:
            pass


if __name__ == "__main__":
    success = test_integration_vllm_kwargs_passing()
    if not success:
        sys.exit(1)
    print("\n✅ All integration tests passed!")