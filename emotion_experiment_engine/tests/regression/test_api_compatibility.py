#!/usr/bin/env python3
"""
API Compatibility Regression Tests

These tests ensure that public APIs remain stable across versions, preventing
breaking changes that would affect research reproducibility and user code.
"""

import inspect
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List, Callable

# Import all public APIs to test
from emotion_experiment_engine.dataset_factory import (
    create_dataset_from_config, 
    DATASET_REGISTRY,
    register_dataset_class
)
from emotion_experiment_engine.data_models import BenchmarkConfig, ExperimentConfig
from emotion_experiment_engine.evaluation_utils import llm_evaluate_response, llm_evaluate_batch
from emotion_experiment_engine.experiment import EmotionMemoryExperiment
from emotion_experiment_engine.datasets.base import BaseBenchmarkDataset


@pytest.mark.regression
class TestDatasetFactoryAPICompatibility:
    """Ensure dataset factory API remains stable"""
    
    def test_create_dataset_from_config_signature(self):
        """Factory function signature must remain stable"""
        sig = inspect.signature(create_dataset_from_config)
        params = list(sig.parameters.keys())
        
        # Required parameters that must exist
        required_params = ['config']
        for param in required_params:
            assert param in params, f"Required parameter '{param}' missing from create_dataset_from_config"
        
        # Verify parameter types
        config_param = sig.parameters['config']
        assert config_param.annotation in [BenchmarkConfig, inspect.Parameter.empty], \
            "config parameter must accept BenchmarkConfig"
    
    def test_dataset_registry_interface_stability(self):
        """Dataset registry must maintain interface"""
        # Registry must exist and be accessible
        assert DATASET_REGISTRY is not None, "DATASET_REGISTRY must be accessible"
        assert isinstance(DATASET_REGISTRY, dict), "DATASET_REGISTRY must be a dictionary"
        
        # Core datasets must be registered
        expected_datasets = ["infinitebench", "longbench", "locomo", "emotion_check"]
        registry_keys = [k.lower() for k in DATASET_REGISTRY.keys()]
        
        for dataset in expected_datasets:
            assert dataset in registry_keys, f"Dataset '{dataset}' missing from registry"
    
    def test_register_dataset_class_signature(self):
        """Dynamic registration function must remain stable"""
        sig = inspect.signature(register_dataset_class)
        params = list(sig.parameters.keys())
        
        expected_params = ['name', 'dataset_class']
        assert len(params) >= len(expected_params), "register_dataset_class missing required parameters"
        
        for param in expected_params:
            assert param in params, f"Parameter '{param}' missing from register_dataset_class"
    
    @patch('emotion_experiment_engine.datasets.base.BaseBenchmarkDataset._load_raw_data')
    def test_dataset_creation_backward_compatibility(self, mock_load):
        """Ensure dataset creation works with legacy config formats"""
        mock_load.return_value = [
            {"id": 1, "input": "test", "answer": "test", "task_name": "test"}
        ]
        
        # Test various config formats that should be supported
        legacy_configs = [
            # Minimal config
            BenchmarkConfig(name="infinitebench", task_type="passkey", data_path=None),
            
            # Config with sample limit
            BenchmarkConfig(name="longbench", task_type="qa", sample_limit=10, data_path=None),
            
            # Config with evaluation method
            BenchmarkConfig(
                name="locomo", 
                task_type="conversation",
                evaluation_method="f1_score",
                data_path=None
            ),
        ]
        
        for config in legacy_configs:
            try:
                dataset = create_dataset_from_config(config)
                assert dataset is not None, f"Failed to create dataset with config: {config}"
                assert isinstance(dataset, BaseBenchmarkDataset), \
                    f"Created dataset is not a BaseBenchmarkDataset: {type(dataset)}"
            except Exception as e:
                pytest.fail(f"Legacy config failed: {config} - Error: {str(e)}")


@pytest.mark.regression
class TestDataModelCompatibility:
    """Ensure data model interfaces remain stable"""
    
    def test_benchmark_config_fields(self):
        """BenchmarkConfig must maintain required fields"""
        # Test that we can create config with minimal required fields
        config = BenchmarkConfig(name="test", task_type="test", data_path=None)
        
        # Required fields must exist
        required_fields = ["name", "task_type"]
        for field in required_fields:
            assert hasattr(config, field), f"BenchmarkConfig missing field '{field}'"
            assert getattr(config, field) is not None, f"BenchmarkConfig field '{field}' is None"
    
    def test_benchmark_config_optional_fields(self):
        """Optional fields should have reasonable defaults"""
        config = BenchmarkConfig(name="test", task_type="test", data_path=None)
        
        # Optional fields with defaults
        optional_fields = {
            "sample_limit": None,
            "evaluation_method": None,
            "base_data_dir": str,  # Should have some default
        }
        
        for field, expected_type in optional_fields.items():
            assert hasattr(config, field), f"BenchmarkConfig missing optional field '{field}'"
            
            value = getattr(config, field)
            if expected_type is not None and value is not None:
                assert isinstance(value, expected_type), \
                    f"Field '{field}' has wrong type: {type(value)} != {expected_type}"
    
    def test_experiment_config_backward_compatibility(self):
        """ExperimentConfig must support legacy initialization"""
        # Test minimal initialization
        benchmark_config = BenchmarkConfig(name="test", task_type="test", data_path=None)
        
        try:
            config = ExperimentConfig(
                model_path="test_model",
                emotions=["anger", "neutral"],
                intensities=[1.0],
                benchmark=benchmark_config,
                output_dir="test_output"
            )
            
            # Verify required fields exist
            assert hasattr(config, "model_path")
            assert hasattr(config, "emotions")
            assert hasattr(config, "benchmark")
            
        except Exception as e:
            pytest.fail(f"ExperimentConfig creation failed: {str(e)}")


@pytest.mark.regression
class TestEvaluationAPICompatibility:
    """Ensure evaluation API remains stable"""
    
    def test_llm_evaluate_response_signature(self):
        """LLM evaluation function signature must be stable"""
        sig = inspect.signature(llm_evaluate_response)
        params = list(sig.parameters.keys())
        
        # Core parameters that must exist
        expected_params = ["system_prompt", "query", "llm_eval_config"]
        for param in expected_params:
            assert param in params, f"Parameter '{param}' missing from llm_evaluate_response"
    
    @patch('emotion_experiment_engine.evaluation_utils.openai.ChatCompletion.create')
    def test_llm_evaluate_response_return_format(self, mock_openai):
        """LLM evaluation must return consistent format"""
        # Mock OpenAI response
        mock_openai.return_value.choices = [
            MagicMock(message=MagicMock(content='{"emotion": "neutral", "confidence": 0.8}'))
        ]
        
        result = llm_evaluate_response(
            system_prompt="Test",
            query="Test query",
            llm_eval_config={"model": "gpt-4o-mini", "temperature": 0.0}
        )
        
        # Verify return format
        assert isinstance(result, dict), "llm_evaluate_response must return a dictionary"
        
        # Expected fields in response
        expected_fields = ["emotion", "confidence"]
        for field in expected_fields:
            assert field in result, f"Response missing field '{field}'"
        
        # Type validation
        assert isinstance(result["emotion"], str), "emotion field must be string"
        assert isinstance(result["confidence"], (int, float)), "confidence must be numeric"
        assert 0.0 <= result["confidence"] <= 1.0, "confidence must be in [0, 1] range"
    
    def test_llm_evaluate_batch_signature(self):
        """Batch evaluation function must maintain signature"""
        sig = inspect.signature(llm_evaluate_batch)
        params = list(sig.parameters.keys())
        
        # Key parameters for batch processing
        expected_params = ["evaluation_requests", "llm_eval_config"]
        for param in expected_params:
            assert param in params, f"Parameter '{param}' missing from llm_evaluate_batch"


@pytest.mark.regression
class TestBaseBenchmarkDatasetInterface:
    """Ensure base dataset interface remains stable"""
    
    def test_abstract_methods_stability(self):
        """Abstract base class must define required methods"""
        # Check that BaseBenchmarkDataset has required abstract methods
        abstract_methods = getattr(BaseBenchmarkDataset, '__abstractmethods__', set())
        
        expected_abstract = {
            "_load_and_parse_data",
            "evaluate_response", 
            "get_task_metrics"
        }
        
        for method in expected_abstract:
            assert method in abstract_methods or hasattr(BaseBenchmarkDataset, method), \
                f"BaseBenchmarkDataset missing required method '{method}'"
    
    def test_pytorch_dataset_interface(self):
        """Must maintain PyTorch Dataset interface"""
        # BaseBenchmarkDataset must have PyTorch Dataset methods
        required_methods = ["__len__", "__getitem__"]
        
        for method in required_methods:
            assert hasattr(BaseBenchmarkDataset, method), \
                f"BaseBenchmarkDataset missing PyTorch Dataset method '{method}'"
            assert callable(getattr(BaseBenchmarkDataset, method)), \
                f"Method '{method}' is not callable"
    
    def test_evaluate_response_signature_stability(self):
        """evaluate_response method signature must be stable across all datasets"""
        # Get all dataset classes
        from emotion_experiment_engine.datasets import infinitebench, longbench, locomo, emotion_check
        
        dataset_classes = [
            infinitebench.InfiniteBenchDataset,
            longbench.LongBenchDataset,
            locomo.LoCoMoDataset,
            emotion_check.EmotionCheckDataset
        ]
        
        # Check signature consistency
        base_sig = inspect.signature(BaseBenchmarkDataset.evaluate_response)
        base_params = list(base_sig.parameters.keys())
        
        for dataset_class in dataset_classes:
            if hasattr(dataset_class, 'evaluate_response'):
                sig = inspect.signature(dataset_class.evaluate_response)
                params = list(sig.parameters.keys())
                
                # Must have at least the base parameters
                for param in base_params:
                    assert param in params, \
                        f"{dataset_class.__name__}.evaluate_response missing parameter '{param}'"


@pytest.mark.regression
class TestExperimentClassCompatibility:
    """Ensure experiment orchestration API remains stable"""
    
    def test_experiment_initialization_compatibility(self):
        """EmotionMemoryExperiment must accept standard config"""
        benchmark_config = BenchmarkConfig(name="test", task_type="test", data_path=None)
        
        experiment_config = ExperimentConfig(
            model_path="test_model",
            emotions=["anger"],
            intensities=[1.0],
            benchmark=benchmark_config,
            output_dir="test_output"
        )
        
        # Mock heavy dependencies
        with patch('emotion_experiment_engine.experiment.load_emotion_readers'), \
             patch('emotion_experiment_engine.experiment.setup_model_and_tokenizer'), \
             patch('emotion_experiment_engine.experiment.get_pipeline'):
            
            try:
                experiment = EmotionMemoryExperiment(experiment_config)
                assert experiment is not None, "Failed to create EmotionMemoryExperiment"
                
                # Verify key attributes exist
                assert hasattr(experiment, 'config'), "Missing config attribute"
                assert hasattr(experiment, 'dataset'), "Missing dataset attribute" 
                
            except Exception as e:
                pytest.fail(f"EmotionMemoryExperiment initialization failed: {str(e)}")
    
    def test_experiment_public_methods_stability(self):
        """Public methods must remain available"""
        # Key public methods that should remain stable
        expected_methods = [
            "__init__",
            "run_sanity_check", 
            # Add other public methods as they stabilize
        ]
        
        for method in expected_methods:
            assert hasattr(EmotionMemoryExperiment, method), \
                f"EmotionMemoryExperiment missing method '{method}'"
            assert callable(getattr(EmotionMemoryExperiment, method)), \
                f"Method '{method}' is not callable"


@pytest.mark.regression
class TestVersionCompatibilityMatrix:
    """Test compatibility across dependency versions"""
    
    def test_python_version_support(self):
        """Ensure Python version requirements haven't changed"""
        import sys
        
        # Minimum Python version requirement
        min_python = (3, 8)
        current_python = sys.version_info[:2]
        
        assert current_python >= min_python, \
            f"Python {current_python} < minimum required {min_python}"
    
    def test_import_stability(self):
        """All public imports must remain available"""
        # Test that key imports work without errors
        try:
            # Core functionality
            from emotion_experiment_engine import experiment, dataset_factory, data_models
            from emotion_experiment_engine.datasets import base, infinitebench, longbench, locomo
            from emotion_experiment_engine import evaluation_utils, config_loader
            
            # Test the new file mentioned by user
            from emotion_experiment_engine.tests import test_answer_wrapper_comprehensive
            
        except ImportError as e:
            pytest.fail(f"Import regression detected: {str(e)}")
    
    @patch('emotion_experiment_engine.evaluation_utils.openai')
    def test_openai_api_version_compatibility(self, mock_openai):
        """Test OpenAI API version compatibility"""
        # Mock OpenAI response structure that should remain stable
        mock_openai.ChatCompletion.create.return_value.choices = [
            MagicMock(message=MagicMock(content='{"emotion": "test", "confidence": 1.0}'))
        ]
        
        try:
            result = llm_evaluate_response(
                system_prompt="test",
                query="test", 
                llm_eval_config={"model": "gpt-4o-mini"}
            )
            assert isinstance(result, dict), "OpenAI API compatibility broken"
            
        except Exception as e:
            pytest.fail(f"OpenAI API compatibility issue: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "regression"])