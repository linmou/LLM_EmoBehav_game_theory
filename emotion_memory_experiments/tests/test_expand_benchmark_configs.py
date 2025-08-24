#!/usr/bin/env python3
"""
# Test file responsible for: memory_experiment_series_runner.py expand_benchmark_configs method
# Purpose: Test BenchmarkConfig creation in pattern discovery scenarios and edge cases

Unit tests for expand_benchmark_configs method and related BenchmarkConfig initialization patterns.
This test suite specifically targets the bug where BenchmarkConfig is created with missing required arguments.
"""

import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import json

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import modules directly to avoid vllm dependency
import importlib.util

# Load data_models directly
data_models_path = Path(__file__).parent.parent / "data_models.py"
spec = importlib.util.spec_from_file_location("data_models", data_models_path)
data_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_models)

BenchmarkConfig = data_models.BenchmarkConfig
create_benchmark_config = data_models.create_benchmark_config

# Load memory_experiment_series_runner directly
runner_path = Path(__file__).parent.parent / "memory_experiment_series_runner.py"
spec = importlib.util.spec_from_file_location("memory_experiment_series_runner", runner_path)
runner_module = importlib.util.module_from_spec(spec)
sys.modules['data_models'] = data_models  # Make data_models available for import
spec.loader.exec_module(runner_module)

MemoryExperimentSeriesRunner = runner_module.MemoryExperimentSeriesRunner


class TestExpandBenchmarkConfigs(unittest.TestCase):
    """Test expand_benchmark_configs method and BenchmarkConfig initialization patterns"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a minimal config for testing
        self.minimal_config = {
            "models": ["/test/model"],
            "emotions": ["anger"],
            "intensities": [1.0],
            "benchmarks": []
        }
        
        # Create test runner instance
        self.runner = MemoryExperimentSeriesRunner(self.minimal_config)
    
    def test_benchmark_config_creation_with_all_required_args(self):
        """Test BenchmarkConfig creation with all required arguments (should pass)"""
        config = BenchmarkConfig(
            name="test_benchmark",
            task_type="test_task", 
            data_path=Path("test.jsonl"),
            sample_limit=10,
            augmentation_config=None,
            enable_auto_truncation=True,
            truncation_strategy="right",
            preserve_ratio=0.8
        )
        
        self.assertEqual(config.name, "test_benchmark")
        self.assertEqual(config.task_type, "test_task")
        self.assertIsInstance(config.data_path, Path)
        self.assertEqual(config.sample_limit, 10)
        self.assertTrue(config.enable_auto_truncation)
        self.assertEqual(config.truncation_strategy, "right")
        self.assertEqual(config.preserve_ratio, 0.8)
    
    def test_benchmark_config_creation_missing_required_args_should_fail(self):
        """Test BenchmarkConfig creation with missing required arguments (should fail)"""
        
        # This test should fail with TypeError - reproducing the actual bug
        with self.assertRaises(TypeError) as context:
            BenchmarkConfig(
                name="test_benchmark",
                task_type="test_task"
                # Missing: data_path, sample_limit, augmentation_config, 
                # enable_auto_truncation, truncation_strategy, preserve_ratio
            )
        
        error_message = str(context.exception)
        self.assertIn("missing", error_message)
        self.assertIn("required positional arguments", error_message)
        
        # Check that all expected missing arguments are mentioned
        expected_missing_args = [
            'data_path', 'sample_limit', 'augmentation_config', 
            'enable_auto_truncation', 'truncation_strategy', 'preserve_ratio'
        ]
        for arg in expected_missing_args:
            self.assertIn(arg, error_message)
    
    def test_expand_benchmark_configs_with_literal_task_type(self):
        """Test expand_benchmark_configs with literal (non-pattern) task types"""
        benchmarks = [{
            "name": "test_bench",
            "task_type": "literal_task",
            "sample_limit": 100
        }]
        
        # This should work without issues since no BenchmarkConfig creation happens
        expanded = self.runner.expand_benchmark_configs(benchmarks)
        
        self.assertEqual(len(expanded), 1)
        self.assertEqual(expanded[0]["name"], "test_bench")
        self.assertEqual(expanded[0]["task_type"], "literal_task")
    
    def test_expand_benchmark_configs_with_pattern_task_type_should_fail(self):
        """Test expand_benchmark_configs with pattern task types (should fail due to bug)"""
        benchmarks = [{
            "name": "test_bench", 
            "task_type": ".*test.*",  # This is a regex pattern
            "sample_limit": 100
        }]
        
        # This should fail with the exact same TypeError as reported
        with self.assertRaises(TypeError) as context:
            self.runner.expand_benchmark_configs(benchmarks)
        
        error_message = str(context.exception)
        self.assertIn("BenchmarkConfig.__init__()", error_message)
        self.assertIn("missing 6 required positional arguments", error_message)
        
        expected_missing = [
            'data_path', 'sample_limit', 'augmentation_config',
            'enable_auto_truncation', 'truncation_strategy', 'preserve_ratio'
        ]
        for arg in expected_missing:
            self.assertIn(arg, error_message)
    
    def test_is_pattern_task_type_detection(self):
        """Test the pattern detection logic"""
        # Test pattern detection
        self.assertTrue(self.runner._is_pattern_task_type(".*"))
        self.assertTrue(self.runner._is_pattern_task_type("test.*"))
        self.assertTrue(self.runner._is_pattern_task_type(".*qa.*"))
        self.assertTrue(self.runner._is_pattern_task_type("[abc]+"))
        
        # Test literal detection
        self.assertFalse(self.runner._is_pattern_task_type("literal_task"))
        self.assertFalse(self.runner._is_pattern_task_type("passkey"))
        self.assertFalse(self.runner._is_pattern_task_type("narrativeqa"))
        self.assertFalse(self.runner._is_pattern_task_type(""))
    
    def test_create_benchmark_config_factory_function(self):
        """Test the factory function for creating BenchmarkConfig with defaults"""
        config = create_benchmark_config(
            name="factory_test",
            task_type="test_task",
            data_path=Path("test.jsonl")
        )
        
        # Check that defaults are applied
        self.assertEqual(config.name, "factory_test")
        self.assertEqual(config.task_type, "test_task")
        self.assertIsNone(config.sample_limit)  # Default None
        self.assertIsNone(config.augmentation_config)  # Default None
        self.assertFalse(config.enable_auto_truncation)  # Default False
        self.assertEqual(config.truncation_strategy, "right")  # Default "right"
        self.assertEqual(config.preserve_ratio, 0.8)  # Default 0.8
    
    @patch('glob.glob')
    def test_discover_datasets_by_pattern_method(self, mock_glob):
        """Test the discover_datasets_by_pattern method in isolation"""
        # Mock file discovery
        mock_glob.return_value = [
            "data/memory_benchmarks/longbench_narrativeqa.jsonl",
            "data/memory_benchmarks/longbench_qasper.jsonl", 
            "data/memory_benchmarks/longbench_hotpotqa.jsonl"
        ]
        
        # Create a properly initialized BenchmarkConfig for testing
        config = create_benchmark_config(
            name="longbench",
            task_type=".*qa.*",  # Pattern that should match narrativeqa, qasper, hotpotqa
            data_path=Path("dummy.jsonl")
        )
        
        discovered = config.discover_datasets_by_pattern("data/memory_benchmarks")
        
        # Should discover task types matching the pattern
        expected_tasks = ["hotpotqa", "narrativeqa", "qasper"]  # Sorted order
        self.assertEqual(discovered, expected_tasks)
    
    def test_multiple_benchmark_expansion_mixed_types(self):
        """Test expansion with mix of literal and pattern task types"""
        benchmarks = [
            {
                "name": "test_bench1",
                "task_type": "literal_task",  # Literal - should pass through
                "sample_limit": 50
            },
            {
                "name": "test_bench2", 
                "task_type": ".*pattern.*",  # Pattern - should fail due to bug
                "sample_limit": 100
            }
        ]
        
        # This should fail when it hits the pattern task type
        with self.assertRaises(TypeError):
            self.runner.expand_benchmark_configs(benchmarks)


if __name__ == '__main__':
    unittest.main()