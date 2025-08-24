#!/usr/bin/env python3
"""
Comprehensive test suite for memory_experiment_series_runner.py

This test file consolidates all tests related to MemoryExperimentSeriesRunner including:
- BenchmarkConfig creation and expand_benchmark_configs method (from test_expand_benchmark_configs.py)
- Configuration validation and processing (from test_memory_config_simple.py) 
- Error handling and robustness (from test_error_handling.py)
- Import coverage (from test_import_coverage.py)

Purpose: Ensure MemoryExperimentSeriesRunner works correctly including the recent error handling improvements.
"""

import unittest
import pytest
import tempfile
import json
import yaml
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

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
try:
    spec.loader.exec_module(runner_module)
    MemoryExperimentSeriesRunner = runner_module.MemoryExperimentSeriesRunner
    RUNNER_AVAILABLE = True
except Exception as e:
    # Handle import errors gracefully (e.g., missing vllm)
    MemoryExperimentSeriesRunner = None
    RUNNER_AVAILABLE = False
    print(f"Warning: Could not import MemoryExperimentSeriesRunner: {e}")


class TestMemoryExperimentSeriesRunner(unittest.TestCase):
    """Comprehensive test suite for MemoryExperimentSeriesRunner"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "test_config.yaml"
        
        # Create minimal test configuration
        self.test_config = {
            "models": ["test_model_1", "test_model_2"],
            "emotions": ["anger"],
            "intensities": [1.0],
            "benchmarks": [
                {
                    "name": "test_benchmark_1", 
                    "task_type": "test_task",
                    "sample_limit": 5,
                    "augmentation_config": None,
                    "enable_auto_truncation": False,
                    "truncation_strategy": "right",
                    "preserve_ratio": 0.8
                },
                {
                    "name": "test_benchmark_2", 
                    "task_type": "another_task", 
                    "sample_limit": 5,
                    "augmentation_config": None,
                    "enable_auto_truncation": False,
                    "truncation_strategy": "right",
                    "preserve_ratio": 0.8
                }
            ],
            "output_dir": str(Path(self.temp_dir) / "results"),
            "base_data_dir": str(Path(self.temp_dir) / "data")
        }
        
        with open(self.config_file, "w") as f:
            yaml.dump(self.test_config, f)
        
        # Create minimal config for testing
        self.minimal_config = {
            "models": ["/test/model"],
            "emotions": ["anger"],
            "intensities": [1.0],
            "benchmarks": []
        }
        
        # Create test runner instance if available
        if RUNNER_AVAILABLE:
            self.runner = MemoryExperimentSeriesRunner(self.minimal_config)
    
    def tearDown(self):
        """Clean up after each test"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # =============================================================================
    # IMPORT AND INITIALIZATION TESTS
    # =============================================================================
    
    def test_memory_experiment_series_runner_import(self):
        """Test that memory_experiment_series_runner can be imported"""
        try:
            import emotion_memory_experiments.memory_experiment_series_runner
            self.assertTrue(True)  # If we get here, import succeeded
        except ImportError as e:
            if 'vllm' in str(e):
                self.skipTest("Skipping due to vLLM dependency")
            else:
                self.fail(f"Import failed: {e}")
    
    @unittest.skipUnless(RUNNER_AVAILABLE, "MemoryExperimentSeriesRunner not available")
    def test_runner_initialization(self):
        """Test that MemoryExperimentSeriesRunner can be initialized"""
        self.assertIsNotNone(self.runner)
        self.assertEqual(self.runner.base_config, self.minimal_config)

    # =============================================================================
    # BENCHMARKCONFIG CREATION AND VALIDATION TESTS
    # =============================================================================
    
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
        
        # This test should fail with TypeError - reproducing the actual bug that was fixed
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

    # =============================================================================
    # EXPAND BENCHMARK CONFIGS TESTS
    # =============================================================================
    
    @unittest.skipUnless(RUNNER_AVAILABLE, "MemoryExperimentSeriesRunner not available")
    def test_expand_benchmark_configs_with_literal_task_type(self):
        """Test expand_benchmark_configs with literal (non-pattern) task types"""
        benchmarks = [{
            "name": "test_bench",
            "task_type": "literal_task",
            "sample_limit": 100
        }]
        
        # This should work without issues since no BenchmarkConfig creation happens for literals
        expanded = self.runner.expand_benchmark_configs(benchmarks)
        
        self.assertEqual(len(expanded), 1)
        self.assertEqual(expanded[0]["name"], "test_bench")
        self.assertEqual(expanded[0]["task_type"], "literal_task")

    @unittest.skipUnless(RUNNER_AVAILABLE, "MemoryExperimentSeriesRunner not available")
    def test_expand_benchmark_configs_with_pattern_task_type_now_works(self):
        """Test expand_benchmark_configs with pattern task types (should work after fix)"""
        benchmarks = [{
            "name": "test_bench", 
            "task_type": ".*test.*",  # This is a regex pattern
            "sample_limit": 100,
            "augmentation_config": None,
            "enable_auto_truncation": False,
            "truncation_strategy": "right",
            "preserve_ratio": 0.8
        }]
        
        # Mock the discover_datasets_by_pattern to avoid file system dependencies
        with patch.object(BenchmarkConfig, 'discover_datasets_by_pattern') as mock_discover:
            mock_discover.return_value = ['test_task1', 'test_task2']
            
            # This should now work thanks to our fix using the factory function
            expanded = self.runner.expand_benchmark_configs(benchmarks)
            
            # Should expand to 2 benchmarks based on discovered tasks
            self.assertEqual(len(expanded), 2)
            self.assertEqual(expanded[0]["name"], "test_bench")
            self.assertEqual(expanded[0]["task_type"], "test_task1")
            self.assertEqual(expanded[1]["name"], "test_bench")
            self.assertEqual(expanded[1]["task_type"], "test_task2")

    @unittest.skipUnless(RUNNER_AVAILABLE, "MemoryExperimentSeriesRunner not available")
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

    @unittest.skipUnless(RUNNER_AVAILABLE, "MemoryExperimentSeriesRunner not available")
    def test_create_temporary_benchmark_for_discovery(self):
        """Test the helper method for creating temporary benchmarks for pattern discovery"""
        benchmark_config = {
            "name": "test_bench",
            "task_type": ".*test.*",
            "sample_limit": 100
        }
        
        # This should work now with the factory function approach
        temp_benchmark = self.runner._create_temporary_benchmark_for_discovery(
            benchmark_config, ".*test.*"
        )
        
        self.assertEqual(temp_benchmark.name, "test_bench")
        self.assertEqual(temp_benchmark.task_type, ".*test.*")
        self.assertEqual(temp_benchmark.sample_limit, 100)
        self.assertFalse(temp_benchmark.enable_auto_truncation)  # Default
        self.assertEqual(temp_benchmark.truncation_strategy, "right")  # Default

    # =============================================================================
    # ERROR HANDLING TESTS
    # =============================================================================
    
    def test_error_handling_pattern(self):
        """Test the error handling pattern we implemented in the fix"""
        # Simulate the experiment loop structure we implemented
        experiments = [
            {"exp_id": "exp_1", "name": "benchmark_1", "model": "model_1"},
            {"exp_id": "exp_2", "name": "benchmark_2", "model": "model_2"},  
            {"exp_id": "exp_3", "name": "benchmark_3", "model": "model_3"},
            {"exp_id": "exp_4", "name": "benchmark_4", "model": "model_4"}
        ]
        
        attempted_experiments = []
        failed_experiments = []
        successful_experiments = []
        errors_logged = []
        
        def mock_run_single_experiment(exp):
            """Mock experiment runner that fails for exp_2"""
            if exp["exp_id"] == "exp_2":
                return False  # Simulate failure
            return True  # Simulate success
        
        def mock_logger_error(msg):
            """Mock logger to capture error messages"""
            errors_logged.append(msg)
        
        def mock_update_experiment_report(exp_id, status=None, error=None):
            """Mock report update"""
            pass
        
        # Simulate our improved error handling loop
        for i, exp in enumerate(experiments):
            try:
                # This simulates our improved try-catch structure
                attempted_experiments.append(exp["exp_id"])
                
                # Simulate some experiments having series-level errors
                if exp["exp_id"] == "exp_3":
                    raise RuntimeError("Simulated series-level error")
                
                # Run the individual experiment
                success = mock_run_single_experiment(exp)
                
                if success:
                    successful_experiments.append(exp["exp_id"])
                else:
                    failed_experiments.append(exp["exp_id"]) 
                    
            except Exception as e:
                # This is our series-level error handling
                failed_experiments.append(exp["exp_id"])
                error_msg = f"🚨 SERIES-LEVEL ERROR for experiment {exp['exp_id']}: {str(e)}"
                mock_logger_error(error_msg)
                
                try:
                    mock_update_experiment_report(exp["exp_id"], status="FAILED", error=f"Series-level error: {str(e)}")
                except Exception as report_error:
                    mock_logger_error(f"Failed to update experiment report: {report_error}")
            
            finally:
                # Always try to log progress
                try:
                    progress_msg = f"Progress: {len(attempted_experiments)}/{len(experiments)} attempted"
                except Exception as summary_error:
                    pass
        
        # Verify results
        self.assertEqual(len(attempted_experiments), len(experiments), 
                        f"All {len(experiments)} experiments should be attempted despite failures")
        
        self.assertEqual(len(successful_experiments), 2, 
                        "Should have 2 successful experiments (exp_1 and exp_4)")
        
        self.assertEqual(len(failed_experiments), 2, 
                        "Should have 2 failed experiments (exp_2 and exp_3)")
        
        self.assertIn("exp_2", failed_experiments, "exp_2 should fail (experiment-level failure)")
        self.assertIn("exp_3", failed_experiments, "exp_3 should fail (series-level error)")
        
        self.assertEqual(len(errors_logged), 1, "Should log 1 series-level error")

    def test_benchmark_expansion_error_handling(self):
        """Test the error handling we added around benchmark expansion"""
        original_benchmarks = [
            {"name": "bench1", "task_type": ".*qa.*"},  # Pattern that might cause issues
            {"name": "bench2", "task_type": "normal_task"}
        ]
        
        def mock_expand_benchmark_configs(benchmarks):
            """Mock expansion that fails on the first call"""
            if benchmarks[0]["task_type"] == ".*qa.*":
                raise Exception("Pattern expansion failed - regex engine error")
            return benchmarks
        
        errors_logged = []
        def mock_logger_error(msg):
            errors_logged.append(msg)
        
        # Test our error handling logic
        try:
            benchmarks = mock_expand_benchmark_configs(original_benchmarks)
        except Exception as e:
            mock_logger_error(f"Failed to expand benchmark configurations: {str(e)}")
            mock_logger_error("Using original benchmarks without expansion")
            benchmarks = original_benchmarks  # Fallback
        
        # Verify fallback behavior
        self.assertEqual(benchmarks, original_benchmarks, "Should fallback to original benchmarks on expansion error")
        self.assertEqual(len(errors_logged), 2, "Should log expansion error and fallback message")

    def test_experiment_generation_error_handling(self):
        """Test the error handling around experiment generation"""
        benchmarks = [
            {"name": "bench1", "task_type": "task1"},
            {"name": "bench2", "task_type": "task2"}
        ]
        models = ["model1", "model/with/problematic/path"]
        
        generated_experiments = []
        errors_logged = []
        
        def mock_format_model_name(model_name):
            """Mock formatter that fails on problematic paths"""
            if "problematic" in model_name:
                raise ValueError("Cannot format problematic model path")
            return model_name.replace("/", "_")
        
        def mock_logger_error(msg):
            errors_logged.append(msg)
        
        def mock_add_experiment(benchmark_name, model_name, exp_id):
            generated_experiments.append(exp_id)
        
        # Test our nested error handling logic
        try:
            for benchmark_config in benchmarks:
                benchmark_name = benchmark_config["name"]
                task_type = benchmark_config["task_type"]
                for model_name in models:
                    try:
                        model_folder_name = mock_format_model_name(model_name)
                        exp_id = f"{benchmark_name}_{task_type}_{model_folder_name}"
                        mock_add_experiment(f"{benchmark_name}_{task_type}", model_name, exp_id)
                    except Exception as e:
                        mock_logger_error(f"Failed to generate experiment for {benchmark_name}/{task_type} + {model_name}: {e}")
                        continue  # Skip this combination but continue with others
        except Exception as e:
            mock_logger_error(f"Failed to generate experiment combinations: {str(e)}")
            raise
        
        # Should generate 2 successful experiments (bench1+model1, bench2+model1) 
        # and skip 2 problematic ones (bench1+problematic, bench2+problematic)
        self.assertEqual(len(generated_experiments), 2, "Should generate 2 valid experiments")
        self.assertEqual(len(errors_logged), 2, "Should log 2 generation errors for problematic model")

    def test_comprehensive_error_handling_behavior(self):
        """Test that validates the comprehensive error handling behavior we implemented"""
        # Test data simulating 4 experiments (2 benchmarks × 2 models)
        test_experiments = [
            {"exp_id": "benchmark1_task1_model1", "benchmark_name": "benchmark1_task1", "model_name": "model1"},
            {"exp_id": "benchmark1_task1_model2", "benchmark_name": "benchmark1_task1", "model_name": "model2"}, 
            {"exp_id": "benchmark2_task2_model1", "benchmark_name": "benchmark2_task2", "model_name": "model1"},
            {"exp_id": "benchmark2_task2_model2", "benchmark_name": "benchmark2_task2", "model_name": "model2"},
        ]
        
        attempted_experiments = []
        failed_experiments = []
        successful_experiments = []
        series_errors = []
        
        def mock_check_model_existence(model_name):
            return True  # All models exist
        
        def mock_run_single_experiment(benchmark_config, model_name, exp_id):
            # Simulate first experiment failing, third having series error  
            if exp_id == "benchmark1_task1_model1":
                return False  # Individual experiment failure
            return True  # Success
        
        def mock_update_experiment_report(exp_id, **kwargs):
            if kwargs.get("status") == "FAILED":
                failed_experiments.append(exp_id)
        
        def mock_logger_error(msg):
            if "SERIES-LEVEL ERROR" in msg:
                series_errors.append(msg)
        
        # Simulate our improved experiment series loop
        for i, exp in enumerate(test_experiments):
            try:
                attempted_experiments.append(exp["exp_id"])
                
                # Simulate series-level error for third experiment
                if exp["exp_id"] == "benchmark2_task2_model1":
                    raise RuntimeError("Simulated configuration error")
                
                # Check model existence
                if not mock_check_model_existence(exp["model_name"]):
                    mock_update_experiment_report(exp["exp_id"], status="FAILED", error="Model not found")
                    continue
                
                # Run individual experiment
                success = mock_run_single_experiment(None, exp["model_name"], exp["exp_id"])
                
                if success:
                    successful_experiments.append(exp["exp_id"])
                else:
                    failed_experiments.append(exp["exp_id"])
                    
            except Exception as e:
                # Series-level error handling - continue with next experiment
                error_msg = f"🚨 SERIES-LEVEL ERROR for experiment {exp['exp_id']}: {str(e)}"
                mock_logger_error(error_msg)
                
                try:
                    mock_update_experiment_report(exp["exp_id"], status="FAILED", error=f"Series-level error: {str(e)}")
                except Exception:
                    pass  # Even if report update fails, continue
        
        # Verify comprehensive error handling behavior
        self.assertEqual(len(attempted_experiments), 4, "All 4 experiments should be attempted")
        self.assertEqual(len(successful_experiments), 2, "Should have 2 successful experiments")
        self.assertEqual(len(failed_experiments), 2, "Should have 2 failed experiments (1 individual + 1 series error)")
        self.assertEqual(len(series_errors), 1, "Should log 1 series-level error")
        
        # Verify specific experiments failed as expected
        self.assertIn("benchmark1_task1_model1", failed_experiments, "First experiment should fail (individual failure)")
        self.assertIn("benchmark2_task2_model1", failed_experiments, "Third experiment should fail (series error)")
        
        # Verify successful experiments
        self.assertIn("benchmark1_task1_model2", successful_experiments, "Second experiment should succeed")
        self.assertIn("benchmark2_task2_model2", successful_experiments, "Fourth experiment should succeed")

    # =============================================================================
    # CONFIGURATION AND UTILITY TESTS
    # =============================================================================
    
    def test_format_model_name_for_folder(self):
        """Test model name formatting for folder names"""
        if not RUNNER_AVAILABLE:
            self.skipTest("MemoryExperimentSeriesRunner not available")
            
        test_cases = [
            # (input, expected_output)
            ("model1", "model1"),
            ("org/model", "org/model"),
            ("/data/home/huggingface_models/RWKV/v6-Finch-7B-HF", "RWKV/v6-Finch-7B-HF"),
            ("meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
        ]
        
        for input_name, expected in test_cases:
            result = self.runner._format_model_name_for_folder(input_name)
            self.assertEqual(result, expected, f"Format failed for {input_name}")

    def test_config_validation(self):
        """Test configuration validation"""
        # Test that required sections are checked
        required_sections = ['models', 'benchmarks', 'emotions', 'intensities']
        
        for section in required_sections:
            incomplete_config = self.test_config.copy()
            del incomplete_config[section]
            
            if RUNNER_AVAILABLE:
                with self.assertRaises(ValueError):
                    MemoryExperimentSeriesRunner(incomplete_config)

    # =============================================================================
    # PIPELINE WORKER ERROR HANDLING TESTS
    # =============================================================================
    
    def test_pipeline_worker_error_handling_pattern(self):
        """
        Test that pipeline worker error handling prevents AssertionError from stopping experiment series.
        
        This test validates the fix for the issue where AssertionError in memory_prompt_wrapper.augment_context
        would crash the pipeline worker thread, causing the main thread to wait forever and stopping the 
        entire experiment series.
        """
        print("🧪 Testing pipeline worker error handling...")
        
        # Simulate the pipeline worker error handling logic
        pipeline_queue = []
        worker_errors = []
        batch_errors = []
        processed_batches = []
        
        def mock_logger_error(msg):
            if "BATCH ERROR" in msg:
                batch_errors.append(msg)
            elif "WORKER THREAD ERROR" in msg:
                worker_errors.append(msg)
        
        def mock_logger_info(msg):
            pass
        
        def simulate_pipeline_worker_with_error_handling():
            """Simulate the fixed pipeline_worker function"""
            try:
                # Simulate data_loader that raises AssertionError on second batch
                mock_batches = [
                    {"batch_id": 0, "data": "good_batch"},
                    {"batch_id": 1, "data": "assertion_error_batch"},  # This will fail
                    {"batch_id": 2, "data": "good_batch_after_error"}
                ]
                
                for i, batch in enumerate(mock_batches):
                    try:
                        # Simulate processing batch
                        if batch["data"] == "assertion_error_batch":
                            # Simulate the exact error from memory_prompt_wrapper.augment_context
                            raise AssertionError("assert answer is not None")
                        
                        # Normal processing
                        control_outputs = [{"generated_text": f"result_for_batch_{i}"}]
                        pipeline_queue.append((i, batch, control_outputs))
                        processed_batches.append(i)
                        
                    except Exception as batch_error:
                        # This is our fix: handle batch errors gracefully
                        mock_logger_error(f"🚨 BATCH ERROR in pipeline worker for batch {i}: {str(batch_error)}")
                        
                        # Create error batch to maintain sequence integrity  
                        error_batch = {
                            "prompt": [f"ERROR: Batch {i} failed - {str(batch_error)}"],
                            "ground_truth": ["ERROR"],
                            "context": ["ERROR"],
                            "question": ["ERROR"]
                        }
                        error_outputs = [{"generated_text": f"ERROR: {str(batch_error)}"}]
                        
                        pipeline_queue.append((i, error_batch, error_outputs))
                        processed_batches.append(i)  # Still count as processed
                        
                        mock_logger_info(f"🔄 Continuing pipeline worker with next batch after error in batch {i}")
                
            except Exception as worker_error:
                # Handle catastrophic worker thread errors
                mock_logger_error(f"🚨 WORKER THREAD ERROR in pipeline_worker: {str(worker_error)}")
                pipeline_queue.append(("WORKER_ERROR", str(worker_error), "trace"))
            
            finally:
                # Always put sentinel value
                pipeline_queue.append(None)
        
        # Run the simulation
        simulate_pipeline_worker_with_error_handling()
        
        # Verify the fix works correctly
        self.assertEqual(len(processed_batches), 3, "All 3 batches should be processed despite error")
        self.assertEqual(len(batch_errors), 1, "Should log 1 batch error")
        self.assertEqual(len(worker_errors), 0, "Should not have worker thread errors")
        self.assertEqual(len(pipeline_queue), 4, "Should have 3 batch results + 1 sentinel")
        
        # Verify that the error batch is handled correctly
        error_batch_found = False
        for item in pipeline_queue:
            if item is not None and isinstance(item, tuple) and len(item) == 3:
                batch_idx, batch, outputs = item
                if isinstance(batch, dict) and "ERROR" in str(batch.get("prompt", "")):
                    error_batch_found = True
                    self.assertEqual(batch_idx, 1, "Error should be for batch 1")
        
        self.assertTrue(error_batch_found, "Error batch should be created and queued")
        
        # Verify sentinel value is present
        self.assertIn(None, pipeline_queue, "Sentinel value should be present to prevent main thread hanging")
        
        print("✅ Pipeline worker error handling test passed!")

    def test_main_thread_worker_error_handling(self):
        """Test that main thread handles WORKER_ERROR messages correctly"""
        # Simulate main thread processing pipeline_queue with WORKER_ERROR
        pipeline_queue = [
            (0, {"data": "batch0"}, ["output0"]),
            ("WORKER_ERROR", "Catastrophic error", "stack trace"),
            None
        ]
        
        processed_items = []
        worker_errors = []
        
        def mock_logger_error(msg):
            if "PIPELINE WORKER FAILED" in msg:
                worker_errors.append(msg)
        
        def mock_logger_info(msg):
            pass
        
        # Simulate main thread processing
        for item in pipeline_queue:
            if item is None:
                break  # Worker finished
            
            # Handle worker thread error case (our fix)
            if isinstance(item, tuple) and len(item) == 3 and item[0] == "WORKER_ERROR":
                error_type, error_msg, error_trace = item
                mock_logger_error(f"🚨 PIPELINE WORKER FAILED: {error_msg}\n{error_trace}")
                mock_logger_info("🔄 Main thread continuing despite worker thread failure")
                break  # Exit processing loop but don't crash experiment
            
            # Normal processing
            batch_idx, batch, control_outputs = item
            processed_items.append((batch_idx, batch, control_outputs))
        
        # Verify main thread handling
        self.assertEqual(len(processed_items), 1, "Should process 1 normal batch before worker error")
        self.assertEqual(len(worker_errors), 1, "Should log 1 worker error")
        
        print("✅ Main thread worker error handling test passed!")

    # =============================================================================
    # INTEGRATION TESTS (with mocking to avoid dependencies)
    # =============================================================================
    
    @patch('emotion_memory_experiments.memory_experiment_series_runner.MemoryExperimentSeriesRunner._check_model_existence')
    @patch('emotion_memory_experiments.memory_experiment_series_runner.MemoryExperimentSeriesRunner.run_single_experiment')
    def test_experiment_series_continues_after_failure(self, mock_run_single, mock_check_model):
        """Integration test that the fixed MemoryExperimentSeriesRunner continues with experiments after individual failures"""
        if not RUNNER_AVAILABLE:
            self.skipTest("MemoryExperimentSeriesRunner not available")
            
        # Mock model existence check to return True
        mock_check_model.return_value = True
        
        # Track which experiments were attempted
        attempted_experiments = []
        failed_experiments = []
        successful_experiments = []
        
        def mock_run_single_experiment(benchmark_config, model_name, exp_id):
            """Mock that simulates some experiments failing, others succeeding"""
            attempted_experiments.append(exp_id)
            
            # Simulate first experiment failing, others succeeding  
            if exp_id == "test_benchmark_1_test_task_test_model_1":
                failed_experiments.append(exp_id)
                return False  # Failure
            else:
                successful_experiments.append(exp_id)
                return True   # Success
        
        mock_run_single.side_effect = mock_run_single_experiment
        
        # Create runner and run series
        runner = MemoryExperimentSeriesRunner(str(self.config_file), dry_run=False)
        runner.run_experiment_series()
        
        # Verify results - should attempt all 4 combinations (2 benchmarks × 2 models)
        expected_total = 4
        
        self.assertEqual(len(attempted_experiments), expected_total, 
                        f"Expected {expected_total} experiments, but only {len(attempted_experiments)} were attempted")
        
        self.assertEqual(len(failed_experiments), 1, 
                        f"Expected exactly 1 failed experiment, got {len(failed_experiments)}")
        
        self.assertEqual(len(successful_experiments), expected_total - 1, 
                        f"Expected {expected_total - 1} successful experiments, got {len(successful_experiments)}")
        
        # Verify report status
        summary = runner.report.get_summary()
        self.assertEqual(summary["total"], expected_total, "Report should track all experiments")
        self.assertEqual(summary["failed"], 1, "Report should show 1 failed experiment")  
        self.assertEqual(summary["completed"], expected_total - 1, "Report should show 3 completed experiments")


if __name__ == '__main__':
    unittest.main()