#!/usr/bin/env python3
"""
Test file for config factory functions.
Ensures factory functions work correctly and maintain separation of concerns.
"""
import unittest
from pathlib import Path

from ..dataset_factory import (
    create_vllm_config_from_dict,
    create_benchmark_config_from_dict,
    create_experiment_config_from_dict
)
from ..data_models import VLLMLoadingConfig, BenchmarkConfig


class TestConfigFactories(unittest.TestCase):
    """Test config factory functions"""
    
    def test_create_vllm_config_from_dict_basic(self):
        """Test basic vLLM config creation"""
        config_dict = {
            "loading_config": {
                "gpu_memory_utilization": 0.85,
                "max_model_len": 16384,
                "dtype": "bfloat16"
            }
        }
        
        vllm_config = create_vllm_config_from_dict(config_dict, "/test/model")
        
        self.assertIsInstance(vllm_config, VLLMLoadingConfig)
        self.assertEqual(vllm_config.model_path, "/test/model")
        self.assertEqual(vllm_config.gpu_memory_utilization, 0.85)
        self.assertEqual(vllm_config.max_model_len, 16384)
        self.assertEqual(vllm_config.dtype, "bfloat16")
    
    def test_create_vllm_config_from_dict_none_when_missing(self):
        """Test that None is returned when no loading_config"""
        config_dict = {}
        vllm_config = create_vllm_config_from_dict(config_dict, "/test/model")
        self.assertIsNone(vllm_config)
    
    def test_create_vllm_config_additional_kwargs(self):
        """Test additional vLLM kwargs support"""
        config_dict = {
            "loading_config": {
                "additional_vllm_kwargs": {
                    "max_num_seqs": 256,
                    "enable_chunked_prefill": True
                }
            }
        }
        
        vllm_config = create_vllm_config_from_dict(config_dict, "/test/model")
        kwargs = vllm_config.to_vllm_kwargs()
        
        self.assertEqual(kwargs["max_num_seqs"], 256)
        self.assertTrue(kwargs["enable_chunked_prefill"])
    
    def test_create_benchmark_config_from_dict(self):
        """Test benchmark config creation with truncation settings"""
        benchmark_data = {
            "name": "testbench",
            "task_type": "passkey",
            "data_path": "test.jsonl",
            "sample_limit": 100
        }
        
        config_dict = {
            "loading_config": {
                "enable_auto_truncation": True,
                "truncation_strategy": "left",
                "preserve_ratio": 0.9
            }
        }
        
        bench_config = create_benchmark_config_from_dict(benchmark_data, config_dict)
        
        self.assertIsInstance(bench_config, BenchmarkConfig)
        self.assertEqual(bench_config.name, "testbench")
        self.assertEqual(bench_config.task_type, "passkey")
        self.assertEqual(bench_config.sample_limit, 100)
        self.assertTrue(bench_config.enable_auto_truncation)
        self.assertEqual(bench_config.truncation_strategy, "left")
        self.assertEqual(bench_config.preserve_ratio, 0.9)
    
    def test_separation_of_concerns(self):
        """Test that truncation settings go to benchmark, not vLLM config"""
        config_dict = {
            "loading_config": {
                "gpu_memory_utilization": 0.8,
                "enable_auto_truncation": True,
                "truncation_strategy": "left"
            }
        }
        
        benchmark_data = {
            "name": "test",
            "task_type": "test",
            "data_path": "test.jsonl"
        }
        
        # vLLM config should NOT have truncation settings
        vllm_config = create_vllm_config_from_dict(config_dict, "/test/model")
        vllm_kwargs = vllm_config.to_vllm_kwargs()
        self.assertNotIn("enable_auto_truncation", vllm_kwargs)
        self.assertNotIn("truncation_strategy", vllm_kwargs)
        
        # Benchmark config SHOULD have truncation settings
        bench_config = create_benchmark_config_from_dict(benchmark_data, config_dict)
        self.assertTrue(bench_config.enable_auto_truncation)
        self.assertEqual(bench_config.truncation_strategy, "left")


if __name__ == "__main__":
    unittest.main()