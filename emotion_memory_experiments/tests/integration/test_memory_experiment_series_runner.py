#!/usr/bin/env python3
"""
Comprehensive test suite for memory_experiment_series_runner.py

Covers:
- BenchmarkConfig creation and pattern expansion behavior
- Error handling paths in the series runner
- Import coverage and integration behavior with mocks

Uses package imports; skips tests if runner unavailable in env.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml

from emotion_memory_experiments.data_models import BenchmarkConfig

try:
    from emotion_memory_experiments.memory_experiment_series_runner import (
        MemoryExperimentSeriesRunner,
    )
    RUNNER_AVAILABLE = True
except Exception as e:
    MemoryExperimentSeriesRunner = None  # type: ignore
    RUNNER_AVAILABLE = False
    print(f"Warning: MemoryExperimentSeriesRunner unavailable: {e}")


class TestMemoryExperimentSeriesRunner(unittest.TestCase):
    """Comprehensive test suite for MemoryExperimentSeriesRunner"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "test_config.yaml"

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
                    "preserve_ratio": 0.8,
                    "llm_eval_config": {"model": "gpt-4o-mini", "temperature": 0.1},
                },
                {
                    "name": "test_benchmark_2",
                    "task_type": "another_task",
                    "sample_limit": 5,
                    "augmentation_config": None,
                    "enable_auto_truncation": False,
                    "truncation_strategy": "right",
                    "preserve_ratio": 0.8,
                    "llm_eval_config": {"model": "gpt-4o-mini", "temperature": 0.1},
                },
            ],
            "output_dir": str(Path(self.temp_dir) / "results"),
            "base_data_dir": str(Path(self.temp_dir) / "data"),
            "loading_config": {
                "model_path": "/data/models/Qwen2.5-0.5B-Instruct",
                "gpu_memory_utilization": 0.8,
                "tensor_parallel_size": 1,
                "max_model_len": 1024,
                "enforce_eager": True,
                "quantization": None,
                "trust_remote_code": True,
                "dtype": "float16",
                "seed": 42,
                "disable_custom_all_reduce": False,
                "additional_vllm_kwargs": {},
            },
        }

        with open(self.config_file, "w") as f:
            yaml.dump(self.test_config, f)

        self.minimal_config = {
            "models": ["/test/model"],
            "emotions": ["anger"],
            "intensities": [1.0],
            "benchmarks": [],
            "loading_config": self.test_config["loading_config"],
        }

        if RUNNER_AVAILABLE:
            # Runner expects a config path; create a minimal YAML
            self.minimal_cfg_file = Path(self.temp_dir) / "minimal.yaml"
            with open(self.minimal_cfg_file, "w") as f:
                yaml.dump(self.minimal_config, f)
            self.runner = MemoryExperimentSeriesRunner(str(self.minimal_cfg_file))

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_memory_experiment_series_runner_import(self):
        try:
            import emotion_memory_experiments.memory_experiment_series_runner  # noqa: F401
            self.assertTrue(True)
        except ImportError as e:
            if "vllm" in str(e).lower():
                self.skipTest("Skipping due to vLLM dependency")
            else:
                self.fail(f"Import failed: {e}")

    @unittest.skipUnless(RUNNER_AVAILABLE, "MemoryExperimentSeriesRunner not available")
    def test_runner_initialization(self):
        self.assertIsNotNone(self.runner)

    @unittest.skipUnless(RUNNER_AVAILABLE, "MemoryExperimentSeriesRunner not available")
    def test_expand_benchmark_configs_with_literal_task_type(self):
        benchmarks = [{"name": "test_bench", "task_type": "literal_task", "sample_limit": 100}]
        expanded = self.runner.expand_benchmark_configs(benchmarks)
        self.assertEqual(len(expanded), 1)
        self.assertEqual(expanded[0]["task_type"], "literal_task")

    @unittest.skipUnless(RUNNER_AVAILABLE, "MemoryExperimentSeriesRunner not available")
    def test_expand_benchmark_configs_with_pattern_task_type_now_works(self):
        benchmarks = [
            {
                "name": "test_bench",
                "task_type": ".*test.*",
                "sample_limit": 100,
                "augmentation_config": None,
                "enable_auto_truncation": False,
                "truncation_strategy": "right",
                "preserve_ratio": 0.8,
            }
        ]
        with patch.object(BenchmarkConfig, "discover_datasets_by_pattern") as mock_discover:
            mock_discover.return_value = ["test_task1", "test_task2"]
            expanded = self.runner.expand_benchmark_configs(benchmarks)
            self.assertEqual(len(expanded), 2)
            self.assertEqual(expanded[0]["task_type"], "test_task1")
            self.assertEqual(expanded[1]["task_type"], "test_task2")

    @unittest.skipUnless(RUNNER_AVAILABLE, "MemoryExperimentSeriesRunner not available")
    def test_is_pattern_task_type_detection(self):
        self.assertTrue(self.runner._is_pattern_task_type(".*"))
        self.assertTrue(self.runner._is_pattern_task_type("test.*"))
        self.assertTrue(self.runner._is_pattern_task_type(".*qa.*"))
        self.assertTrue(self.runner._is_pattern_task_type("[abc]+"))
        self.assertFalse(self.runner._is_pattern_task_type("literal_task"))
        self.assertFalse(self.runner._is_pattern_task_type("passkey"))
        self.assertFalse(self.runner._is_pattern_task_type("narrativeqa"))
        self.assertFalse(self.runner._is_pattern_task_type(""))

    @unittest.skipUnless(RUNNER_AVAILABLE, "MemoryExperimentSeriesRunner not available")
    def test_create_temporary_benchmark_for_discovery(self):
        benchmark_config = {"name": "test_bench", "task_type": ".*test.*", "sample_limit": 100}
        temp_benchmark = self.runner._create_temporary_benchmark_for_discovery(
            benchmark_config, ".*test.*"
        )
        self.assertEqual(temp_benchmark.name, "test_bench")
        self.assertEqual(temp_benchmark.task_type, ".*test.*")
        self.assertEqual(temp_benchmark.sample_limit, 100)

    @unittest.skipUnless(RUNNER_AVAILABLE, "MemoryExperimentSeriesRunner not available")
    def test_experiment_series_continues_after_failure(self):
        with patch(
            "emotion_memory_experiments.memory_experiment_series_runner.MemoryExperimentSeriesRunner._check_model_existence"
        ) as mock_check_model, patch(
            "emotion_memory_experiments.memory_experiment_series_runner.MemoryExperimentSeriesRunner.run_single_experiment"
        ) as mock_run_single:
            mock_check_model.return_value = True
            attempted, failed, succeeded = [], [], []

            def _run(benchmark_config, model_name, exp_id):
                attempted.append(exp_id)
                if exp_id == "test_benchmark_1_test_task_test_model_1":
                    failed.append(exp_id)
                    return False
                else:
                    succeeded.append(exp_id)
                    return True

            mock_run_single.side_effect = _run
            runner = MemoryExperimentSeriesRunner(str(self.config_file), dry_run=False)
            runner.run_experiment_series()

            expected_total = 4
            self.assertEqual(len(attempted), expected_total)
            self.assertEqual(len(failed), 1)
            self.assertEqual(len(succeeded), expected_total - 1)
            summary = runner.report.get_summary()
            self.assertEqual(summary["total"], expected_total)
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(summary["completed"], expected_total - 1)

    @unittest.skipUnless(RUNNER_AVAILABLE, "MemoryExperimentSeriesRunner not available")
    def test_dry_run_errors_bubble_up(self):
        # Test for emotion_memory_experiments.memory_experiment_series_runner.dry_run_series error propagation when setup fails.
        runner = MemoryExperimentSeriesRunner(str(self.config_file), dry_run=True)

        with patch.object(runner, "setup_experiment", side_effect=RuntimeError("boom")):
            with self.assertRaisesRegex(RuntimeError, "Config 1 failed"):
                runner.dry_run_series()


if __name__ == "__main__":
    unittest.main()
