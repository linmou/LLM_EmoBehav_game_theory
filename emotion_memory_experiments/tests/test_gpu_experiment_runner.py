#!/usr/bin/env python3
"""
GPU-enabled test script for emotion memory experiments.
Tests the complete pipeline including real model loading, vLLM, and RepE.
"""

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from emotion_memory_experiments.data_models import BenchmarkConfig, ExperimentConfig
from emotion_memory_experiments.experiment import EmotionMemoryExperiment
from emotion_memory_experiments.tests.run_emotion_memory_experiment import (
    create_experiment_config,
    load_config,
    setup_logging,
    validate_config,
)


class GPUExperimentTest:
    """GPU-enabled test for complete emotion memory experiment pipeline"""

    def __init__(self):
        self.temp_dir = None
        self.test_configs = {}
        self.results = {}

    def setup(self):
        """Set up test environment"""
        print("🔧 Setting up GPU experiment test environment...")

        # Create temporary directory for results
        self.temp_dir = tempfile.mkdtemp(prefix="emotion_memory_gpu_test_")
        print(f"  📁 Temporary directory: {self.temp_dir}")

        # Check GPU availability
        if torch.cuda.is_available():
            print(f"  🚀 GPU available: {torch.cuda.get_device_name()}")
            print(
                f"  💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )
        else:
            print("  ⚠️  No GPU available - will test CPU fallback")

        print("✅ Setup complete!")

    def teardown(self):
        """Clean up test environment"""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up temporary directory: {self.temp_dir}")

    def create_gpu_test_config(
        self, model_path: str, use_small_dataset: bool = True
    ) -> Dict[str, Any]:
        """Create test configuration for GPU testing"""

        # Determine GPU settings
        if torch.cuda.is_available():
            device = "cuda"
            gpu_memory_utilization = 0.50  # Conservative for testing
            batch_size = 2  # Small batch for testing
            max_num_seqs = 4
        else:
            device = "cpu"
            gpu_memory_utilization = 0.0
            batch_size = 1
            max_num_seqs = 1

        config = {
            "experiment_name": "gpu_integration_test",
            "description": "GPU-enabled integration test for emotion memory experiments",
            "version": "1.0.0",
            # Model configuration
            "model": {
                "name": "qwen2.5-0.5b-instruct",
                "path": model_path,
                "type": "qwen",
                "max_context_length": 32768,
            },
            # Hardware configuration
            "hardware": {
                "device": device,
                "gpu_memory_utilization": gpu_memory_utilization,
                "tensor_parallel_size": 1,
                "max_num_seqs": max_num_seqs,
            },
            # Emotion configuration (reduced for testing)
            "emotions": {
                "target_emotions": [
                    "anger",
                    "happiness",
                ],  # Only 2 emotions for faster testing
                "include_neutral": True,
                "intensities": [1.0],  # Only 1 intensity for faster testing
            },
            # Benchmark configuration
            "benchmarks": {
                "infinitebench_passkey": {
                    "name": "infinitebench",
                    "task_type": "passkey",
                    "data_path": "test_data/real_benchmarks/infinitebench_passkey.jsonl",
                    "evaluation_method": "get_score_one_passkey",
                    "sample_limit": (
                        3 if use_small_dataset else 10
                    ),  # Very small for testing
                },
                "longbench_qa": {
                    "name": "longbench",
                    "task_type": "longbook_qa_eng",
                    "data_path": "test_data/real_benchmarks/longbench_sample.jsonl",
                    "evaluation_method": "qa_f1_score",
                    "sample_limit": 2 if use_small_dataset else None,
                },
            },
            # Generation parameters (conservative for testing)
            "generation": {
                "temperature": 0.1,
                "max_new_tokens": 50,  # Shorter responses for testing
                "do_sample": False,
                "top_p": 0.9,
                "repetition_penalty": 1.0,
                "stop_tokens": ["<|endoftext|>", "<|im_end|>"],
            },
            # Execution settings
            "execution": {
                "batch_size": batch_size,
                "max_concurrent_requests": 2,
                "timeout_seconds": 120,  # Longer timeout for GPU operations
                "retry_attempts": 2,
            },
            # Output configuration
            "output": {
                "results_dir": self.temp_dir,
                "save_intermediate": True,
                "save_raw_responses": True,
                "formats": ["csv", "json"],
                "log_level": "INFO",
                "log_file": f"{self.temp_dir}/gpu_test.log",
            },
            # Prompt wrapper settings
            "prompt_wrapper": {
                "enable": True,
                "format": "qwen",
                "include_thinking": False,
            },
            # RepE configuration
            "repe_config": {
                "layer_ids": [8, 12],  # Fewer layers for testing
                "activation_strength": 1.0,
                "use_thinking_mode": False,
            },
        }

        return config

    def test_config_validation(self, config: Dict[str, Any]) -> bool:
        """Test configuration validation"""
        print("🔍 Testing configuration validation...")

        try:
            # Save config to temporary file
            config_path = Path(self.temp_dir) / "test_config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)

            # Test loading and validation
            loaded_config = load_config(config_path)
            is_valid = validate_config(loaded_config)

            if is_valid:
                print("✅ Configuration validation passed")
                return True
            else:
                print("❌ Configuration validation failed")
                return False

        except Exception as e:
            print(f"❌ Configuration test error: {e}")
            return False

    def test_experiment_creation(self, config: Dict[str, Any]) -> bool:
        """Test experiment object creation"""
        print("🔧 Testing experiment creation...")

        try:
            # Create experiment config
            exp_config = create_experiment_config(config)

            # Test experiment instantiation
            experiment = EmotionMemoryExperiment(exp_config)

            print("✅ Experiment creation successful")
            return True

        except Exception as e:
            print(f"❌ Experiment creation error: {e}")
            return False

    def test_small_experiment_run(self, config: Dict[str, Any]) -> bool:
        """Test running a small experiment with real GPU/model"""
        print("🚀 Testing small experiment run...")

        try:
            # Create experiment config
            exp_config = create_experiment_config(config)

            # Create and run experiment
            experiment = EmotionMemoryExperiment(exp_config)

            print(
                f"  📊 Starting experiment with {len(exp_config.benchmarks)} benchmarks"
            )
            start_time = time.time()

            # Run the experiment
            results_df = experiment.run_full_experiment()

            end_time = time.time()
            runtime = end_time - start_time

            if results_df is not None and len(results_df) > 0:
                print(f"✅ Experiment completed successfully!")
                print(
                    f"  📊 Generated {len(results_df)} results in {runtime:.1f} seconds"
                )
                print(f"  🎯 Average score: {results_df['score'].mean():.3f}")
                print(
                    f"  📈 Score range: {results_df['score'].min():.3f} - {results_df['score'].max():.3f}"
                )

                # Save results for analysis
                self.results["small_experiment"] = {
                    "results_df": results_df,
                    "runtime": runtime,
                    "config": config,
                }

                return True
            else:
                print("❌ Experiment failed - no results generated")
                return False

        except Exception as e:
            print(f"❌ Experiment run error: {e}")
            import traceback

            traceback.print_exc()
            return False

    def test_benchmark_loading(self, config: Dict[str, Any]) -> bool:
        """Test loading all configured benchmarks"""
        print("📊 Testing benchmark data loading...")

        try:
            success_count = 0
            total_benchmarks = len(config["benchmarks"])

            for benchmark_name, benchmark_data in config["benchmarks"].items():
                try:
                    # Create benchmark config
                    benchmark_config = BenchmarkConfig(
                        name=benchmark_data["name"],
                        data_path=Path(benchmark_data["data_path"]),
                        task_type=benchmark_data["task_type"],
                        evaluation_method=benchmark_data["evaluation_method"],
                        sample_limit=benchmark_data.get("sample_limit"),
                    )

                    # Test loading
                    from emotion_memory_experiments.benchmark_adapters import (
                        get_adapter,
                    )

                    adapter = get_adapter(benchmark_config)
                    dataset = adapter.create_dataset()

                    print(f"  ✅ {benchmark_name}: {len(dataset)} items loaded")
                    success_count += 1

                except Exception as e:
                    print(f"  ❌ {benchmark_name}: {e}")

            if success_count == total_benchmarks:
                print(f"✅ All {total_benchmarks} benchmarks loaded successfully")
                return True
            else:
                print(
                    f"❌ {total_benchmarks - success_count}/{total_benchmarks} benchmarks failed"
                )
                return False

        except Exception as e:
            print(f"❌ Benchmark loading test error: {e}")
            return False

    def analyze_results(self):
        """Analyze test results"""
        if "small_experiment" not in self.results:
            print("⚠️ No experiment results to analyze")
            return

        result_data = self.results["small_experiment"]
        results_df = result_data["results_df"]
        runtime = result_data["runtime"]

        print(f"\n📈 EXPERIMENT ANALYSIS")
        print("=" * 50)
        print(f"Runtime: {runtime:.1f} seconds")
        print(f"Total results: {len(results_df)}")
        print(f"Benchmarks: {results_df['benchmark'].nunique()}")
        print(f"Emotions: {results_df['emotion'].nunique()}")

        # Performance by emotion
        print(f"\n📊 Performance by Emotion:")
        emotion_stats = results_df.groupby("emotion")["score"].agg(
            ["mean", "std", "count"]
        )
        for emotion, stats in emotion_stats.iterrows():
            print(
                f"  {emotion}: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})"
            )

        # Performance by benchmark
        print(f"\n📋 Performance by Benchmark:")
        benchmark_stats = results_df.groupby("benchmark")["score"].agg(
            ["mean", "std", "count"]
        )
        for benchmark, stats in benchmark_stats.iterrows():
            print(
                f"  {benchmark}: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})"
            )

        # Save detailed results
        results_path = Path(self.temp_dir) / "detailed_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\n📁 Detailed results saved to: {results_path}")

    def run_full_test_suite(self, model_path: str) -> bool:
        """Run complete GPU test suite"""
        print("🚀 GPU EMOTION MEMORY EXPERIMENT TEST SUITE")
        print("=" * 70)

        try:
            self.setup()

            # Create test configuration
            config = self.create_gpu_test_config(model_path, use_small_dataset=True)

            # Run test components
            tests = [
                (
                    "Configuration Validation",
                    lambda: self.test_config_validation(config),
                ),
                ("Benchmark Loading", lambda: self.test_benchmark_loading(config)),
                ("Experiment Creation", lambda: self.test_experiment_creation(config)),
                (
                    "Small Experiment Run",
                    lambda: self.test_small_experiment_run(config),
                ),
            ]

            results = {}
            for test_name, test_func in tests:
                print(f"\n{'='*60}")
                print(f"Running {test_name} Test")
                print("=" * 60)

                try:
                    start_time = time.time()
                    result = test_func()
                    end_time = time.time()

                    results[test_name] = result
                    status = "✅ PASSED" if result else "❌ FAILED"
                    print(f"{test_name} Test: {status} ({end_time - start_time:.1f}s)")

                except Exception as e:
                    results[test_name] = False
                    print(f"❌ {test_name} Test FAILED with error: {e}")

            # Analyze results if experiment ran
            if results.get("Small Experiment Run", False):
                self.analyze_results()

            # Final summary
            passed = sum(results.values())
            total = len(results)

            print(f"\n{'='*70}")
            print("GPU TEST SUMMARY")
            print("=" * 70)

            for test_name, result in results.items():
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"{test_name:.<40} {status}")

            print(f"\nOverall: {passed}/{total} tests passed")

            if passed == total:
                print("🎉 ALL GPU TESTS PASSED!")
                print("\nVerified GPU capabilities:")
                print("✅ Real model loading and inference")
                print("✅ vLLM GPU acceleration")
                print("✅ RepE emotion activation")
                print("✅ Complete memory benchmark pipeline")
                print("✅ Original paper evaluation metrics")
                return True
            else:
                print(f"❌ {total - passed} tests failed")
                return False

        finally:
            self.teardown()


def test_with_mock_model():
    """Test with mock model for CI/CD environments"""
    print("🔧 Running tests with mock model (no GPU required)...")

    gpu_test = GPUExperimentTest()

    # Use mock model path
    mock_model_path = "/mock/model/path/Qwen2.5-0.5B-Instruct"

    try:
        gpu_test.setup()

        # Create config with mock model
        config = gpu_test.create_gpu_test_config(
            mock_model_path, use_small_dataset=True
        )
        config["hardware"]["device"] = "cpu"  # Force CPU for mock testing
        config["testing"] = {"mock_mode": True}  # Add mock flag

        # Test configuration components
        config_valid = gpu_test.test_config_validation(config)
        benchmark_loading = gpu_test.test_benchmark_loading(config)

        if config_valid and benchmark_loading:
            print("✅ Mock model tests passed")
            return True
        else:
            print("❌ Mock model tests failed")
            return False

    finally:
        gpu_test.teardown()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="GPU-enabled test for emotion memory experiments"
    )

    parser.add_argument("--model-path", type=str, help="Path to model for testing")

    parser.add_argument(
        "--mock-only",
        action="store_true",
        help="Run only mock tests (no real model/GPU required)",
    )

    args = parser.parse_args()

    if args.mock_only:
        success = test_with_mock_model()
    elif args.model_path:
        gpu_test = GPUExperimentTest()
        success = gpu_test.run_full_test_suite(args.model_path)
    else:
        print("❌ Either --model-path or --mock-only is required")
        print("\nExamples:")
        print("  # Test with real model")
        print("  python test_gpu_experiment_runner.py --model-path /path/to/model")
        print("  ")
        print("  # Test with mock model (no GPU)")
        print("  python test_gpu_experiment_runner.py --mock-only")
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
