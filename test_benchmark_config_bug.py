#!/usr/bin/env python3
"""
# Test file responsible for: memory_experiment_series_runner.py expand_benchmark_configs method
# Purpose: Reproduce the TypeError bug in BenchmarkConfig creation

Standalone test to reproduce and fix the BenchmarkConfig initialization bug.
This test isolates the specific issue without requiring heavy dependencies.
"""

import sys
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


# Directly include the BenchmarkConfig class to avoid import issues
@dataclass
class BenchmarkConfig:
    name: str
    task_type: str  # e.g., 'passkey', 'kv_retrieval', 'longbook_qa_eng'
    data_path: Path  # Auto-generated if None
    sample_limit: Optional[int]
    augmentation_config: Optional[
        Dict[str, str]
    ]  # Custom prefix/suffix for context and answer marking

    # Context truncation settings (dataset-specific)
    enable_auto_truncation: bool  # Enable automatic context truncation
    truncation_strategy: str  # "right" or "left" (via tokenizer)
    preserve_ratio: float  # Ratio of max_model_len to use for context

    def discover_datasets_by_pattern(
        self, base_data_dir: str = "data/memory_benchmarks"
    ) -> List[str]:
        """Mock discovery method for testing"""
        # Mock discovered tasks for testing
        return ["narrativeqa", "qasper", "hotpotqa"]


def create_benchmark_config(
    name: str,
    task_type: str,
    data_path: Path,
    sample_limit: Optional[int] = None,
    augmentation_config: Optional[Dict[str, str]] = None,
    enable_auto_truncation: bool = False,
    truncation_strategy: str = "right",
    preserve_ratio: float = 0.8,
) -> BenchmarkConfig:
    """Factory function to create BenchmarkConfig with safe defaults"""
    return BenchmarkConfig(
        name=name,
        task_type=task_type,
        data_path=data_path,
        sample_limit=sample_limit,
        augmentation_config=augmentation_config,
        enable_auto_truncation=enable_auto_truncation,
        truncation_strategy=truncation_strategy,
        preserve_ratio=preserve_ratio,
    )


class MockMemoryExperimentSeriesRunner:
    """Mock runner class containing the buggy method"""
    
    def __init__(self, config):
        self.base_config = config
    
    def _is_pattern_task_type(self, task_type: str) -> bool:
        """Check if task_type contains regex pattern characters"""
        if not task_type:
            return False
        
        pattern_chars = ['*', '?', '[', ']', '{', '}', '(', ')', '^', '$', '+', '.', '|', '\\']
        return any(char in task_type for char in pattern_chars)
    
    def expand_benchmark_configs(self, benchmarks):
        """BUGGY METHOD - reproduces the original bug"""
        expanded_benchmarks = []

        for benchmark_config in benchmarks:
            task_type = benchmark_config.get("task_type", "")

            # Check if task_type is a pattern (contains wildcards or regex characters)
            if self._is_pattern_task_type(task_type):
                # THIS IS THE BUG: Create a temporary BenchmarkConfig with missing required args
                temp_benchmark = BenchmarkConfig(
                    name=benchmark_config["name"], 
                    task_type=task_type
                    # MISSING: data_path, sample_limit, augmentation_config, 
                    # enable_auto_truncation, truncation_strategy, preserve_ratio
                )

                # Discover task types matching the pattern
                base_data_dir = self.base_config.get(
                    "base_data_dir", "data/memory_benchmarks"
                )
                discovered_tasks = temp_benchmark.discover_datasets_by_pattern(
                    base_data_dir
                )

                # Create individual configs for each discovered task
                for discovered_task_type in discovered_tasks:
                    expanded_config = benchmark_config.copy()
                    expanded_config["task_type"] = discovered_task_type
                    expanded_benchmarks.append(expanded_config)
            else:
                # Keep literal task types as-is
                expanded_benchmarks.append(benchmark_config)

        return expanded_benchmarks


def test_benchmark_config_missing_args_reproduces_bug():
    """Test that reproduces the exact bug reported"""
    print("🔍 Testing BenchmarkConfig creation with missing required arguments...")
    
    try:
        # This should fail with the exact same error as reported
        BenchmarkConfig(
            name="test_benchmark",
            task_type="test_task"
            # Missing: data_path, sample_limit, augmentation_config, 
            # enable_auto_truncation, truncation_strategy, preserve_ratio
        )
        print("❌ Expected TypeError was not raised!")
        return False
    except TypeError as e:
        error_message = str(e)
        print(f"✅ Successfully reproduced the bug: {error_message}")
        
        # Verify it contains the expected missing arguments
        expected_missing_args = [
            'data_path', 'sample_limit', 'augmentation_config',
            'enable_auto_truncation', 'truncation_strategy', 'preserve_ratio'
        ]
        
        missing_count = 0
        for arg in expected_missing_args:
            if arg in error_message:
                missing_count += 1
        
        if missing_count == len(expected_missing_args):
            print(f"✅ All {len(expected_missing_args)} expected missing arguments found in error message")
            return True
        else:
            print(f"⚠️  Only {missing_count}/{len(expected_missing_args)} expected missing arguments found")
            return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_expand_benchmark_configs_with_pattern_reproduces_bug():
    """Test that reproduces the bug in the expand_benchmark_configs method"""
    print("\n🔍 Testing expand_benchmark_configs with pattern task type...")
    
    # Create runner with minimal config
    minimal_config = {
        "models": ["/test/model"],
        "emotions": ["anger"],
        "intensities": [1.0],
        "benchmarks": []
    }
    runner = MockMemoryExperimentSeriesRunner(minimal_config)
    
    # Create benchmark config with pattern task type
    benchmarks = [{
        "name": "test_bench", 
        "task_type": ".*test.*",  # This is a regex pattern
        "sample_limit": 100
    }]
    
    try:
        # This should fail with the same TypeError
        runner.expand_benchmark_configs(benchmarks)
        print("❌ Expected TypeError was not raised!")
        return False
    except TypeError as e:
        error_message = str(e)
        print(f"✅ Successfully reproduced the bug in expand_benchmark_configs: {error_message}")
        
        # Check for the specific error signature
        if "BenchmarkConfig.__init__()" in error_message and "missing 6 required positional arguments" in error_message:
            print("✅ Error signature matches exactly with the reported bug")
            return True
        else:
            print(f"⚠️  Error signature doesn't match expected pattern")
            return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_literal_task_type_works():
    """Test that literal task types work without issues"""
    print("\n🔍 Testing expand_benchmark_configs with literal task type...")
    
    minimal_config = {
        "models": ["/test/model"], 
        "emotions": ["anger"],
        "intensities": [1.0],
        "benchmarks": []
    }
    runner = MockMemoryExperimentSeriesRunner(minimal_config)
    
    benchmarks = [{
        "name": "test_bench",
        "task_type": "literal_task",  # Not a pattern
        "sample_limit": 100
    }]
    
    try:
        result = runner.expand_benchmark_configs(benchmarks)
        print(f"✅ Literal task type works fine, got {len(result)} benchmarks")
        return True
    except Exception as e:
        print(f"❌ Unexpected error with literal task type: {e}")
        return False


def main():
    """Run all bug reproduction tests"""
    print("🚀 Starting BenchmarkConfig bug reproduction tests...\n")
    
    # Test 1: Direct BenchmarkConfig creation with missing args
    test1_passed = test_benchmark_config_missing_args_reproduces_bug()
    
    # Test 2: expand_benchmark_configs method with pattern task type  
    test2_passed = test_expand_benchmark_configs_with_pattern_reproduces_bug()
    
    # Test 3: Verify literal task types still work
    test3_passed = test_literal_task_type_works()
    
    print("\n📊 Test Results:")
    print(f"   • Direct BenchmarkConfig bug reproduction: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"   • expand_benchmark_configs bug reproduction: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print(f"   • Literal task types still work: {'✅ PASS' if test3_passed else '❌ FAIL'}")
    
    if all([test1_passed, test2_passed, test3_passed]):
        print("\n🎉 All bug reproduction tests passed! The bug has been successfully reproduced.")
        print("\n📋 Bug Analysis:")
        print("   • The bug occurs when expand_benchmark_configs encounters a pattern task_type")
        print("   • It tries to create BenchmarkConfig with only name and task_type")
        print("   • Missing 6 required arguments: data_path, sample_limit, augmentation_config,")
        print("     enable_auto_truncation, truncation_strategy, preserve_ratio")
        print("\n💡 Next step: Fix the bug by providing all required arguments or using factory function")
        return True
    else:
        print("\n💥 Some bug reproduction tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)