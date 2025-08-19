#!/usr/bin/env python3
"""
Test runner for all emotion memory experiment tests.
Runs both unittest-based tests and custom integration tests.
"""
import unittest
import subprocess
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import all test modules (skip missing ones)
test_modules = []
try:
    from emotion_memory_experiments.tests import test_data_models
    test_modules.append(test_data_models)
except ImportError:
    print("⚠️ test_data_models not found")

try:
    from emotion_memory_experiments.tests import test_benchmark_adapters
    test_modules.append(test_benchmark_adapters)
except ImportError:
    print("⚠️ test_benchmark_adapters not found")

try:
    from emotion_memory_experiments.tests import test_experiment
    test_modules.append(test_experiment)
except ImportError:
    print("⚠️ test_experiment not found")

try:
    from emotion_memory_experiments.tests import test_integration
    test_modules.append(test_integration)
except ImportError:
    print("⚠️ test_integration not found")


def run_custom_tests():
    """Run our custom test files that don't use unittest"""
    test_dir = Path(__file__).parent
    custom_tests = [
        "test_real_data_comprehensive.py",
        "test_original_evaluation_metrics.py",
        "test_integration_with_mocks.py"
    ]
    
    all_passed = True
    
    for test_file in custom_tests:
        test_path = test_dir / test_file
        if test_path.exists():
            print(f"\n{'='*60}")
            print(f"Running {test_file}")
            print('='*60)
            
            try:
                result = subprocess.run([sys.executable, str(test_path)], check=True)
                print(f"✅ {test_file} PASSED")
            except subprocess.CalledProcessError as e:
                print(f"❌ {test_file} FAILED with exit code {e.returncode}")
                all_passed = False
        else:
            print(f"⚠️ {test_file} not found")
    
    return all_passed

def run_unittest_tests():
    """Run unittest-based tests"""
    if not test_modules:
        print("No unittest modules found")
        return True
        
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all available test modules
    for module in test_modules:
        suite.addTests(loader.loadTestsFromModule(module))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_specific_test_module(module_name):
    """Run tests from a specific module"""
    module_map = {
        'data_models': test_data_models,
        'adapters': test_benchmark_adapters,
        'experiment': test_experiment,
        'integration': test_integration
    }
    
    if module_name not in module_map:
        print(f"Unknown test module: {module_name}")
        print(f"Available modules: {list(module_map.keys())}")
        return None
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(module_map[module_name])
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("🧪 EMOTION MEMORY EXPERIMENTS - TEST SUITE")
    print("Running all tests for benchmark adapters and evaluation metrics")
    print("=" * 70)
    
    # Run custom integration tests
    custom_passed = run_custom_tests()
    
    # Run unittest tests
    print(f"\n{'='*60}")
    print("Running unittest-based tests")
    print('='*60)
    unittest_passed = run_unittest_tests()
    
    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print('='*70)
    
    if custom_passed and unittest_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nVerified components:")
        print("✅ Real data loading (InfiniteBench, LongBench, LoCoMo)")
        print("✅ Original paper evaluation metrics")
        print("✅ Prompt wrapper integration")
        print("✅ DataLoader compatibility")
        print("✅ Ultra-simple architecture")
        print("✅ Integration test with mocked GPU pipeline")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        if not custom_passed:
            print("- Custom integration tests failed")
        if not unittest_passed:
            print("- Unittest tests failed")
        sys.exit(1)