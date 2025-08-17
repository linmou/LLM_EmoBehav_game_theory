"""
Test runner for all emotion memory experiment tests.
"""
import unittest
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import all test modules
from emotion_memory_experiments.tests import (
    test_data_models,
    test_benchmark_adapters,
    test_experiment,
    test_integration
)


def run_all_tests():
    """Run all tests and return results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test modules
    test_modules = [
        test_data_models,
        test_benchmark_adapters,
        test_experiment,
        test_integration
    ]
    
    for module in test_modules:
        suite.addTests(loader.loadTestsFromModule(module))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


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
    if len(sys.argv) > 1:
        # Run specific test module
        module_name = sys.argv[1]
        result = run_specific_test_module(module_name)
    else:
        # Run all tests
        print("Running all emotion memory experiment tests...")
        print("=" * 60)
        result = run_all_tests()
        
        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Skipped: {len(result.skipped)}")
        
        if result.failures:
            print("\nFAILURES:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        
        if result.errors:
            print("\nERRORS:")
            for test, traceback in result.errors:
                print(f"  - {test}")
        
        if result.wasSuccessful():
            print("\n✅ All tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)