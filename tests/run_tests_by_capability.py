#!/usr/bin/env python3
"""
Smart Test Runner - Runs tests based on available hardware capabilities.

This script automatically detects available hardware and runs appropriate tests:
- Mock-only tests: Always available (no hardware requirements)
- CPU tests: Require sufficient RAM and/or model files
- GPU tests: Require GPU with sufficient VRAM

Usage:
    python run_tests_by_capability.py                    # Auto-detect and run appropriate tests
    python run_tests_by_capability.py --mode cpu_only    # Force CPU-only tests
    python run_tests_by_capability.py --mode mock_only   # Force mock-only tests
    python run_tests_by_capability.py --mode gpu_only    # Force GPU tests only
    python run_tests_by_capability.py --info            # Just show environment info
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parents[1]
sys.path.insert(0, str(project_root))

from tests.unit.neuro_manipulation.repe.test_environment import TestEnvironment


class TestRunner:
    """Smart test runner that adapts to available hardware."""
    
    def __init__(self):
        self.test_suites = {
            "mock_only": [
                {
                    "name": "Mock-based Unit Tests", 
                    "command": ["python", "tests/unit/neuro_manipulation/repe/test_multimodal_rep_reading.py"],
                    "description": "Pure mock tests, no hardware requirements"
                },
                {
                    "name": "CPU-Friendly Tests",
                    "command": ["python", "tests/cpu_friendly/test_multimodal_cpu_friendly.py"], 
                    "description": "Logic tests without model loading"
                }
            ],
            "cpu_capable": [
                {
                    "name": "Prompt Format Integration",
                    "command": ["python", "tests/unit/test_prompt_format_integration.py"],
                    "description": "Tokenizer-based format validation"
                }
            ],
            "gpu_required": [
                {
                    "name": "Real Model Integration",
                    "command": [
                        "python", "-m", "pytest", 
                        "tests/integration/test_real_multimodal_integration.py::TestRealMultimodalIntegration::test_emotion_vector_extraction_basic",
                        "-v", "-s"
                    ],
                    "description": "Full model inference with emotion vector extraction",
                    "gpu_env": "CUDA_VISIBLE_DEVICES=3"
                },
                {
                    "name": "Model Forward Pass Test", 
                    "command": [
                        "python", "-m", "pytest",
                        "tests/integration/test_real_multimodal_integration.py::TestRealMultimodalIntegration::test_model_forward_pass",
                        "-v", "-s"
                    ],
                    "description": "Forward pass validation with real model",
                    "gpu_env": "CUDA_VISIBLE_DEVICES=3"
                }
            ]
        }
    
    def run_test_suite(self, suite_name: str, test_info: dict) -> bool:
        """Run a single test suite."""
        print(f"\n🧪 {test_info['name']}")
        print(f"   {test_info['description']}")
        print("-" * 60)
        
        # Prepare environment
        env = os.environ.copy()
        if "gpu_env" in test_info:
            gpu_env = test_info["gpu_env"]
            key, value = gpu_env.split("=", 1)
            env[key] = value
        
        # Run command
        try:
            result = subprocess.run(
                test_info["command"],
                env=env,
                capture_output=False,  # Show output in real-time
                text=True
            )
            
            success = result.returncode == 0
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"\n{status}: {test_info['name']}")
            return success
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            return False
    
    def run_tests_by_mode(self, mode: str) -> dict:
        """Run tests appropriate for the given mode."""
        results = {"passed": 0, "failed": 0, "skipped": 0, "total": 0}
        
        print(f"🚀 Running tests in mode: {mode.upper()}")
        
        # Determine which test suites to run
        suites_to_run = []
        
        if mode in ["mock_only"]:
            suites_to_run = ["mock_only"]
        elif mode in ["cpu_only", "cpu_capable"]:
            suites_to_run = ["mock_only", "cpu_capable"]
        elif mode in ["gpu_only", "gpu_required"]:
            suites_to_run = ["gpu_required"]
        elif mode in ["full_gpu", "all"]:
            suites_to_run = ["mock_only", "cpu_capable", "gpu_required"]
        else:
            print(f"❌ Unknown mode: {mode}")
            return results
        
        # Run selected test suites
        for suite_name in suites_to_run:
            if suite_name in self.test_suites:
                print(f"\n📋 Running {suite_name.upper()} tests:")
                
                for test_info in self.test_suites[suite_name]:
                    results["total"] += 1
                    
                    if self.run_test_suite(suite_name, test_info):
                        results["passed"] += 1
                    else:
                        results["failed"] += 1
        
        return results
    
    def run_adaptive_tests(self) -> dict:
        """Run tests based on automatically detected capabilities."""
        test_mode = TestEnvironment.get_test_mode()
        print(f"🔍 Auto-detected test mode: {test_mode}")
        
        return self.run_tests_by_mode(test_mode)
    
    def print_summary(self, results: dict):
        """Print test results summary."""
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        total = results["total"]
        passed = results["passed"] 
        failed = results["failed"]
        
        if total == 0:
            print("❌ No tests were run")
            return
        
        print(f"Total Tests:  {total}")
        print(f"Passed:       {passed} ✅")
        print(f"Failed:       {failed} ❌")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if failed == 0:
            print("\n🎉 All tests passed!")
        else:
            print(f"\n⚠️  {failed} test(s) failed - check output above")


def main():
    parser = argparse.ArgumentParser(
        description="Smart test runner for multimodal RepE tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests_by_capability.py                 # Auto-detect capabilities
  python run_tests_by_capability.py --mode cpu_only # CPU-only tests
  python run_tests_by_capability.py --mode gpu_only # GPU tests only  
  python run_tests_by_capability.py --info          # Show environment info
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["auto", "mock_only", "cpu_only", "gpu_only", "all"],
        default="auto",
        help="Test mode to run (default: auto-detect)"
    )
    
    parser.add_argument(
        "--info",
        action="store_true", 
        help="Show environment information and exit"
    )
    
    args = parser.parse_args()
    
    # Show environment info
    if args.info:
        TestEnvironment.print_environment_info()
        return 0
    
    print("🔍 Multimodal RepE Test Runner")
    print("=" * 60)
    
    # Show environment info first
    TestEnvironment.print_environment_info()
    
    # Initialize test runner
    runner = TestRunner()
    
    # Run tests
    if args.mode == "auto":
        results = runner.run_adaptive_tests()
    else:
        results = runner.run_tests_by_mode(args.mode)
    
    # Print summary
    runner.print_summary(results)
    
    # Return appropriate exit code
    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())