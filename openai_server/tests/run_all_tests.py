#!/usr/bin/env python3
"""
Test Runner for OpenAI Server

This script runs all tests for the OpenAI server module.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def run_unit_tests(verbose=False):
    """Run unit tests"""
    print("\n" + "=" * 60)
    print("Running Unit Tests")
    print("=" * 60)

    cmd = [sys.executable, "-m", "pytest", "test_unit_server.py"]
    if verbose:
        cmd.append("-v")

    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0


def run_integration_tests(verbose=False, skip_gpu_check=False):
    """Run integration tests"""
    print("\n" + "=" * 60)
    print("Running Integration Tests")
    print("=" * 60)

    if not skip_gpu_check and not torch.cuda.is_available():
        print("WARNING: No GPU available - skipping integration tests")
        print("Use --skip-gpu-check to force run")
        return True

    cmd = [sys.executable, "-m", "pytest", "test_integration_server.py"]
    if verbose:
        cmd.extend(["-v", "-s"])

    # Set test environment variables
    env = os.environ.copy()
    env["VLLM_GPU_MEMORY_UTILIZATION"] = "0.5"

    result = subprocess.run(cmd, cwd=Path(__file__).parent, env=env)
    return result.returncode == 0


def run_existing_tests(verbose=False):
    """Run existing test files"""
    print("\n" + "=" * 60)
    print("Running Existing Tests")
    print("=" * 60)

    test_files = [
        "test_openai_server.py",
        "test_integrated_openai_server.py",
        "test_server_connectivity.py",
    ]

    all_passed = True

    for test_file in test_files:
        if Path(test_file).exists():
            print(f"\nRunning {test_file}...")
            try:
                result = subprocess.run(
                    [sys.executable, test_file], cwd=Path(__file__).parent, timeout=60
                )
                if result.returncode != 0:
                    all_passed = False
                    print(f"❌ {test_file} failed")
                else:
                    print(f"✅ {test_file} passed")
            except subprocess.TimeoutExpired:
                print(f"⏰ {test_file} timed out")
                all_passed = False
            except Exception as e:
                print(f"❌ {test_file} error: {e}")
                all_passed = False

    return all_passed


def run_coverage_report():
    """Generate coverage report"""
    print("\n" + "=" * 60)
    print("Generating Coverage Report")
    print("=" * 60)

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--cov=openai_server",
        "--cov-report=term-missing",
        "--cov-report=html",
        "test_unit_server.py",
    ]

    subprocess.run(cmd, cwd=Path(__file__).parent)


def main():
    parser = argparse.ArgumentParser(description="Run OpenAI Server tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--unit-only", action="store_true", help="Run only unit tests")
    parser.add_argument(
        "--integration-only", action="store_true", help="Run only integration tests"
    )
    parser.add_argument("--skip-gpu-check", action="store_true", help="Skip GPU availability check")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--existing", action="store_true", help="Run existing test files")

    args = parser.parse_args()

    all_passed = True

    # Run tests based on arguments
    if args.unit_only:
        all_passed = run_unit_tests(args.verbose)
    elif args.integration_only:
        all_passed = run_integration_tests(args.verbose, args.skip_gpu_check)
    elif args.existing:
        all_passed = run_existing_tests(args.verbose)
    else:
        # Run all tests
        unit_passed = run_unit_tests(args.verbose)
        integration_passed = run_integration_tests(args.verbose, args.skip_gpu_check)
        existing_passed = run_existing_tests(args.verbose)
        all_passed = unit_passed and integration_passed and existing_passed

    # Generate coverage if requested
    if args.coverage:
        run_coverage_report()

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
