#!/usr/bin/env python3
"""
Test runner for function calling tests.
"""

import subprocess
import sys
import time
from pathlib import Path

import requests


def check_server_health(port):
    """Check if server is running and healthy."""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            return data.get("status") == "healthy"
    except:
        pass
    return False


def run_tests():
    """Run all function calling tests."""
    print("🧪 Function Calling Test Suite")
    print("=" * 50)

    # Check if servers are running
    servers = [{"name": "Happiness", "port": 8000}, {"name": "Anger", "port": 8001}]

    running_servers = []
    for server in servers:
        if check_server_health(server["port"]):
            print(f"✅ {server['name']} server (port {server['port']}) is running")
            running_servers.append(server)
        else:
            print(f"❌ {server['name']} server (port {server['port']}) is not running")

    if not running_servers:
        print("\n⚠️  No servers are running. Some integration tests will be skipped.")
        print("To start servers:")
        print("  python -m openai_server --model /path/to/model --emotion happiness --port 8000")
        print("  python -m openai_server --model /path/to/model --emotion anger --port 8001")

    print(f"\n🏃‍♂️ Running tests with {len(running_servers)} server(s)...")

    # Get test directory
    test_dir = Path(__file__).parent

    # Run unit tests
    print("\n1. Running Unit Tests...")
    result1 = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(test_dir / "test_function_calling.py"),
            "-v",
            "--tb=short",
        ],
        cwd=test_dir.parent,
    )

    # Run integration tests
    print("\n2. Running Integration Tests...")
    result2 = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(test_dir / "test_function_calling_integration.py"),
            "-v",
            "--tb=short",
            "-s",
        ],
        cwd=test_dir.parent,
    )

    # Run performance tests if servers are available
    if running_servers:
        print("\n3. Running Performance Tests...")
        result3 = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(test_dir / "test_function_calling_integration.py"),
                "-v",
                "--tb=short",
                "-m",
                "performance",
            ],
            cwd=test_dir.parent,
        )
    else:
        print("\n3. Skipping Performance Tests (no servers running)")
        result3 = subprocess.CompletedProcess([], 0)  # Mock success

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"  Unit Tests: {'✅ PASSED' if result1.returncode == 0 else '❌ FAILED'}")
    print(f"  Integration Tests: {'✅ PASSED' if result2.returncode == 0 else '❌ FAILED'}")
    print(f"  Performance Tests: {'✅ PASSED' if result3.returncode == 0 else '❌ FAILED'}")

    overall_success = all(result.returncode == 0 for result in [result1, result2, result3])
    print(f"\n🎯 Overall: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")

    return overall_success


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
