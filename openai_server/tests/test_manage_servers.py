#!/usr/bin/env python3
"""
Test script for manage_servers.py functionality
"""

import os
import subprocess
import sys
from pathlib import Path

# Add current directory to path to import manage_servers
sys.path.insert(0, str(Path(__file__).parent))

from manage_servers import ServerManager, ServerProcess, parse_selection


def test_server_detection():
    """Test if the script can detect current running servers."""
    print("=" * 60)
    print("TEST 1: Server Detection")
    print("=" * 60)

    manager = ServerManager()
    servers = manager.find_servers()

    print(f"Found {len(servers)} OpenAI servers")

    if servers:
        print("\nDetected servers:")
        for i, server in enumerate(servers, 1):
            print(f"{i}. {server}")
        print("✓ Server detection working")
    else:
        print("⚠ No servers detected")

    return len(servers) > 0


def test_server_parsing():
    """Test server information parsing."""
    print("\n" + "=" * 60)
    print("TEST 2: Server Information Parsing")
    print("=" * 60)

    # Test with sample command lines
    test_cases = [
        {
            "cmd": "python -m openai_server.server --model /data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct --emotion anger --port 8000 --model_name qwen-anger",
            "expected": {
                "model": "Qwen2.5-0.5B-Instruct",
                "emotion": "anger",
                "port": "8000",
                "model_name": "qwen-anger",
            },
        },
        {
            "cmd": "python init_openai_server.py --model /path/to/model --emotion happiness --port 8001",
            "expected": {
                "model": "model",
                "emotion": "happiness",
                "port": "8001",
                "model_name": "Default",
            },
        },
    ]

    all_passed = True
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest case {i}:")
        server = ServerProcess(12345, test_case["cmd"])

        # Check each field
        for field, expected in test_case["expected"].items():
            actual = getattr(server, field)
            if actual == expected:
                print(f"  ✓ {field}: {actual}")
            else:
                print(f"  ✗ {field}: expected '{expected}', got '{actual}'")
                all_passed = False

    if all_passed:
        print("\n✓ All parsing tests passed")
    else:
        print("\n✗ Some parsing tests failed")

    return all_passed


def test_selection_parsing():
    """Test user selection parsing."""
    print("\n" + "=" * 60)
    print("TEST 3: Selection Parsing")
    print("=" * 60)

    test_cases = [
        {"input": "1", "max": 5, "expected": [1]},
        {"input": "1,3,5", "max": 5, "expected": [1, 3, 5]},
        {"input": "1-3", "max": 5, "expected": [1, 2, 3]},
        {"input": "1,3-5", "max": 5, "expected": [1, 3, 4, 5]},
        {"input": "1,1,2", "max": 5, "expected": [1, 2]},  # Deduplication
        {"input": "1,10", "max": 5, "expected": [1]},  # Invalid filtered out
    ]

    all_passed = True
    for test_case in test_cases:
        result = parse_selection(test_case["input"], test_case["max"])
        expected = sorted(test_case["expected"])
        actual = sorted(result)

        if actual == expected:
            print(f"  ✓ '{test_case['input']}' -> {actual}")
        else:
            print(f"  ✗ '{test_case['input']}' -> expected {expected}, got {actual}")
            all_passed = False

    if all_passed:
        print("\n✓ All selection parsing tests passed")
    else:
        print("\n✗ Some selection parsing tests failed")

    return all_passed


def test_process_commands():
    """Test that process detection commands work."""
    print("\n" + "=" * 60)
    print("TEST 4: Process Detection Commands")
    print("=" * 60)

    try:
        # Test ps command
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True, check=True)
        print("✓ 'ps aux' command works")

        # Test kill command (dry run)
        result = subprocess.run(["kill", "-0", str(os.getpid())], capture_output=True, check=True)
        print("✓ 'kill' command available")

        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed: {e}")
        return False


def test_script_execution():
    """Test that the main script can be executed."""
    print("\n" + "=" * 60)
    print("TEST 5: Script Execution")
    print("=" * 60)

    try:
        # Test script with quit command
        process = subprocess.Popen(
            ["python", "manage_servers.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Send quit command
        stdout, stderr = process.communicate(input="q\n", timeout=10)

        if process.returncode == 0:
            print("✓ Script executes successfully")
            if "RUNNING OPENAI SERVERS" in stdout:
                print("✓ Server list displayed")
            if "Options:" in stdout:
                print("✓ Interactive menu displayed")
            return True
        else:
            print(f"✗ Script failed with return code {process.returncode}")
            if stderr:
                print(f"Error: {stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("✗ Script execution timed out")
        process.kill()
        return False
    except Exception as e:
        print(f"✗ Script execution failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("SERVER MANAGEMENT SCRIPT VALIDATION")
    print("=" * 60)

    tests = [
        test_server_detection,
        test_server_parsing,
        test_selection_parsing,
        test_process_commands,
        test_script_execution,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! The script is working correctly.")
    else:
        print(f"⚠ {total - passed} test(s) failed. Check the output above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
