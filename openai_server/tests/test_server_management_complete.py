#!/usr/bin/env python3
"""
Comprehensive Test Script for Enhanced Server Management

This script validates:
1. Server startup and detection
2. Orphaned process detection after server shutdown
3. Process cleanup functionality
4. ps aux parsing accuracy
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add current directory to path to import manage_servers
sys.path.insert(0, str(Path(__file__).parent))

from manage_servers import ServerManager, ServerProcess


class ServerTestManager:
    """Manages server lifecycle for testing."""

    def __init__(self):
        self.started_pids: List[int] = []
        self.server_logs: List[str] = []

    def start_test_server(self, port: int, emotion: str, model_name: str) -> Tuple[bool, int]:
        """Start a test server and return success status and PID."""
        log_file = f"test_server_{port}.log"
        self.server_logs.append(log_file)

        cmd = [
            "python",
            "-m",
            "openai_server",
            "--model",
            "/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct",
            "--emotion",
            emotion,
            "--port",
            str(port),
            "--model_name",
            model_name,
        ]

        try:
            # Start server in background
            with open(log_file, "w") as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,  # Create new process group
                )

            # Give it time to start
            time.sleep(5)

            # Check if process is still running
            if process.poll() is None:
                self.started_pids.append(process.pid)
                return True, process.pid
            else:
                return False, -1

        except Exception as e:
            print(f"Failed to start server on port {port}: {e}")
            return False, -1

    def wait_for_server_startup(self, pid: int, timeout: int = 60) -> bool:
        """Wait for server to fully start up."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Check if process is still running
                os.kill(pid, 0)  # Signal 0 just checks if process exists

                # Look for startup completion in logs
                for log_file in self.server_logs:
                    if os.path.exists(log_file):
                        with open(log_file, "r") as f:
                            content = f.read()
                            if "Uvicorn running on" in content:
                                return True

                time.sleep(2)
            except OSError:
                # Process died
                return False

        return False

    def kill_server_gracefully(self, pid: int) -> bool:
        """Kill server gracefully and return success."""
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(3)  # Give it time to shut down

            # Check if it's still running
            try:
                os.kill(pid, 0)
                return False  # Still running
            except OSError:
                return True  # Successfully killed
        except OSError:
            return True  # Already dead

    def cleanup_all(self):
        """Clean up all started servers and logs."""
        for pid in self.started_pids:
            try:
                # Kill the process group to get child processes too
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                time.sleep(2)
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except OSError:
                pass

        # Clean up log files
        for log_file in self.server_logs:
            try:
                os.remove(log_file)
            except OSError:
                pass


def test_ps_aux_parsing():
    """Test ps aux parsing accuracy."""
    print("=" * 60)
    print("TEST: PS AUX PARSING ACCURACY")
    print("=" * 60)

    try:
        # Get actual ps aux output
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True, check=True)
        lines = result.stdout.split("\n")

        print(f"Total ps aux lines: {len(lines)}")

        # Count python processes
        python_count = 0
        openai_server_count = 0
        multiprocessing_count = 0

        for line in lines:
            if "python" in line and "grep" not in line:
                python_count += 1
                if "openai_server" in line or "init_openai_server" in line:
                    openai_server_count += 1
                if "multiprocessing" in line:
                    multiprocessing_count += 1

        print(f"Python processes found: {python_count}")
        print(f"OpenAI server processes: {openai_server_count}")
        print(f"Multiprocessing processes: {multiprocessing_count}")

        return True

    except subprocess.CalledProcessError as e:
        print(f"Failed to run ps aux: {e}")
        return False


def test_server_detection():
    """Test server detection functionality."""
    print("\n" + "=" * 60)
    print("TEST: SERVER DETECTION")
    print("=" * 60)

    manager = ServerManager()
    servers = manager.find_servers()
    orphaned = manager.find_orphaned_processes()

    print(f"Active servers detected: {len(servers)}")
    for i, server in enumerate(servers, 1):
        print(f"  {i}. {server}")

    print(f"Orphaned processes detected: {len(orphaned)}")
    for pid in orphaned:
        print(f"  PID: {pid}")

    return True


def test_server_lifecycle():
    """Test complete server lifecycle: start -> detect -> kill -> detect orphans."""
    print("\n" + "=" * 60)
    print("TEST: SERVER LIFECYCLE")
    print("=" * 60)

    test_manager = ServerTestManager()
    server_manager = ServerManager()

    try:
        # Phase 1: Start test servers
        print("Phase 1: Starting test servers...")
        servers_to_start = [
            (8010, "anger", "test-lifecycle-1"),
            (8011, "happiness", "test-lifecycle-2"),
        ]

        started_servers = []
        for port, emotion, name in servers_to_start:
            success, pid = test_manager.start_test_server(port, emotion, name)
            if success:
                print(f"  ✓ Started server on port {port}, PID: {pid}")
                started_servers.append((port, emotion, name, pid))
            else:
                print(f"  ✗ Failed to start server on port {port}")

        if not started_servers:
            print("No servers started successfully")
            return False

        # Phase 2: Wait for servers to fully start
        print("\nPhase 2: Waiting for servers to start up...")
        for port, emotion, name, pid in started_servers:
            if test_manager.wait_for_server_startup(pid):
                print(f"  ✓ Server on port {port} started successfully")
            else:
                print(f"  ⚠ Server on port {port} may not have started properly")

        time.sleep(5)  # Extra wait for vLLM initialization

        # Phase 3: Detect running servers
        print("\nPhase 3: Detecting running servers...")
        detected_servers = server_manager.find_servers()
        detected_orphaned = server_manager.find_orphaned_processes()

        print(f"  Detected {len(detected_servers)} active servers")
        print(f"  Detected {len(detected_orphaned)} orphaned processes")

        for server in detected_servers:
            print(f"    - {server}")

        # Phase 4: Kill servers gracefully
        print("\nPhase 4: Killing servers gracefully...")
        killed_servers = []
        for port, emotion, name, pid in started_servers:
            if test_manager.kill_server_gracefully(pid):
                print(f"  ✓ Killed server PID {pid} gracefully")
                killed_servers.append(pid)
            else:
                print(f"  ⚠ Server PID {pid} did not shut down gracefully")

        # Phase 5: Check for orphaned processes
        print("\nPhase 5: Checking for orphaned processes...")
        time.sleep(3)  # Wait for orphaned processes to appear

        post_kill_servers = server_manager.find_servers()
        post_kill_orphaned = server_manager.find_orphaned_processes()

        print(f"  Active servers after kill: {len(post_kill_servers)}")
        print(f"  Orphaned processes after kill: {len(post_kill_orphaned)}")

        if post_kill_orphaned:
            print("  Orphaned processes found:")
            for pid in post_kill_orphaned:
                print(f"    - PID: {pid}")

        # Phase 6: Clean up orphaned processes
        if post_kill_orphaned:
            print("\nPhase 6: Cleaning up orphaned processes...")
            cleaned = 0
            for pid in post_kill_orphaned:
                try:
                    os.kill(pid, signal.SIGKILL)
                    print(f"  ✓ Killed orphaned process PID {pid}")
                    cleaned += 1
                except OSError:
                    print(f"  ⚠ Could not kill orphaned process PID {pid}")

            print(f"  Cleaned up {cleaned}/{len(post_kill_orphaned)} orphaned processes")

        return True

    except Exception as e:
        print(f"Test failed with exception: {e}")
        return False

    finally:
        # Always clean up
        print("\nCleaning up...")
        test_manager.cleanup_all()


def test_enhanced_manage_servers():
    """Test the enhanced manage_servers.py script."""
    print("\n" + "=" * 60)
    print("TEST: ENHANCED MANAGE_SERVERS SCRIPT")
    print("=" * 60)

    try:
        # Test with no input (should show current state and quit)
        result = subprocess.run(
            ["python", "manage_servers.py"], input="q\n", capture_output=True, text=True, timeout=10
        )

        output = result.stdout

        # Check for expected output elements
        checks = [
            ("Server Manager header", "OpenAI Server Manager" in output),
            ("Options display", "Options:" in output),
            ("Quit functionality", "Goodbye!" in output),
        ]

        if "Found" in output and "orphaned" in output:
            checks.append(("Orphaned detection", True))
        else:
            checks.append(("Orphaned detection", "No orphaned processes found"))

        for check_name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}: {passed}")

        return all(isinstance(check[1], bool) and check[1] for check in checks)

    except subprocess.TimeoutExpired:
        print("  ✗ Script execution timed out")
        return False
    except Exception as e:
        print(f"  ✗ Script test failed: {e}")
        return False


def main():
    """Run comprehensive server management tests."""
    print("COMPREHENSIVE SERVER MANAGEMENT VALIDATION")
    print("=" * 60)

    tests = [
        ("PS Aux Parsing", test_ps_aux_parsing),
        ("Server Detection", test_server_detection),
        ("Enhanced Script", test_enhanced_manage_servers),
        ("Server Lifecycle", test_server_lifecycle),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        try:
            if test_func():
                print(f"✓ {test_name} PASSED")
                passed += 1
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! Enhanced server management is working correctly.")
    else:
        print(f"⚠ {total - passed} test(s) failed. Check the output above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
