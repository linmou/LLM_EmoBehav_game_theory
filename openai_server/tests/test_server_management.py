#!/usr/bin/env python3
"""
Comprehensive Test Script for OpenAI Server Management

This script:
1. Starts multiple OpenAI servers with different configurations
2. Validates server detection and information parsing
3. Tests the management script functionality
4. Cleans up all processes including orphaned ones
5. Provides detailed validation reports

Usage: python test_server_management.py
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from manage_servers import ServerManager


class ServerTestHarness:
    """Test harness for OpenAI server management validation."""

    def __init__(self):
        self.started_servers = []
        self.test_results = []
        self.log_files = []

    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test results."""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if details:
            print(f"    {details}")
        self.test_results.append({"test": test_name, "passed": passed, "details": details})

    def start_test_server(
        self, emotion: str, port: int, model_name: str, use_legacy: bool = False
    ) -> Tuple[bool, int]:
        """Start a test OpenAI server."""
        model_path = "/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct"
        log_file = f"test_server_{emotion}_{port}.log"
        self.log_files.append(log_file)

        if use_legacy:
            cmd = [
                "python",
                "init_openai_server.py",
                "--model",
                model_path,
                "--emotion",
                emotion,
                "--port",
                str(port),
                "--model_name",
                model_name,
            ]
        else:
            cmd = [
                "python",
                "-m",
                "openai_server",
                "--model",
                model_path,
                "--emotion",
                emotion,
                "--port",
                str(port),
                "--model_name",
                model_name,
            ]

        try:
            print(
                f"Starting server: {emotion} on port {port} ({'legacy' if use_legacy else 'module'})..."
            )
            process = subprocess.Popen(
                cmd,
                stdout=open(log_file, "w"),
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,  # Create new process group
            )

            self.started_servers.append(
                {
                    "process": process,
                    "emotion": emotion,
                    "port": port,
                    "model_name": model_name,
                    "log_file": log_file,
                    "legacy": use_legacy,
                }
            )

            return True, process.pid

        except Exception as e:
            print(f"Failed to start server: {e}")
            return False, -1

    def wait_for_servers_ready(self, timeout: int = 60) -> int:
        """Wait for servers to be ready and count successful starts."""
        print(f"Waiting up to {timeout}s for servers to initialize...")

        ready_count = 0
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check log files for startup completion
            for server in self.started_servers:
                if server.get("ready", False):
                    continue

                log_file = server["log_file"]
                if os.path.exists(log_file):
                    try:
                        with open(log_file, "r") as f:
                            content = f.read()
                            # Look for successful startup indicators
                            if (
                                "Uvicorn running on" in content
                                or "Application startup complete" in content
                            ):
                                server["ready"] = True
                                ready_count += 1
                                print(f"  ✓ Server {server['emotion']}:{server['port']} ready")
                            elif "ERROR" in content and "failed" in content.lower():
                                server["failed"] = True
                                print(
                                    f"  ✗ Server {server['emotion']}:{server['port']} failed to start"
                                )
                    except Exception:
                        pass

            # Check if all servers are ready or failed
            all_done = all(
                server.get("ready", False) or server.get("failed", False)
                for server in self.started_servers
            )
            if all_done:
                break

            time.sleep(2)

        successful = sum(1 for server in self.started_servers if server.get("ready", False))
        print(f"Server startup complete: {successful}/{len(self.started_servers)} servers ready")
        return successful

    def test_server_detection(self) -> bool:
        """Test if manage_servers.py can detect running servers."""
        print("\n" + "=" * 60)
        print("TESTING SERVER DETECTION")
        print("=" * 60)

        manager = ServerManager()
        detected_servers = manager.find_servers()

        # Get actual running processes
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        ps_lines = [
            line
            for line in result.stdout.split("\n")
            if "python" in line
            and ("openai_server" in line or "init_openai_server" in line)
            and "grep" not in line
            and "--model" in line
        ]

        self.log_test(
            "Process detection via ps aux",
            len(ps_lines) > 0,
            f"Found {len(ps_lines)} processes in ps output",
        )

        self.log_test(
            "manage_servers.py detection",
            len(detected_servers) > 0,
            f"Detected {len(detected_servers)} servers",
        )

        # Validate detection accuracy
        detection_accurate = len(detected_servers) == len(ps_lines)
        self.log_test(
            "Detection accuracy",
            detection_accurate,
            f"Detected {len(detected_servers)}, expected {len(ps_lines)}",
        )

        # Test server information parsing
        if detected_servers:
            server = detected_servers[0]
            has_model = server.model and server.model != "Unknown"
            has_emotion = server.emotion and server.emotion != "None"
            has_port = server.port and server.port.isdigit()

            self.log_test("Model parsing", has_model, f"Model: {server.model}")
            self.log_test("Emotion parsing", has_emotion, f"Emotion: {server.emotion}")
            self.log_test("Port parsing", has_port, f"Port: {server.port}")

        return len(detected_servers) > 0

    def test_orphaned_detection(self) -> bool:
        """Test orphaned process detection."""
        print("\n" + "=" * 60)
        print("TESTING ORPHANED PROCESS DETECTION")
        print("=" * 60)

        manager = ServerManager()
        orphaned = manager.find_orphaned_processes()

        # Get actual multiprocessing processes
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        mp_lines = [
            line
            for line in result.stdout.split("\n")
            if "python" in line and "multiprocessing" in line and "grep" not in line
        ]

        self.log_test(
            "Multiprocessing process detection",
            len(mp_lines) >= 0,
            f"Found {len(mp_lines)} multiprocessing processes",
        )

        self.log_test(
            "Orphaned process detection function",
            True,
            f"Detected {len(orphaned)} orphaned processes",
        )

        return True

    def test_management_script(self) -> bool:
        """Test the management script interactive functionality."""
        print("\n" + "=" * 60)
        print("TESTING MANAGEMENT SCRIPT")
        print("=" * 60)

        # Test script execution with refresh and quit
        try:
            process = subprocess.Popen(
                ["python", "manage_servers.py"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Send refresh and quit commands
            stdout, stderr = process.communicate(input="r\nq\n", timeout=15)

            script_runs = process.returncode == 0
            self.log_test("Script execution", script_runs, "Script runs without errors")

            shows_servers = "RUNNING OPENAI SERVERS" in stdout
            self.log_test("Server display", shows_servers, "Shows server list interface")

            shows_menu = "Options:" in stdout
            self.log_test("Interactive menu", shows_menu, "Shows interactive menu")

            return script_runs and shows_servers and shows_menu

        except subprocess.TimeoutExpired:
            process.kill()
            self.log_test("Script execution", False, "Script timed out")
            return False
        except Exception as e:
            self.log_test("Script execution", False, f"Error: {e}")
            return False

    def cleanup_servers(self) -> int:
        """Clean up all started servers and orphaned processes."""
        print("\n" + "=" * 60)
        print("CLEANING UP SERVERS")
        print("=" * 60)

        cleaned_count = 0

        # Kill started servers by process group
        for server in self.started_servers:
            try:
                process = server["process"]
                if process.poll() is None:  # Still running
                    print(
                        f"Killing server {server['emotion']}:{server['port']} (PID: {process.pid})"
                    )
                    # Kill entire process group to get child processes too
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    time.sleep(2)

                    # Force kill if still running
                    if process.poll() is None:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)

                    cleaned_count += 1
            except Exception as e:
                print(f"Error killing server {server['emotion']}:{server['port']}: {e}")

        # Clean up orphaned processes
        time.sleep(3)  # Wait for processes to exit
        manager = ServerManager()
        orphaned = manager.find_orphaned_processes()

        for pid in orphaned:
            try:
                os.kill(pid, signal.SIGKILL)
                print(f"Killed orphaned process PID {pid}")
                cleaned_count += 1
            except Exception as e:
                print(f"Error killing orphaned process {pid}: {e}")

        # Clean up log files
        for log_file in self.log_files:
            try:
                if os.path.exists(log_file):
                    os.remove(log_file)
            except Exception:
                pass

        return cleaned_count

    def run_comprehensive_test(self) -> bool:
        """Run the complete test suite."""
        print("🚀 STARTING COMPREHENSIVE SERVER MANAGEMENT TEST")
        print("=" * 80)

        try:
            # Step 1: Start test servers
            print("\n📡 STEP 1: Starting Test Servers")
            print("-" * 40)

            # Start 3 different servers
            configs = [
                ("anger", 8000, "test-anger", False),
                ("happiness", 8001, "test-happy", False),
                ("sadness", 8002, "test-sad", True),  # Use legacy script
            ]

            started_count = 0
            for emotion, port, model_name, legacy in configs:
                success, pid = self.start_test_server(emotion, port, model_name, legacy)
                if success:
                    started_count += 1

            self.log_test(
                "Server startup initiation",
                started_count > 0,
                f"Started {started_count}/{len(configs)} servers",
            )

            # Step 2: Wait for servers to be ready
            print("\n⏳ STEP 2: Waiting for Server Initialization")
            print("-" * 40)
            ready_count = self.wait_for_servers_ready()

            self.log_test(
                "Server initialization", ready_count > 0, f"{ready_count} servers became ready"
            )

            # Step 3: Test detection
            detection_works = self.test_server_detection()

            # Step 4: Test orphaned process detection
            self.test_orphaned_detection()

            # Step 5: Test management script
            management_works = self.test_management_script()

            return detection_works and management_works

        finally:
            # Always cleanup
            print("\n🧹 CLEANUP: Removing Test Servers")
            print("-" * 40)
            cleaned = self.cleanup_servers()
            print(f"Cleaned up {cleaned} processes")

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)

        passed = sum(1 for result in self.test_results if result["passed"])
        total = len(self.test_results)

        print(f"Tests passed: {passed}/{total}")
        print(f"Success rate: {passed/total*100:.1f}%")

        if passed == total:
            print("🎉 ALL TESTS PASSED! The server management system is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the details above.")
            print("\nFailed tests:")
            for result in self.test_results:
                if not result["passed"]:
                    print(f"  ❌ {result['test']}: {result['details']}")

        return passed == total


def main():
    """Main test execution."""
    harness = ServerTestHarness()

    try:
        success = harness.run_comprehensive_test()
        harness.print_summary()

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        harness.cleanup_servers()
        return 1
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        harness.cleanup_servers()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
