#!/usr/bin/env python3
"""
Interactive OpenAI Server Manager

This script provides an interactive command-line interface to:
1. List all running OpenAI servers started by openai_server module
2. Kill selected servers or all servers
3. View server details including PID, model, emotion, and port
"""

import os
import re
import subprocess
import signal
import time
from typing import List, Optional, Set, Dict
from collections import defaultdict


class ServerProcess:
    """Represents a running OpenAI server process."""

    def __init__(self, pid: int, cmd: str):
        self.pid = pid
        self.cmd = cmd
        self.model = self._extract_model()
        self.emotion = self._extract_emotion()
        self.port = self._extract_port()
        self.model_name = self._extract_model_name()
        self.children = set()  # Will be populated later

    def _extract_model(self) -> Optional[str]:
        """Extract model path from command line."""
        match = re.search(r"--model\s+([^\s]+)", self.cmd)
        if match:
            model_path = match.group(1)
            # Get just the model name from path
            return os.path.basename(model_path)
        return "Unknown"

    def _extract_emotion(self) -> Optional[str]:
        """Extract emotion from command line."""
        match = re.search(r"--emotion\s+([^\s]+)", self.cmd)
        return match.group(1) if match else "None"

    def _extract_port(self) -> Optional[str]:
        """Extract port from command line."""
        match = re.search(r"--port\s+([^\s]+)", self.cmd)
        return match.group(1) if match else "8000"

    def _extract_model_name(self) -> Optional[str]:
        """Extract model name from command line."""
        match = re.search(r"--model_name\s+([^\s]+)", self.cmd)
        return match.group(1) if match else "Default"

    def __str__(self):
        children_str = f" (+{len(self.children)} children)" if self.children else ""
        return (
            f"PID: {self.pid:>6}{children_str} | Port: {self.port:>4} | "
            f"Model: {self.model:<20} | Emotion: {self.emotion:<8} | "
            f"Name: {self.model_name}"
        )


class ServerManager:
    """Manages OpenAI server processes."""

    def __init__(self):
        self.servers: List[ServerProcess] = []
        self.all_related_pids: Set[int] = set()
        self.process_tree: Dict[int, Set[int]] = defaultdict(set)

    def find_process_tree(self, parent_pid: int) -> Set[int]:
        """Recursively find all child processes of a given parent."""
        children = set()
        
        try:
            # Method 1: Use pgrep
            result = subprocess.run(
                ["pgrep", "-P", str(parent_pid)], 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line:
                        try:
                            child_pid = int(line)
                            children.add(child_pid)
                            # Recursively find children of children
                            children.update(self.find_process_tree(child_pid))
                        except ValueError:
                            pass
        except (subprocess.CalledProcessError, FileNotFoundError):
            # pgrep might not be available, try ps
            try:
                result = subprocess.run(
                    ["ps", "--ppid", str(parent_pid), "-o", "pid="],
                    capture_output=True,
                    text=True
                )
                
                for line in result.stdout.strip().split("\n"):
                    if line:
                        try:
                            child_pid = int(line.strip())
                            children.add(child_pid)
                            children.update(self.find_process_tree(child_pid))
                        except ValueError:
                            pass
            except:
                pass
                
        return children

    def find_servers(self) -> List[ServerProcess]:
        """Find all running OpenAI server processes and their children."""
        try:
            # Look for processes containing openai_server
            result = subprocess.run(["ps", "auxww"], capture_output=True, text=True, check=True)

            servers = []
            for line in result.stdout.split("\n"):
                # Look for python processes running openai_server
                if (
                    "python" in line
                    and ("openai_server" in line or "init_openai_server" in line)
                    and "grep" not in line
                    and "--model" in line
                    and "manage_servers" not in line  # Don't include this script
                ):

                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            pid = int(parts[1])
                            # Find where the command starts - look for python
                            cmd_start = -1
                            for i, part in enumerate(parts):
                                if "python" in part:
                                    cmd_start = i
                                    break

                            if cmd_start >= 0:
                                cmd = " ".join(parts[cmd_start:])
                                server = ServerProcess(pid, cmd)
                                
                                # Find all child processes
                                server.children = self.find_process_tree(pid)
                                self.all_related_pids.add(pid)
                                self.all_related_pids.update(server.children)
                                
                                servers.append(server)
                        except (ValueError, IndexError):
                            continue

            self.servers = servers
            return servers

        except subprocess.CalledProcessError as e:
            print(f"Error finding servers: {e}")
            return []

    def find_orphaned_processes(self) -> List[int]:
        """Find orphaned multiprocessing and vLLM processes from OpenAI servers."""
        try:
            result = subprocess.run(["ps", "auxww"], capture_output=True, text=True, check=True)

            orphaned_pids = []
            gpu_holding_pids = set()
            
            # First get PIDs holding GPU memory
            try:
                gpu_result = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True
                )
                if gpu_result.returncode == 0:
                    for line in gpu_result.stdout.strip().split("\n"):
                        if line:
                            try:
                                gpu_holding_pids.add(int(line.strip()))
                            except ValueError:
                                pass
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
            
            for line in result.stdout.split("\n"):
                # Skip if already identified as server or child
                parts = line.split(None, 10)
                if len(parts) < 11:
                    continue
                    
                try:
                    pid = int(parts[1])
                    cmd = parts[10]
                    
                    # Skip if already tracked
                    if pid in self.all_related_pids:
                        continue
                    
                    # Look for multiprocessing processes that might be from OpenAI servers
                    if (
                        "python" in line
                        and (
                            "multiprocessing.resource_tracker" in cmd
                            or "multiprocessing.spawn" in cmd
                            or "vllm.worker" in cmd
                            or "vllm.engine" in cmd
                            or "ray::" in cmd
                        )
                        and "grep" not in line
                    ):
                        orphaned_pids.append(pid)
                    # Also include Python processes holding GPU memory that aren't tracked
                    elif pid in gpu_holding_pids and "python" in cmd:
                        orphaned_pids.append(pid)
                        
                except (ValueError, IndexError):
                    continue

            return orphaned_pids

        except subprocess.CalledProcessError as e:
            print(f"Error finding orphaned processes: {e}")
            return []

    def display_servers(self):
        """Display all found servers in a formatted table."""
        if not self.servers:
            print("No OpenAI servers found running.")
            return

        print("\n" + "=" * 100)
        print("RUNNING OPENAI SERVERS")
        print("=" * 100)
        print(
            f"{'#':>2} | {'PID':>6} | {'Port':>4} | {'Model':<20} | {'Emotion':<8} | {'Name':<15}"
        )
        print("-" * 100)

        for i, server in enumerate(self.servers, 1):
            print(f"{i:>2} | {server}")

        print("=" * 100)

    def kill_server(self, pid: int, force: bool = False) -> bool:
        """Kill a server by PID with improved handling."""
        # Check if process exists first
        try:
            os.kill(pid, 0)  # Signal 0 just checks if process exists
        except ProcessLookupError:
            # Process already dead
            return True
        except PermissionError:
            print(f"✗ No permission to kill PID {pid}")
            return False
        
        # Try SIGTERM first (unless force is specified)
        if not force:
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.2)  # Give it a moment
                
                # Check if it died
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    return True
            except (ProcessLookupError, PermissionError):
                pass
        
        # Try SIGKILL
        try:
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.1)
            
            # Final check
            try:
                os.kill(pid, 0)
                # Still alive - might be zombie
                return False
            except ProcessLookupError:
                return True
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        
        return False

    def kill_servers(self, indices: List[int]) -> int:
        """Kill servers and their children by display indices."""
        killed_count = 0
        for idx in indices:
            if 1 <= idx <= len(self.servers):
                server = self.servers[idx - 1]
                
                # Kill children first
                for child_pid in server.children:
                    if self.kill_server(child_pid, force=True):
                        killed_count += 1
                
                # Then kill parent
                if self.kill_server(server.pid):
                    print(f"✓ Killed server PID {server.pid} ({server.model} - {server.emotion})")
                    if server.children:
                        print(f"  Also killed {len(server.children)} child processes")
                    killed_count += 1
                else:
                    print(f"✗ Failed to kill server PID {server.pid}")
            else:
                print(f"✗ Invalid server number: {idx}")
        return killed_count

    def kill_all_servers(self) -> int:
        """Kill all found servers and their children."""
        killed_count = 0
        for server in self.servers:
            # Kill children first
            for child_pid in server.children:
                if self.kill_server(child_pid, force=True):
                    killed_count += 1
            
            # Then kill parent
            if self.kill_server(server.pid):
                print(f"✓ Killed server PID {server.pid} ({server.model} - {server.emotion})")
                if server.children:
                    print(f"  Also killed {len(server.children)} child processes")
                killed_count += 1
            else:
                print(f"✗ Failed to kill server PID {server.pid}")
        return killed_count


def parse_selection(selection: str, max_num: int) -> List[int]:
    """Parse user selection string into list of indices."""
    indices = []

    for part in selection.split(","):
        part = part.strip()
        if "-" in part:
            # Range selection like "1-3"
            try:
                start, end = map(int, part.split("-", 1))
                indices.extend(range(start, end + 1))
            except ValueError:
                continue
        else:
            # Single selection like "1"
            try:
                indices.append(int(part))
            except ValueError:
                continue

    # Remove duplicates and filter valid indices
    return list(set(idx for idx in indices if 1 <= idx <= max_num))


def main():
    """Main interactive loop."""
    manager = ServerManager()

    print("OpenAI Server Manager")
    print("=" * 50)

    while True:
        # Refresh server list
        servers = manager.find_servers()
        orphaned = manager.find_orphaned_processes()

        manager.display_servers()

        if orphaned:
            print(
                f"\n⚠️  Found {len(orphaned)} orphaned multiprocessing "
                f"processes (may be using VRAM):"
            )
            for pid in orphaned:
                print(f"   PID: {pid}")

        if not servers and not orphaned:
            print("\nNo servers or orphaned processes to manage.")
            break

        print("\nOptions:")
        if servers:
            print("  [1-N]: Kill specific server(s) (e.g., '1', '1,3', '1-3')")
            print("  [a/all]: Kill ALL servers")
        if orphaned:
            print("  [o/orphaned]: Kill orphaned multiprocessing processes")
        if servers and orphaned:
            print("  [c/cleanup]: Kill ALL servers AND orphaned processes")
        print("  [r/refresh]: Refresh server list")
        print("  [q/quit]: Quit")

        try:
            choice = input("\nEnter your choice: ").strip().lower()

            if choice in ["q", "quit"]:
                print("Goodbye!")
                break

            elif choice in ["r", "refresh"]:
                print("Refreshing server list...")
                continue

            elif choice in ["o", "orphaned"] and orphaned:
                confirm = input(f"Kill {len(orphaned)} orphaned processes? (y/N): ").strip().lower()
                if confirm in ["y", "yes"]:
                    killed = 0
                    for pid in orphaned:
                        if manager.kill_server(pid, force=True):  # Use force for orphaned processes
                            print(f"✓ Killed orphaned process PID {pid}")
                            killed += 1
                        else:
                            print(f"✗ Failed to kill orphaned process PID {pid}")
                    print(f"\nKilled {killed}/{len(orphaned)} orphaned processes.")
                    if killed > 0:
                        input("Press Enter to continue...")
                else:
                    print("Cancelled.")

            elif choice in ["c", "cleanup"] and (servers or orphaned):
                confirm = (
                    input(
                        f"Kill ALL {len(servers)} servers and "
                        f"{len(orphaned)} orphaned processes? (y/N): "
                    )
                    .strip()
                    .lower()
                )
                if confirm in ["y", "yes"]:
                    killed_servers = manager.kill_all_servers() if servers else 0
                    killed_orphaned = 0
                    for pid in orphaned:
                        if manager.kill_server(pid, force=True):  # Use force for orphaned processes
                            print(f"✓ Killed orphaned process PID {pid}")
                            killed_orphaned += 1
                        else:
                            print(f"✗ Failed to kill orphaned process PID {pid}")

                    print(
                        f"\nKilled {killed_servers}/{len(servers)} servers and "
                        f"{killed_orphaned}/{len(orphaned)} orphaned processes."
                    )
                    if killed_servers + killed_orphaned > 0:
                        input("Press Enter to continue...")
                else:
                    print("Cancelled.")

            elif choice in ["a", "all"] and servers:
                confirm = (
                    input(f"Are you sure you want to kill ALL {len(servers)} servers? (y/N): ")
                    .strip()
                    .lower()
                )
                if confirm in ["y", "yes"]:
                    killed = manager.kill_all_servers()
                    print(f"\nKilled {killed}/{len(servers)} servers.")
                    if killed > 0:
                        input("Press Enter to continue...")
                else:
                    print("Cancelled.")

            else:
                # Parse numeric selection for servers
                if servers:
                    indices = parse_selection(choice, len(servers))
                    if indices:
                        confirm = input(f"Kill server(s) {indices}? (y/N): ").strip().lower()
                        if confirm in ["y", "yes"]:
                            killed = manager.kill_servers(indices)
                            print(f"\nKilled {killed}/{len(indices)} servers.")
                            if killed > 0:
                                input("Press Enter to continue...")
                        else:
                            print("Cancelled.")
                    else:
                        print("Invalid selection. Try again.")
                else:
                    print("Invalid selection. Try again.")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
