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
from typing import List, Optional


class ServerProcess:
    """Represents a running OpenAI server process."""

    def __init__(self, pid: int, cmd: str):
        self.pid = pid
        self.cmd = cmd
        self.model = self._extract_model()
        self.emotion = self._extract_emotion()
        self.port = self._extract_port()
        self.model_name = self._extract_model_name()

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
        return (
            f"PID: {self.pid:>6} | Port: {self.port:>4} | "
            f"Model: {self.model:<20} | Emotion: {self.emotion:<8} | "
            f"Name: {self.model_name}"
        )


class ServerManager:
    """Manages OpenAI server processes."""

    def __init__(self):
        self.servers: List[ServerProcess] = []

    def find_servers(self) -> List[ServerProcess]:
        """Find all running OpenAI server processes."""
        try:
            # Look for processes containing openai_server
            result = subprocess.run(["ps", "aux"], capture_output=True, text=True, check=True)

            servers = []
            for line in result.stdout.split("\n"):
                # Look for python processes running openai_server
                if (
                    "python" in line
                    and ("openai_server" in line or "init_openai_server" in line)
                    and "grep" not in line
                    and "--model" in line
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
                                servers.append(ServerProcess(pid, cmd))
                        except (ValueError, IndexError):
                            continue

            self.servers = servers
            return servers

        except subprocess.CalledProcessError as e:
            print(f"Error finding servers: {e}")
            return []

    def find_orphaned_processes(self) -> List[int]:
        """Find orphaned multiprocessing processes from OpenAI servers."""
        try:
            result = subprocess.run(["ps", "aux"], capture_output=True, text=True, check=True)

            orphaned_pids = []
            for line in result.stdout.split("\n"):
                # Look for multiprocessing processes that might be from OpenAI servers
                if (
                    "python" in line
                    and (
                        "multiprocessing.resource_tracker" in line
                        or "multiprocessing.spawn" in line
                    )
                    and "grep" not in line
                ):

                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            pid = int(parts[1])
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

    def kill_server(self, pid: int) -> bool:
        """Kill a server by PID."""
        try:
            subprocess.run(["kill", str(pid)], check=True)
            return True
        except subprocess.CalledProcessError:
            try:
                # Try force kill
                subprocess.run(["kill", "-9", str(pid)], check=True)
                return True
            except subprocess.CalledProcessError:
                return False

    def kill_servers(self, indices: List[int]) -> int:
        """Kill servers by their display indices."""
        killed_count = 0
        for idx in indices:
            if 1 <= idx <= len(self.servers):
                server = self.servers[idx - 1]
                if self.kill_server(server.pid):
                    print(f"✓ Killed server PID {server.pid} ({server.model} - {server.emotion})")
                    killed_count += 1
                else:
                    print(f"✗ Failed to kill server PID {server.pid}")
            else:
                print(f"✗ Invalid server number: {idx}")
        return killed_count

    def kill_all_servers(self) -> int:
        """Kill all found servers."""
        killed_count = 0
        for server in self.servers:
            if self.kill_server(server.pid):
                print(f"✓ Killed server PID {server.pid} ({server.model} - {server.emotion})")
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
                        if manager.kill_server(pid):
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
                        if manager.kill_server(pid):
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
