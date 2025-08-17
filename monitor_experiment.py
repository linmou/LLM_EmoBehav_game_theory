#!/usr/bin/env python3
"""
Monitor experiment progress in real-time by tailing log files.
Usage: python monitor_experiment.py [log_file_path]
"""

import sys
import time
import re
from pathlib import Path


def tail_log(log_file, follow=True):
    """Tail a log file and show progress."""
    log_file = Path(log_file)
    
    if not log_file.exists():
        print(f"❌ Log file not found: {log_file}")
        return
    
    print(f"📊 Monitoring: {log_file}")
    print("="*60)
    
    # Track progress
    scenarios_processed = 0
    total_scenarios = 0
    current_emotion = "unknown"
    
    with open(log_file, 'r') as f:
        # Read existing content
        for line in f:
            parse_and_display_line(line.rstrip())
        
        if follow:
            # Follow new content
            while True:
                line = f.readline()
                if line:
                    parse_and_display_line(line.rstrip())
                else:
                    time.sleep(0.1)


def parse_and_display_line(line):
    """Parse and display important log lines."""
    # Extract key information
    if "Randomly sampled" in line:
        print(f"🎲 {line}")
    elif "Total requests:" in line:
        print(f"📊 {line}")
    elif "Processing" in line and "emotion" in line:
        print(f"🧠 {line}")
    elif "Running iteration" in line:
        print(f"🔄 {line}")
    elif "Results saved" in line:
        print(f"💾 {line}")
    elif "Analysis results" in line:
        print(f"📈 {line}")
    elif "completed successfully" in line:
        print(f"✅ {line}")
    elif any(keyword in line.lower() for keyword in ["error", "failed", "exception"]):
        print(f"❌ {line}")
    elif "Step" in line and ("Running" in line or "Creating" in line):
        print(f"📋 {line}")
    elif "experiment:" in line:
        print(f"🎯 {line}")


def find_latest_logs():
    """Find the latest experiment log files."""
    log_dir = Path("results/neural_test/logs")
    if not log_dir.exists():
        print("❌ No log directory found")
        return []
    
    log_files = list(log_dir.glob("*_experiment_*.log"))
    if not log_files:
        print("❌ No experiment log files found")
        return []
    
    # Sort by modification time
    log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return log_files


def main():
    """Main monitoring function."""
    if len(sys.argv) > 1:
        # Monitor specific log file
        log_file = sys.argv[1]
        tail_log(log_file)
    else:
        # Find and monitor latest logs
        print("🔍 Finding latest experiment logs...")
        log_files = find_latest_logs()
        
        if not log_files:
            print("\n💡 Usage:")
            print("  python monitor_experiment.py [log_file_path]")
            print("\n📋 Or run an experiment first:")
            print("  python run_emotion_comparison.py")
            return
        
        print(f"\n📋 Found {len(log_files)} recent log files:")
        for i, log_file in enumerate(log_files[:5]):  # Show top 5
            mod_time = time.strftime('%H:%M:%S', time.localtime(log_file.stat().st_mtime))
            print(f"  {i+1}. {log_file.name} (modified: {mod_time})")
        
        # Monitor the most recent
        latest_log = log_files[0]
        print(f"\n👀 Monitoring latest: {latest_log.name}")
        
        try:
            tail_log(latest_log)
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped")


if __name__ == "__main__":
    main()