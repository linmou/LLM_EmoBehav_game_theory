#!/usr/bin/env python3
"""
Run complete emotion comparison experiment: happiness vs anger with 300 scenarios each.
This script runs both experiments and compares the results automatically.
"""

import subprocess
import sys
import time
from pathlib import Path


def run_experiment(config_file, experiment_name):
    """Run a single experiment with given config."""
    print(f"\n{'='*60}")
    print(f"🚀 Starting {experiment_name} experiment...")
    print(f"📄 Config: {config_file}")
    print(f"🕐 Start time: {time.strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    
    # Create log file for this experiment
    log_dir = Path("results/neural_test/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{experiment_name.lower()}_experiment_{time.strftime('%Y%m%d_%H%M%S')}.log"
    
    start_time = time.time()
    
    try:
        # Run with output redirected to both console and log file
        with open(log_file, 'w') as f:
            f.write(f"Starting {experiment_name} experiment at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Config: {config_file}\n")
            f.write("="*60 + "\n\n")
            f.flush()
            
            print(f"📝 Logging to: {log_file}")
            
            result = subprocess.run([
                sys.executable, "prompt_experiment.py", config_file
            ], stdout=f, stderr=subprocess.STDOUT, timeout=3600)  # 60 minute timeout
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {experiment_name} experiment completed successfully!")
            print(f"⏱️  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            
            # Show quick summary from log
            with open(log_file, 'r') as f:
                log_content = f.read()
                
            if "Total requests:" in log_content:
                for line in log_content.split('\n'):
                    if "Total requests:" in line:
                        print(f"📊 {line}")
                        break
                        
            print(f"📝 Full log available at: {log_file}")
            return True, str(log_file)
        else:
            print(f"❌ {experiment_name} experiment failed!")
            print(f"Return code: {result.returncode}")
            print(f"📝 Error log available at: {log_file}")
            
            # Show last few lines of log for debugging
            with open(log_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    print("Last few log lines:")
                    for line in lines[-5:]:
                        print(f"  {line.rstrip()}")
            
            return False, f"Failed with return code {result.returncode}, log: {log_file}"
    
    except subprocess.TimeoutExpired:
        print(f"❌ {experiment_name} experiment timed out after 60 minutes!")
        print(f"📝 Timeout log available at: {log_file}")
        return False, f"Timeout, log: {log_file}"
    except Exception as e:
        print(f"❌ {experiment_name} experiment failed with exception: {e}")
        return False, str(e)


def find_result_files():
    """Find the most recent result files for happiness and anger."""
    results_dir = Path("results/neural_test")
    
    # Look for the most recent happiness and anger experiment directories
    happiness_dirs = list(results_dir.glob("*happiness*")) + list(results_dir.glob("*neutral*"))
    anger_dirs = list(results_dir.glob("*anger*"))
    
    if not happiness_dirs or not anger_dirs:
        print("❌ Could not find result directories!")
        return None, None
    
    # Get the most recent directories
    happiness_dir = max(happiness_dirs, key=lambda p: p.stat().st_mtime)
    anger_dir = max(anger_dirs, key=lambda p: p.stat().st_mtime)
    
    # Look for results files
    happiness_results = list(happiness_dir.glob("*results.json"))
    anger_results = list(anger_dir.glob("*results.json"))
    
    if not happiness_results or not anger_results:
        print("❌ Could not find results.json files!")
        return None, None
    
    return happiness_results[0], anger_results[0]


def main():
    """Run complete emotion comparison experiment."""
    print("🧠 Neural Emotion Comparison Experiment")
    print("Testing happiness vs anger in prisoner's dilemma with 300 scenarios each")
    print()
    
    # Check if servers are running
    print("Checking server status...")
    try:
        import requests
        
        # Test happiness server (port 8000)
        response = requests.get("http://localhost:8000/v1/models", timeout=5)
        if response.status_code == 200:
            print("✅ Happiness server (port 8000) is running")
        else:
            print("❌ Happiness server (port 8000) not responding")
            return
        
        # Test anger server (port 8001)  
        response = requests.get("http://localhost:8001/v1/models", timeout=5)
        if response.status_code == 200:
            print("✅ Anger server (port 8001) is running")
        else:
            print("❌ Anger server (port 8001) not responding")
            return
            
    except Exception as e:
        print(f"❌ Server check failed: {e}")
        print("Make sure both emotion servers are running:")
        print("  Happiness: port 8000")
        print("  Anger: port 8001")
        return
    
    overall_start = time.time()
    
    # Run happiness experiment
    happiness_success, happiness_log = run_experiment(
        "config/priDeli_neural_test_config.yaml", 
        "Happiness"
    )
    
    if not happiness_success:
        print("❌ Happiness experiment failed, aborting comparison")
        return
    
    # Run anger experiment
    anger_success, anger_log = run_experiment(
        "config/priDeli_neural_anger_test_config.yaml",
        "Anger"
    )
    
    if not anger_success:
        print("❌ Anger experiment failed, aborting comparison")
        return
    
    overall_end = time.time()
    total_duration = overall_end - overall_start
    
    print(f"\n{'='*60}")
    print("BOTH EXPERIMENTS COMPLETED!")
    print(f"Total time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"{'='*60}")
    
    # Find and compare results
    print("\nSearching for result files...")
    happiness_file, anger_file = find_result_files()
    
    if happiness_file and anger_file:
        print(f"Found happiness results: {happiness_file}")
        print(f"Found anger results: {anger_file}")
        
        # Run comparison
        print("\nRunning statistical comparison...")
        try:
            result = subprocess.run([
                sys.executable, "compare_emotion_results.py", 
                str(happiness_file), str(anger_file), "results/neural_test/comparison"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(result.stdout)
                print("✅ Comparison completed successfully!")
                print("📊 Results saved to results/neural_test/comparison/")
            else:
                print(f"❌ Comparison failed: {result.stderr}")
        except Exception as e:
            print(f"❌ Comparison failed: {e}")
    else:
        print("❌ Could not find result files for comparison")
        print("You can run comparison manually with:")
        print("python compare_emotion_results.py <happiness_results.json> <anger_results.json>")


if __name__ == "__main__":
    main()