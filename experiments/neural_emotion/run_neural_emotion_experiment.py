#!/usr/bin/env python3
"""
Run neural emotion activation experiment comparing neutral vs anger conditions.
This script runs experiments sequentially to avoid VRAM limitations.
"""

import subprocess
import time
import os
import sys
import signal
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def kill_server_process():
    """Kill any existing OpenAI server processes."""
    try:
        subprocess.run(["pkill", "-f", "python -m openai_server"], check=False)
        time.sleep(2)
    except Exception as e:
        logger.warning(f"Error killing server process: {e}")

def start_server(emotion, port=8000):
    """Start OpenAI server with specified emotion."""
    logger.info(f"Starting server with emotion: {emotion} on port {port}")
    
    model_path = "/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct"
    model_name = "Qwen2.5-0.5B-Instruct"
    
    # Set environment variables for better memory management
    env = os.environ.copy()
    env["VLLM_GPU_MEMORY_UTILIZATION"] = "0.8"  # Use less GPU memory
    
    cmd = [
        "python", "-m", "openai_server",
        "--model", model_path,
        "--model_name", model_name,
        "--emotion", emotion,
        "--port", str(port)
    ]
    
    # Start server in background
    log_file = f"{emotion}_server.log"
    with open(log_file, "w") as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    
    # Wait for server to start
    logger.info("Waiting for server to initialize...")
    for i in range(60):  # Wait up to 60 seconds
        time.sleep(1)
        # Check if server is ready by looking for the Uvicorn message
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                content = f.read()
                if "Uvicorn running on" in content:
                    logger.info(f"Server started successfully after {i+1} seconds")
                    return process
    
    logger.error("Server failed to start within timeout")
    process.terminate()
    raise RuntimeError("Server startup timeout")

def run_experiment(config_file, experiment_name):
    """Run a single experiment with the given configuration."""
    logger.info(f"Running experiment: {experiment_name}")
    logger.info(f"Config: {config_file}")
    
    cmd = ["python", "prompt_experiment.py", config_file]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Experiment {experiment_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Experiment {experiment_name} failed:")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return False

def main():
    """Run the complete neural emotion activation experiment."""
    logger.info("="*60)
    logger.info("Neural Emotion Activation Experiment")
    logger.info("Comparing neutral baseline vs anger activation")
    logger.info("="*60)
    
    # Ensure we start clean
    kill_server_process()
    
    try:
        # Experiment 1: Baseline (using happiness as a mild emotion for baseline)
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT 1: BASELINE (using happiness emotion)")
        logger.info("="*60)
        
        # Start server with happiness (mild positive emotion as baseline)
        server_process = start_server("happiness", port=8000)
        
        # Update config to use correct port
        logger.info("Running baseline experiment...")
        success1 = run_experiment("config/priDeli_neural_test_config.yaml", "baseline")
        
        # Stop server
        logger.info("Stopping baseline server...")
        server_process.terminate()
        server_process.wait(timeout=10)
        kill_server_process()
        
        # Wait a bit to free VRAM
        logger.info("Waiting for VRAM to be freed...")
        time.sleep(5)
        
        # Experiment 2: Anger activation
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT 2: ANGER ACTIVATION")
        logger.info("="*60)
        
        # Start server with anger
        server_process = start_server("anger", port=8001)
        
        # Run anger experiment
        logger.info("Running anger activation experiment...")
        success2 = run_experiment("config/priDeli_neural_anger_test_config.yaml", "anger")
        
        # Stop server
        logger.info("Stopping anger server...")
        server_process.terminate()
        server_process.wait(timeout=10)
        kill_server_process()
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("="*60)
        logger.info(f"Baseline experiment: {'SUCCESS' if success1 else 'FAILED'}")
        logger.info(f"Anger experiment: {'SUCCESS' if success2 else 'FAILED'}")
        
        if success1 and success2:
            logger.info("\n✅ Both experiments completed successfully!")
            logger.info("Results are saved in:")
            logger.info("  - Baseline: results/neural_test/")
            logger.info("  - Anger: results/neural_test/")
            logger.info("\nNext step: Analyze the results to compare cooperation rates")
            return 0
        else:
            logger.error("\n❌ Some experiments failed. Check logs for details.")
            return 1
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        kill_server_process()
        return 1
    finally:
        # Ensure cleanup
        kill_server_process()

if __name__ == "__main__":
    sys.exit(main())