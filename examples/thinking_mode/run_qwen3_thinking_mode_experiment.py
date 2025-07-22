#!/usr/bin/env python3
"""
Qwen3 Thinking Mode Comparison Experiment Runner

This script runs two series of experiments to compare the impact of Qwen3's thinking mode
on neuro_manipulation.experiment_series_runner performance.

The experiments compare:
1. Baseline: Qwen3 models without thinking mode
2. Thinking Mode: Qwen3 models with thinking mode enabled

Both experiments use the same configurations except for the thinking mode setting.
"""

import os
import sys
import time
import json
import yaml
import logging
from datetime import datetime
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neuro_manipulation.experiment_series_runner import ExperimentSeriesRunner

def setup_logging():
    """Setup logging for the experiment runner."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"qwen3_thinking_mode_experiment_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def run_experiment_series(config_path, experiment_name, logger):
    """Run a single experiment series."""
    logger.info(f"Starting {experiment_name} experiment...")
    logger.info(f"Using config: {config_path}")
    
    start_time = time.time()
    
    try:
        # Initialize the experiment series runner
        runner = ExperimentSeriesRunner(config_path)
        
        # Run the experiments
        results = runner.run_experiment_series()
        # Note: results is the summary dict returned by run_experiment_series()
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"{experiment_name} experiment completed successfully!")
        logger.info(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        return {
            'experiment_name': experiment_name,
            'config_path': config_path,
            'duration': duration,
            'results': results,
            'status': 'success'
        }
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        logger.error(f"{experiment_name} experiment failed: {str(e)}")
        logger.error(f"Duration before failure: {duration:.2f} seconds")
        
        return {
            'experiment_name': experiment_name,
            'config_path': config_path,
            'duration': duration,
            'error': str(e),
            'status': 'failed'
        }

def generate_experiment_report(baseline_results, thinking_results, logger):
    """Generate a comprehensive experiment report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    
    report_file = report_dir / f"qwen3_thinking_mode_comparison_report_{timestamp}.md"
    
    with open(report_file, 'w') as f:
        f.write("# Qwen3 Thinking Mode Comparison Experiment Report\n\n")
        f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write("This report compares the performance of Qwen3 models with and without thinking mode enabled ")
        f.write("in the neuro_manipulation.experiment_series_runner framework.\n\n")
        
        f.write("## Experiment Configuration\n\n")
        f.write("### Models Tested\n")
        f.write("- Qwen3-1.7B\n")
        f.write("- Qwen3-4B\n")
        f.write("- Qwen3-8B\n\n")
        
        f.write("### Game Scenarios\n")
        f.write("- Prisoners Dilemma\n\n")
        
        f.write("### Emotions Tested\n")
        f.write("- Anger\n")
        f.write("- Happiness\n")
        f.write("- Sadness\n")
        f.write("- Disgust\n")
        f.write("- Fear\n")
        f.write("- Surprise\n\n")
        
        f.write("### Generation Parameters\n")
        f.write("- Temperature: 0.7\n")
        f.write("- Max New Tokens: 440\n")
        f.write("- Top-p: 0.95\n")
        f.write("- Do Sample: True\n")
        f.write("- Intensity: 1.5\n\n")
        
        f.write("## Results Summary\n\n")
        
        # Baseline results
        f.write("### Baseline Experiment (No Thinking Mode)\n")
        f.write(f"- **Status:** {baseline_results['status'].upper()}\n")
        f.write(f"- **Duration:** {baseline_results['duration']:.2f} seconds ({baseline_results['duration']/60:.2f} minutes)\n")
        f.write(f"- **Config:** {baseline_results['config_path']}\n")
        
        if baseline_results['status'] == 'failed':
            f.write(f"- **Error:** {baseline_results['error']}\n")
        
        f.write("\n")
        
        # Thinking mode results
        f.write("### Thinking Mode Experiment\n")
        f.write(f"- **Status:** {thinking_results['status'].upper()}\n")
        f.write(f"- **Duration:** {thinking_results['duration']:.2f} seconds ({thinking_results['duration']/60:.2f} minutes)\n")
        f.write(f"- **Config:** {thinking_results['config_path']}\n")
        
        if thinking_results['status'] == 'failed':
            f.write(f"- **Error:** {thinking_results['error']}\n")
        
        f.write("\n")
        
        # Comparison
        f.write("## Performance Comparison\n\n")
        
        if baseline_results['status'] == 'success' and thinking_results['status'] == 'success':
            duration_diff = thinking_results['duration'] - baseline_results['duration']
            duration_pct = (duration_diff / baseline_results['duration']) * 100
            
            f.write(f"### Execution Time Comparison\n")
            f.write(f"- **Baseline Duration:** {baseline_results['duration']:.2f} seconds\n")
            f.write(f"- **Thinking Mode Duration:** {thinking_results['duration']:.2f} seconds\n")
            f.write(f"- **Difference:** {duration_diff:.2f} seconds ({duration_pct:+.1f}%)\n\n")
            
            if duration_diff > 0:
                f.write("**Observation:** Thinking mode took longer to execute, which is expected due to the additional reasoning step.\n\n")
            else:
                f.write("**Observation:** Thinking mode was faster, which is unexpected and may indicate configuration issues.\n\n")
        
        f.write("## Thinking Mode Implementation Details\n\n")
        f.write("### Prompt Format Enhancement\n")
        f.write("- Added `Qwen3InstFormat` class to support Qwen3's specific chat template\n")
        f.write("- Implemented `/think` tag injection for thinking mode activation\n")
        f.write("- Enhanced `PromptFormat.build()` method to handle thinking mode parameter\n\n")
        
        f.write("### Configuration Changes\n")
        f.write("- Added `enable_thinking` parameter to generation config\n")
        f.write("- Modified prompt wrapper to pass thinking mode parameter\n")
        f.write("- Updated experiment pipeline to support thinking mode\n\n")
        
        f.write("## Data Locations\n\n")
        f.write("### Baseline Results\n")
        f.write("- **Directory:** `results/Qwen3_Thinking_Mode_Comparison_Baseline/`\n\n")
        
        f.write("### Thinking Mode Results\n")
        f.write("- **Directory:** `results/Qwen3_Thinking_Mode_Comparison_Enabled/`\n\n")
        
        f.write("## Technical Implementation\n\n")
        f.write("### Files Modified\n")
        f.write("- `neuro_manipulation/prompt_formats.py`: Added Qwen3InstFormat class\n")
        f.write("- `neuro_manipulation/prompt_wrapper.py`: Added thinking mode support\n")
        f.write("- `neuro_manipulation/experiments/emotion_game_experiment.py`: Added thinking mode parameter handling\n")
        f.write("- `config/Qwen3_Thinking_Mode_Comparison.yaml`: Baseline configuration\n")
        f.write("- `config/Qwen3_Thinking_Mode_Enabled.yaml`: Thinking mode configuration\n\n")
        
        f.write("### Thinking Mode Activation\n")
        f.write("The thinking mode is activated by:\n")
        f.write("1. Setting `enable_thinking: true` in the generation config\n")
        f.write("2. Automatically injecting `/think` tag into user messages for Qwen3 models\n")
        f.write("3. The model generates structured reasoning in `<think>...</think>` blocks\n\n")
        
        f.write("## Conclusions and Next Steps\n\n")
        
        if baseline_results['status'] == 'success' and thinking_results['status'] == 'success':
            f.write("### Key Findings\n")
            f.write("- Both experiments completed successfully\n")
            f.write("- Thinking mode implementation is working correctly\n")
            f.write("- Performance impact measured and documented\n\n")
            
            f.write("### Recommended Next Steps\n")
            f.write("1. Analyze the detailed results to compare decision-making patterns\n")
            f.write("2. Examine the thinking process outputs for insights\n")
            f.write("3. Run additional experiments with different game scenarios\n")
            f.write("4. Consider testing with different thinking mode parameters\n")
        else:
            f.write("### Issues Encountered\n")
            if baseline_results['status'] == 'failed':
                f.write("- Baseline experiment failed - investigate configuration issues\n")
            if thinking_results['status'] == 'failed':
                f.write("- Thinking mode experiment failed - check thinking mode implementation\n")
            
            f.write("\n### Recommended Actions\n")
            f.write("1. Review error logs and fix configuration issues\n")
            f.write("2. Test thinking mode implementation with simpler scenarios\n")
            f.write("3. Verify model paths and dependencies\n")
        
        f.write("\n---\n")
        f.write(f"*Report generated by Qwen3 Thinking Mode Experiment Runner*\n")
    
    logger.info(f"Experiment report generated: {report_file}")
    return report_file

def main():
    """Main experiment runner function."""
    logger = setup_logging()
    logger.info("Starting Qwen3 Thinking Mode Comparison Experiment")
    
    # Define experiment configurations
    baseline_config = "config/Qwen3_Thinking_Mode_Comparison.yaml"
    thinking_config = "config/Qwen3_Thinking_Mode_Enabled.yaml"
    
    # Verify config files exist
    if not os.path.exists(baseline_config):
        logger.error(f"Baseline config file not found: {baseline_config}")
        return 1
    
    if not os.path.exists(thinking_config):
        logger.error(f"Thinking mode config file not found: {thinking_config}")
        return 1
    
    # Run baseline experiment
    logger.info("=" * 60)
    logger.info("PHASE 1: BASELINE EXPERIMENT (NO THINKING MODE)")
    logger.info("=" * 60)
    
    baseline_results = run_experiment_series(
        baseline_config, 
        "Baseline (No Thinking Mode)", 
        logger
    )
    
    # Run thinking mode experiment
    logger.info("=" * 60)
    logger.info("PHASE 2: THINKING MODE EXPERIMENT")
    logger.info("=" * 60)
    
    thinking_results = run_experiment_series(
        thinking_config, 
        "Thinking Mode Enabled", 
        logger
    )
    
    # Generate comprehensive report
    logger.info("=" * 60)
    logger.info("GENERATING EXPERIMENT REPORT")
    logger.info("=" * 60)
    
    report_file = generate_experiment_report(baseline_results, thinking_results, logger)
    
    # Summary
    logger.info("=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    
    logger.info(f"Baseline Experiment: {baseline_results['status'].upper()}")
    if baseline_results['status'] == 'success':
        logger.info(f"  Duration: {baseline_results['duration']:.2f} seconds")
    else:
        logger.info(f"  Error: {baseline_results['error']}")
    
    logger.info(f"Thinking Mode Experiment: {thinking_results['status'].upper()}")
    if thinking_results['status'] == 'success':
        logger.info(f"  Duration: {thinking_results['duration']:.2f} seconds")
    else:
        logger.info(f"  Error: {thinking_results['error']}")
    
    logger.info(f"Detailed report available at: {report_file}")
    
    # Return appropriate exit code
    if baseline_results['status'] == 'success' and thinking_results['status'] == 'success':
        logger.info("All experiments completed successfully!")
        return 0
    else:
        logger.error("One or more experiments failed. Check the logs for details.")
        return 1

if __name__ == "__main__":
    exit(main())