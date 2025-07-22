#!/usr/bin/env python3
"""
Example script for running a sanity check on prompt-based experiments.

This script demonstrates how to:
1. Load configuration from YAML files
2. Set up a prompt experiment
3. Run a quick sanity check with limited samples
4. Validate the experiment setup before full execution

Usage:
    python examples/run_prompt_experiment_sanity_check.py
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from prompt_experiment import PromptExperiment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PromptExperimentSanityCheck(PromptExperiment):
    """Extended PromptExperiment class with sanity check functionality."""
    
    def run_sanity_check(self, num_samples=5):
        """
        Run a sanity check with a limited number of samples.
        
        Args:
            num_samples: Number of samples to test (default: 5)
            
        Returns:
            dict: Summary of sanity check results
        """
        logger.info(f"Starting sanity check with {num_samples} samples")
        
        # Store original configuration
        original_config = self.config.copy()
        original_output_dir = self.output_dir
        
        # Create sanity check output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        sanity_dir = Path(original_output_dir).parent / f"sanity_check_{timestamp}"
        sanity_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = sanity_dir
        
        # Limit the number of scenarios for testing
        if 'data' in self.config['experiment']:
            self.config['experiment']['data']['num_scenarios'] = num_samples
            self.config['experiment']['data']['batch_size'] = min(
                num_samples, 
                self.config['experiment']['data'].get('batch_size', 5)
            )
        
        # Limit emotions and intensities for quick testing
        self.config['experiment']['emotions'] = self.config['experiment']['emotions'][:1]
        self.config['experiment']['intensity'] = self.config['experiment']['intensity'][:1]
        self.config['experiment']['repeat'] = 1
        
        try:
            # Run the experiment steps
            logger.info("Step 1: Checking data creation...")
            if self.should_generate_data():
                logger.info(f"Data generation enabled - would create {num_samples} scenarios")
            else:
                logger.info("Data generation disabled - using existing data")
            
            logger.info("Step 2: Running limited API tests...")
            output_files = self.run_api_tests()
            
            logger.info("Step 3: Checking statistical analysis...")
            self.run_statistical_analysis()
            
            # Generate sanity check report
            report = self._generate_sanity_report(output_files)
            
            # Save report
            report_path = sanity_dir / "sanity_check_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Sanity check completed. Report saved to: {report_path}")
            
            # Print summary
            self._print_sanity_summary(report)
            
            return report
            
        finally:
            # Restore original configuration
            self.config = original_config
            self.output_dir = original_output_dir
    
    def _generate_sanity_report(self, output_files):
        """Generate a summary report of the sanity check."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'config_file': self.config_path,
            'experiment_name': self.config['experiment']['name'],
            'model': self.config['experiment']['llm']['llm_config'],
            'game': self.config['experiment']['game']['name'],
            'tested_emotion': self.config['experiment']['emotions'][0],
            'tested_intensity': self.config['experiment']['intensity'][0],
            'output_files': output_files,
            'checks': {}
        }
        
        # Check if output files were created
        report['checks']['files_created'] = all(Path(f).exists() for f in output_files)
        
        # Check if results contain expected fields
        if output_files:
            with open(output_files[0], 'r') as f:
                sample_results = json.load(f)
                if sample_results:
                    sample = sample_results[0]
                    expected_fields = ['emotion', 'intensity', 'decision', 'rationale']
                    report['checks']['has_expected_fields'] = all(
                        field in sample for field in expected_fields
                    )
                    report['sample_output'] = sample
        
        # Check analysis file
        analysis_file = Path(self.output_dir) / "analysis_results.json"
        report['checks']['analysis_created'] = analysis_file.exists()
        
        return report
    
    def _print_sanity_summary(self, report):
        """Print a formatted summary of the sanity check."""
        print("\n" + "="*60)
        print("SANITY CHECK SUMMARY")
        print("="*60)
        print(f"Experiment: {report['experiment_name']}")
        print(f"Model: {report['model']}")
        print(f"Game: {report['game']}")
        print(f"Tested Emotion: {report['tested_emotion']}")
        print(f"Tested Intensity: {report['tested_intensity']}")
        print("\nChecks:")
        for check, result in report['checks'].items():
            status = "✅" if result else "❌"
            print(f"  {status} {check}: {result}")
        
        if 'sample_output' in report:
            print("\nSample Output:")
            print(f"  Decision: {report['sample_output'].get('decision', 'N/A')}")
            print(f"  Emotion: {report['sample_output'].get('emotion', 'N/A')}")
            print(f"  Intensity: {report['sample_output'].get('intensity', 'N/A')}")
        
        print("="*60 + "\n")


def main():
    """Main function to run the sanity check."""
    # Available config files
    config_files = [
        "config/priDeli_experiment_config.yaml",
        "config/trusteeGame_experiment_config.yaml",
        # Add more config files as needed
    ]
    
    # Select config file
    print("Available experiment configurations:")
    for i, config in enumerate(config_files):
        print(f"{i+1}. {config}")
    
    # For this example, use the first config
    config_path = config_files[0]
    
    if not Path(config_path).exists():
        logger.error(f"Configuration file not found: {config_path}")
        return
    
    print(f"\nUsing configuration: {config_path}")
    
    try:
        # Create experiment instance with sanity check capability
        experiment = PromptExperimentSanityCheck(config_path)
        
        # Run sanity check
        report = experiment.run_sanity_check(num_samples=5)
        
        # Check if all tests passed
        all_passed = all(report['checks'].values())
        
        if all_passed:
            print("✅ All sanity checks passed!")
            print("You can now run the full experiment with confidence.")
            print(f"\nTo run the full experiment:")
            print(f"  python prompt_experiment.py")
        else:
            print("⚠️  Some sanity checks failed.")
            print("Please review the report and fix any issues before running the full experiment.")
            
    except Exception as e:
        logger.error(f"Sanity check failed: {e}", exc_info=True)
        print(f"❌ Sanity check failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())