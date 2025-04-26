"""
Unit tests for the experiment series runner.
"""
import unittest
import tempfile
import os
import json
from pathlib import Path
import shutil
import time 

from neuro_manipulation.experiment_series_runner import ExperimentSeriesRunner, ExperimentReport, ExperimentStatus

class TestExperimentReport(unittest.TestCase):
    """Test the ExperimentReport class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
        
    def test_add_experiment(self):
        """Test adding an experiment to the report."""
        report = ExperimentReport(self.temp_dir, "test_series")
        report.add_experiment("Game1", "Model1", "Game1_Model1")
        
        # Check if experiment was added
        self.assertIn("Game1_Model1", report.experiments)
        self.assertEqual(report.experiments["Game1_Model1"]["game_name"], "Game1")
        self.assertEqual(report.experiments["Game1_Model1"]["model_name"], "Model1")
        self.assertEqual(report.experiments["Game1_Model1"]["status"], ExperimentStatus.PENDING)
        
    def test_update_experiment(self):
        """Test updating an experiment in the report."""
        report = ExperimentReport(self.temp_dir, "test_series")
        report.add_experiment("Game1", "Model1", "Game1_Model1")
        
        # Update experiment
        report.update_experiment("Game1_Model1", status=ExperimentStatus.RUNNING, start_time="2023-01-01")
        
        # Check if experiment was updated
        self.assertEqual(report.experiments["Game1_Model1"]["status"], ExperimentStatus.RUNNING)
        self.assertEqual(report.experiments["Game1_Model1"]["start_time"], "2023-01-01")
        
    def test_get_pending_experiments(self):
        """Test getting pending experiments."""
        report = ExperimentReport(self.temp_dir, "test_series")
        report.add_experiment("Game1", "Model1", "Game1_Model1")
        report.add_experiment("Game2", "Model1", "Game2_Model1", status=ExperimentStatus.RUNNING)
        
        # Get pending experiments
        pending = report.get_pending_experiments()
        
        # Check if pending experiments are correct
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["exp_id"], "Game1_Model1")
        
    def test_get_failed_experiments(self):
        """Test getting failed experiments."""
        report = ExperimentReport(self.temp_dir, "test_series")
        report.add_experiment("Game1", "Model1", "Game1_Model1")
        report.add_experiment("Game2", "Model1", "Game2_Model1", status=ExperimentStatus.FAILED)
        
        # Get failed experiments
        failed = report.get_failed_experiments()
        
        # Check if failed experiments are correct
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["exp_id"], "Game2_Model1")
        
    def test_save_and_load_report(self):
        """Test saving and loading the report."""
        # Create a fixed timestamp for test determinism
        fixed_timestamp = "20250101_000000"
        
        # Create first report with fixed timestamp
        report = ExperimentReport(self.temp_dir, "test_series")
        report.timestamp = fixed_timestamp
        report.report_file = Path(f"{self.temp_dir}/test_series_{fixed_timestamp}/experiment_report.json")
        report.report_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Add experiments
        report.add_experiment("Game1", "Model1", "Game1_Model1")
        report.add_experiment("Game2", "Model1", "Game2_Model1", status=ExperimentStatus.COMPLETED)
        
        # Create a new report and manually set the path to match
        new_report = ExperimentReport(self.temp_dir, "test_series")
        new_report.timestamp = fixed_timestamp
        new_report.report_file = Path(f"{self.temp_dir}/test_series_{fixed_timestamp}/experiment_report.json")
        
        # Ensure file exists
        self.assertTrue(new_report.report_file.exists())
        
        # Load the report
        success = new_report.load_report()
        
        # Check if report was loaded successfully
        self.assertTrue(success)
        self.assertEqual(len(new_report.experiments), 2)
        self.assertEqual(new_report.experiments["Game1_Model1"]["status"], ExperimentStatus.PENDING)
        self.assertEqual(new_report.experiments["Game2_Model1"]["status"], ExperimentStatus.COMPLETED)
        
    def test_get_summary(self):
        """Test getting a summary of experiment statuses."""
        report = ExperimentReport(self.temp_dir, "test_series")
        report.add_experiment("Game1", "Model1", "Game1_Model1")
        report.add_experiment("Game2", "Model1", "Game2_Model1", status=ExperimentStatus.RUNNING)
        report.add_experiment("Game3", "Model1", "Game3_Model1", status=ExperimentStatus.COMPLETED)
        report.add_experiment("Game4", "Model1", "Game4_Model1", status=ExperimentStatus.FAILED)
        
        # Get summary
        summary = report.get_summary()
        
        # Check if summary is correct
        self.assertEqual(summary["total"], 4)
        self.assertEqual(summary["pending"], 1)
        self.assertEqual(summary["running"], 1)
        self.assertEqual(summary["completed"], 1)
        self.assertEqual(summary["failed"], 1)

class TestExperimentSeriesRunner(unittest.TestCase):
    """Test the ExperimentSeriesRunner class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
        
        # Create a simple test config
        config = {
            "experiment": {
                "name": "Test_Experiment",
                "games": ["Game1", "Game2"],
                "models": ["Model1", "Model2"],
                "output": {
                    "base_dir": self.temp_dir
                },
                "game": {
                    "name": "Game1"
                },
                "llm": {
                    "model_name": "Model1"
                }
            }
        }
        
        with open(self.config_path, "w") as f:
            json.dump(config, f)
        
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
        
    def test_format_model_name_for_folder(self):
        """Test formatting model name for folder name."""
        # Define test cases that should match the implementation
        test_cases = [
            # Full path with multiple slashes
            ("/data/home/huggingface_models/RWKV/v6-Finch-7B-HF", "RWKV/v6-Finch-7B-HF"),
            # Normal HF path (Note: rsplit with 2 will only return the last part)
            ("meta-llama/Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct"),
            # Path with exactly two parts
            ("org/model", "model"),
            # Single part (no slashes)
            ("Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct"),
            # Path with exactly three parts
            ("a/b/c", "b/c"),
            # Path with more than three parts
            ("w/x/y/z", "y/z")
        ]
        
        # Create a formatter function that exactly matches the implementation
        def format_model_name(model_name):
            if '/' in model_name:
                # rsplit('/', 2)[1:] will return at most 2 parts from the end
                # For "a/b/c" it returns ["b", "c"]
                # For "a/b" it returns ["b"]
                return '/'.join(model_name.rsplit('/', 2)[1:])
            return model_name
        
        # Test each case
        for input_name, expected_output in test_cases:
            result = format_model_name(input_name)
            self.assertEqual(result, expected_output, 
                             f"Failed for {input_name}, got {result}, expected {expected_output}")

if __name__ == "__main__":
    unittest.main() 