"""
This module implements a pipeline for running series of experiments with different combinations
of game names and model names, with support for graceful shutdown and resumption.
"""
import time
import yaml
import signal
import os
import sys
import traceback
import json
from datetime import datetime
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional, Tuple
import threading

from constants import GameNames
from neuro_manipulation.repe import repe_pipeline_registry
from neuro_manipulation.configs.experiment_config import get_exp_config, get_repe_eng_config, get_model_config
from neuro_manipulation.experiments.emotion_game_experiment import EmotionGameExperiment
from games.game_configs import get_game_config

class ExperimentStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class ExperimentReport:
    """Manages and persists the status of experiments in a series"""
    
    def __init__(self, base_dir: str, experiment_series_name: str):
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_file = Path(f"{base_dir}/{experiment_series_name}_{self.timestamp}/experiment_report.json")
        self.report_file.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.experiments = {}
        self._save_report()
        
    def add_experiment(self, game_name: str, model_name: str, exp_id: str, status: str = ExperimentStatus.PENDING) -> None:
        """Add a new experiment to the report"""
        with self.lock:
            self.experiments[exp_id] = {
                "game_name": game_name,
                "model_name": model_name,
                "status": status,
                "start_time": None,
                "end_time": None,
                "error": None,
                "output_dir": None,
                "exp_id": exp_id
            }
            self._save_report()
    
    def update_experiment(self, exp_id: str, **kwargs) -> None:
        """Update experiment status and details"""
        with self.lock:
            if exp_id in self.experiments:
                self.experiments[exp_id].update(kwargs)
                self._save_report()
    
    def get_pending_experiments(self) -> List[Dict[str, Any]]:
        """Get list of pending experiments"""
        with self.lock:
            return [exp for exp in self.experiments.values() 
                   if exp["status"] == ExperimentStatus.PENDING]
    
    def get_failed_experiments(self) -> List[Dict[str, Any]]:
        """Get list of failed experiments"""
        with self.lock:
            return [exp for exp in self.experiments.values() 
                   if exp["status"] == ExperimentStatus.FAILED]
    
    def _save_report(self) -> None:
        """Save the report to disk"""
        with open(self.report_file, 'w') as f:
            json.dump({
                "last_updated": datetime.now().isoformat(),
                "experiments": self.experiments
            }, f, indent=2)
    
    def load_report(self) -> bool:
        """Load a report from disk if it exists"""
        if self.report_file.exists():
            with open(self.report_file, 'r') as f:
                report_data = json.load(f)
                self.experiments = report_data.get("experiments", {})
            return True
        return False
    
    def get_summary(self) -> Dict[str, int]:
        """Get summary of experiment statuses"""
        with self.lock:
            summary = {
                "total": len(self.experiments),
                "pending": sum(1 for exp in self.experiments.values() if exp["status"] == ExperimentStatus.PENDING),
                "running": sum(1 for exp in self.experiments.values() if exp["status"] == ExperimentStatus.RUNNING),
                "completed": sum(1 for exp in self.experiments.values() if exp["status"] == ExperimentStatus.COMPLETED),
                "failed": sum(1 for exp in self.experiments.values() if exp["status"] == ExperimentStatus.FAILED)
            }
            return summary

class ExperimentSeriesRunner:
    """Manages running a series of experiments with different game/model combinations"""
    
    def __init__(self, config_path: str, series_name: str = None, resume: bool = False):
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        self.config_path = config_path
        self.exp_config = get_exp_config(config_path)
        self.series_name = series_name or f"experiment_series"
        
        # Initialize shutdown flag
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._handle_shutdown)
        
        # Create or load experiment report
        base_dir = self.exp_config['experiment']['output']['base_dir']
        self.report = ExperimentReport(base_dir, self.series_name)
        
        # Check if resuming
        self.resume = resume
        if resume:
            if not self.report.load_report():
                self.logger.warning("No previous experiment report found. Starting fresh.")
            else:
                self.logger.info(f"Resumed experiment series. Status: {self.report.get_summary()}")
    
    def _handle_shutdown(self, sig, frame):
        """Handle SIGINT (Ctrl+C)"""
        if not self.shutdown_requested:
            self.logger.info("Shutdown requested. Finishing current experiment and stopping...")
            self.shutdown_requested = True
        else:
            self.logger.warning("Forced shutdown requested. Exiting immediately.")
            sys.exit(1)
    
    def _format_model_name_for_folder(self, model_name: str) -> str:
        """Format model name for folder name by removing path prefix"""
        if '/' in model_name:
            return '/'.join(model_name.rsplit('/', 2)[1:])
        return model_name
    
    def setup_experiment(self, game_name_str: str, model_name: str) -> EmotionGameExperiment:
        """Set up a single experiment with the given game and model"""
        repe_pipeline_registry()
        
        # Create a copy of the config that we can modify
        exp_config = dict(self.exp_config)
        
        # Update game and model in config
        exp_config['experiment']['game']['name'] = game_name_str
        exp_config['experiment']['llm']['model_name'] = model_name
        
        game_name = GameNames.from_string(game_name_str)
        repe_eng_config = get_repe_eng_config(model_name)
        game_config = get_game_config(game_name)
        
        if game_name.is_sequential():
            game_config['previous_actions_length'] = exp_config['experiment']['game']['previous_actions_length']
        
        experiment = EmotionGameExperiment(
            repe_eng_config, 
            exp_config, 
            game_config,
            repeat=exp_config['experiment']['repeat'],
            batch_size=exp_config['experiment'].get('batch_size', 300)
        )
        
        return experiment
    
    def run_single_experiment(self, game_name: str, model_name: str, exp_id: str) -> bool:
        """Run a single experiment with the specified game and model"""
        self.logger.info(f"Starting experiment with game: {game_name}, model: {model_name}")
        
        # Update experiment status
        self.report.update_experiment(
            exp_id, 
            status=ExperimentStatus.RUNNING,
            start_time=datetime.now().isoformat()
        )
        
        try:
            experiment = self.setup_experiment(game_name, model_name)
            
            # Record output directory for later reference
            output_dir = experiment.output_dir
            self.report.update_experiment(exp_id, output_dir=output_dir)
            
            # Run the experiment
            if self.exp_config['experiment'].get('run_sanity_check', False):
                experiment.run_sanity_check()
            else:
                experiment.run_experiment()
            
            # Update experiment status
            self.report.update_experiment(
                exp_id, 
                status=ExperimentStatus.COMPLETED,
                end_time=datetime.now().isoformat()
            )
            
            self.logger.info(f"Experiment completed: {game_name}, {model_name}")
            return True
            
        except Exception as e:
            error_trace = traceback.format_exc()
            self.logger.error(f"Experiment failed: {game_name}, {model_name}\nError: {str(e)}\n{error_trace}")
            
            # Update experiment status
            self.report.update_experiment(
                exp_id, 
                status=ExperimentStatus.FAILED,
                end_time=datetime.now().isoformat(),
                error=f"{str(e)}\n{error_trace}"
            )
            return False
    
    def run_experiment_series(self) -> None:
        """Run the full series of experiments with all game/model combinations"""
        # Get lists of games and models from config
        games = self.exp_config['experiment'].get('games', [])
        models = self.exp_config['experiment'].get('models', [])
        
        # If no games/models specified, use the defaults from config
        if not games:
            games = [self.exp_config['experiment']['game']['name']]
        
        if not models:
            models = [self.exp_config['experiment']['llm']['model_name']]
        
        self.logger.info(f"Starting experiment series with {len(games)} games and {len(models)} models")
        
        # Generate experiment IDs and initialize report
        for game_name in games:
            for model_name in models:
                model_folder_name = self._format_model_name_for_folder(model_name)
                exp_id = f"{game_name}_{model_folder_name}"
                
                # Only add if not resuming or not already in report
                if not self.resume or exp_id not in self.report.experiments:
                    self.report.add_experiment(game_name, model_name, exp_id)
        
        # Get pending experiments
        pending_experiments = self.report.get_pending_experiments()
        total_experiments = len(pending_experiments)
        self.logger.info(f"Total pending experiments: {total_experiments}")
        
        # Run each experiment
        for i, exp in enumerate(pending_experiments):
            if self.shutdown_requested:
                self.logger.info("Shutdown requested. Stopping experiment series.")
                break
            
            self.logger.info(f"Running experiment {i+1}/{total_experiments}: {exp['game_name']}, {exp['model_name']}")
            self.run_single_experiment(exp['game_name'], exp['model_name'], exp['exp_id'])
            
            # Print summary after each experiment
            summary = self.report.get_summary()
            self.logger.info(f"Experiment series progress: {summary}")
        
        # Final summary
        summary = self.report.get_summary()
        self.logger.info(f"Experiment series completed. Final status: {summary}")
        
        if summary['failed'] > 0:
            self.logger.info("Failed experiments:")
            for exp in self.report.get_failed_experiments():
                self.logger.info(f"  - {exp['game_name']}, {exp['model_name']}: {exp.get('error', 'Unknown error')[:100]}...")

def main():
    """Run the experiment series from command line"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run an experiment series with multiple games and models')
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config file')
    parser.add_argument('--name', type=str, default=None, help='Custom name for the experiment series')
    parser.add_argument('--resume', action='store_true', help='Resume interrupted experiment series')
    
    args = parser.parse_args()
    
    runner = ExperimentSeriesRunner(args.config, args.name, args.resume)
    runner.run_experiment_series()

if __name__ == "__main__":
    main() 