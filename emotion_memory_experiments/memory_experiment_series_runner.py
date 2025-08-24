"""
Memory experiment series runner for running batches of emotion memory experiments.
Adapted from neuro_manipulation/experiment_series_runner.py for EmotionMemoryExperiment.
"""

import copy
import gc

# Use dynamic import to avoid relative import issues
import importlib.util
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from transformers import AutoConfig

from neuro_manipulation.repe.pipelines import repe_pipeline_registry

# Load data_models directly
_spec = importlib.util.spec_from_file_location(
    "data_models", os.path.join(os.path.dirname(__file__), "data_models.py")
)
if _spec is not None and _spec.loader is not None:
    _data_models = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_data_models)
else:
    raise ImportError("Cannot load data_models module")

BenchmarkConfig = _data_models.BenchmarkConfig
ExperimentConfig = _data_models.ExperimentConfig
LoadingConfig = _data_models.LoadingConfig


class ExperimentStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class MemoryExperimentReport:
    """Manages and persists the status of memory experiments in a series

    Adapted from ExperimentReport to track memory experiment combinations of
    benchmarks, models, and emotion configurations.
    """

    def __init__(self, base_dir: str, experiment_series_name: str):
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H")
        self.report_file = Path(
            f"{base_dir}/{experiment_series_name}_{self.timestamp}_memory_experiment_report.json"
        )
        self.report_file.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.experiments: Dict[str, Dict[str, Any]] = {}
        self.series_start_time = datetime.now()
        self._save_report()

    def add_experiment(
        self,
        benchmark_name: str,
        model_name: str,
        exp_id: str,
        status: str = ExperimentStatus.PENDING,
    ) -> None:
        """Add a new memory experiment to the report"""
        with self.lock:
            self.experiments[exp_id] = {
                "benchmark_name": benchmark_name,
                "model_name": model_name,
                "status": status,
                "start_time": None,
                "end_time": None,
                "time_cost_seconds": None,
                "error": None,
                "output_dir": None,
                "exp_id": exp_id,
            }
            self._save_report()

    def update_experiment(self, exp_id: str, **kwargs) -> None:
        """Update experiment status and details"""
        with self.lock:
            if exp_id in self.experiments:
                self.experiments[exp_id].update(kwargs)

                # Calculate time cost if we have both start and end times
                if (
                    "start_time" in self.experiments[exp_id]
                    and "end_time" in self.experiments[exp_id]
                ):
                    start = self.experiments[exp_id]["start_time"]
                    end = self.experiments[exp_id]["end_time"]
                    if (
                        start
                        and end
                        and not self.experiments[exp_id].get("time_cost_seconds")
                    ):
                        start_dt = datetime.fromisoformat(start)
                        end_dt = datetime.fromisoformat(end)
                        time_cost = (end_dt - start_dt).total_seconds()
                        self.experiments[exp_id]["time_cost_seconds"] = time_cost

                self._save_report()

    def get_pending_experiments(self) -> List[Dict[str, Any]]:
        """Get list of pending experiments"""
        with self.lock:
            return [
                exp
                for exp in self.experiments.values()
                if exp["status"] == ExperimentStatus.PENDING
            ]

    def get_failed_experiments(self) -> List[Dict[str, Any]]:
        """Get list of failed experiments"""
        with self.lock:
            return [
                exp
                for exp in self.experiments.values()
                if exp["status"] == ExperimentStatus.FAILED
            ]

    def _save_report(self) -> None:
        """Save the report to disk"""
        with open(self.report_file, "w") as f:
            # Calculate series duration so far
            series_duration = (datetime.now() - self.series_start_time).total_seconds()

            json.dump(
                {
                    "last_updated": datetime.now().isoformat(),
                    "series_start_time": self.series_start_time.isoformat(),
                    "series_duration_seconds": series_duration,
                    "experiments": self.experiments,
                },
                f,
                indent=2,
            )

    def load_report(self) -> bool:
        """Load a report from disk if it exists"""
        if self.report_file.exists():
            with open(self.report_file, "r") as f:
                report_data = json.load(f)
                self.experiments = report_data.get("experiments", {})
            return True
        return False

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of experiment statuses"""
        with self.lock:
            # Calculate total and average time costs
            completed_exps = [
                exp
                for exp in self.experiments.values()
                if exp["status"] == ExperimentStatus.COMPLETED
                and exp.get("time_cost_seconds")
            ]

            total_time_cost = (
                sum(exp["time_cost_seconds"] for exp in completed_exps)
                if completed_exps
                else 0
            )
            avg_time_cost = (
                total_time_cost / len(completed_exps) if completed_exps else 0
            )

            # Calculate series duration so far
            series_duration = (datetime.now() - self.series_start_time).total_seconds()

            summary = {
                "total": len(self.experiments),
                "pending": sum(
                    1
                    for exp in self.experiments.values()
                    if exp["status"] == ExperimentStatus.PENDING
                ),
                "running": sum(
                    1
                    for exp in self.experiments.values()
                    if exp["status"] == ExperimentStatus.RUNNING
                ),
                "completed": sum(
                    1
                    for exp in self.experiments.values()
                    if exp["status"] == ExperimentStatus.COMPLETED
                ),
                "failed": sum(
                    1
                    for exp in self.experiments.values()
                    if exp["status"] == ExperimentStatus.FAILED
                ),
                "total_time_cost_seconds": total_time_cost,
                "avg_time_cost_seconds": avg_time_cost,
                "formatted_avg_time": str(timedelta(seconds=int(avg_time_cost))),
                "series_duration_seconds": series_duration,
                "formatted_series_duration": str(
                    timedelta(seconds=int(series_duration))
                ),
            }
            return summary


class MemoryExperimentSeriesRunner:
    """Manages running a series of memory experiments with different benchmark/model combinations

    Adapted from ExperimentSeriesRunner to work with EmotionMemoryExperiment.
    Supports:
    - Running multiple benchmark/model combinations in sequence
    - Graceful shutdown and resumption of experiment series
    - Model download and verification
    - CUDA memory cleanup between experiments
    """

    def __init__(
        self,
        config_path: str,
        series_name: Optional[str] = None,
        resume: bool = False,
        dry_run: bool = False,
    ):
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.config_path = config_path
        self.series_name = series_name or f"memory_experiment_series"
        self.dry_run = dry_run

        # Load and parse config
        self._load_config()

        # Initialize shutdown flag
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._handle_shutdown)

        # Create or load experiment report
        base_dir = self.base_config.get("output_dir", "results/memory_experiments")
        self.report = MemoryExperimentReport(base_dir, self.series_name)

        # Check if resuming
        self.resume = resume
        if resume:
            if not self.report.load_report():
                self.logger.warning(
                    "No previous experiment report found. Starting fresh."
                )
            else:
                self.logger.info(
                    f"Resumed experiment series. Status: {self.report.get_summary()}"
                )

    def _load_config(self) -> None:
        """Load configuration from YAML file"""
        with open(self.config_path, "r") as f:
            self.base_config = yaml.safe_load(f)

        # Validate required sections
        if "models" not in self.base_config:
            raise ValueError("Configuration must include 'models' section")
        if "benchmarks" not in self.base_config:
            raise ValueError("Configuration must include 'benchmarks' section")
        if "emotions" not in self.base_config:
            raise ValueError("Configuration must include 'emotions' section")
        if "intensities" not in self.base_config:
            raise ValueError("Configuration must include 'intensities' section")

    def _check_model_existence(self, model_name: str) -> bool:
        """
        Check if the model exists in either ~/.cache/huggingface/hub/ or ../huggingface.
        If not, download it to ../huggingface.

        Args:
            model_name: The name of the model to check

        Returns:
            bool: True if model exists or was successfully downloaded, False otherwise
        """
        # Skip for local paths (starting with /)
        if model_name.startswith("/"):
            self.logger.info(f"Skipping model check for local path: {model_name}")
            return True

        # Define paths to check
        home_dir = os.path.expanduser("~")
        cache_path = os.path.join(home_dir, ".cache", "huggingface", "hub")
        parent_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        alt_path = os.path.join(parent_dir, "huggingface_models")

        # Create model-specific paths based on the model name structure (org/model format)
        if "/" in model_name:
            model_parts = model_name.split("/")
            model_org = model_parts[0]
            model_name_part = "/".join(model_parts[1:])

            cache_model_path = os.path.join(
                cache_path,
                "models--" + model_org + "--" + model_name_part.replace("/", "--"),
            )
            alt_model_path = os.path.join(alt_path, model_org, model_name_part)
        else:
            # For models without organization prefix
            cache_model_path = os.path.join(cache_path, "models--" + model_name)
            alt_model_path = os.path.join(alt_path, model_name)

        self.logger.info(f"Checking if model {model_name} exists...")
        self.logger.info(f"Checking path: {cache_model_path}")
        self.logger.info(f"Checking alternative path: {alt_model_path}")

        # Check if model exists in either location
        if os.path.exists(cache_model_path) or os.path.exists(alt_model_path):
            self.logger.info(f"Model {model_name} found.")
            return True

        # If model doesn't exist, download it to ../huggingface_models
        self.logger.info(
            f"Model {model_name} not found. Downloading to {alt_model_path}..."
        )
        try:
            # Make sure the target directory exists
            os.makedirs(os.path.dirname(alt_model_path), exist_ok=True)

            # First verify the model exists on HuggingFace
            try:
                AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            except Exception as e:
                self.logger.error(
                    f"Model {model_name} not found on HuggingFace: {str(e)}"
                )
                return False

            # Download model using huggingface-cli command
            self.logger.info(
                f"Starting download of model {model_name} to {alt_model_path} using huggingface-cli..."
            )

            # Prepare environment with HF_HUB_ENABLE_HF_TRANSFER=1 for faster downloads
            env = os.environ.copy()
            env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

            # Run huggingface-cli download command
            cmd = [
                "huggingface-cli",
                "download",
                model_name,
                "--local-dir",
                alt_model_path,
            ]
            self.logger.info(f"Running command: {' '.join(cmd)}")

            # Execute the command and capture output
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True
            )

            # Stream output in real-time
            while True:
                if process.stdout is not None:
                    output = process.stdout.readline()
                    if output == "" and process.poll() is not None:
                        break
                    if output:
                        self.logger.info(output.strip())
                else:
                    break

            # Get return code and stderr
            return_code = process.poll()
            stderr = process.stderr.read() if process.stderr else ""

            if return_code != 0:
                self.logger.error(
                    f"Download failed with return code {return_code}: {stderr}"
                )
                return False

            self.logger.info(
                f"Model {model_name} successfully downloaded to {alt_model_path}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to download model {model_name}: {str(e)}")
            return False

    def _handle_shutdown(self, sig, frame):
        """Handle SIGINT (Ctrl+C)"""
        if not self.shutdown_requested:
            self.logger.info(
                "Shutdown requested. Finishing current experiment and stopping..."
            )
            self.shutdown_requested = True
        else:
            self.logger.warning("Forced shutdown requested. Exiting immediately.")
            sys.exit(1)

    def _format_model_name_for_folder(self, model_name: str) -> str:
        """Format model name for folder name by removing path prefix"""
        # Count the number of forward slashes in the model name
        slash_count = model_name.count("/")

        # Special case for full paths with multiple parts
        if slash_count >= 2 and model_name.startswith("/"):
            # For paths like /data/home/huggingface_models/RWKV/v6-Finch-7B-HF
            # Extract the last two parts
            parts = model_name.split("/")
            if len(parts) >= 3:  # Make sure we have enough parts
                return f"{parts[-2]}/{parts[-1]}"

        # For HuggingFace model paths like meta-llama/Llama-3.1-8B-Instruct
        elif slash_count == 1:
            # Return as is
            return model_name
        elif slash_count == 2:
            # For paths with exactly three parts
            parts = model_name.split("/")
            return f"{parts[1]}/{parts[2]}"

        # For paths with more than three parts
        elif slash_count > 2:
            # Extract the last two parts
            parts = model_name.split("/")
            return f"{parts[-2]}/{parts[-1]}"

        return model_name

    def setup_experiment(self, benchmark_config: Dict, model_name: str):
        """Set up a single memory experiment with the given benchmark and model"""

        # Create BenchmarkConfig from dictionary
        benchmark = BenchmarkConfig(
            name=benchmark_config["name"],
            task_type=benchmark_config["task_type"],
            data_path=(
                Path(benchmark_config["data_path"])
                if "data_path" in benchmark_config
                else None
            ),
            sample_limit=benchmark_config.get("sample_limit"),
            augmentation_config=benchmark_config.get("augmentation_config"),
        )

        # Create LoadingConfig if specified
        loading_config = None
        if "loading_config" in self.base_config:
            loading_config = LoadingConfig(
                model_path=model_name, **self.base_config["loading_config"]
            )

        # Create ExperimentConfig
        experiment_config = ExperimentConfig(
            model_path=model_name,
            emotions=self.base_config["emotions"],
            intensities=self.base_config["intensities"],
            benchmark=benchmark,
            output_dir=self.base_config.get("output_dir", "results/memory_experiments"),
            batch_size=self.base_config.get("batch_size", 4),
            generation_config=self.base_config.get("generation_config"),
            loading_config=loading_config,
            repe_eng_config=self.base_config.get("repe_eng_config"),
            max_evaluation_workers=self.base_config.get("max_evaluation_workers", 2),
            pipeline_queue_size=self.base_config.get("pipeline_queue_size", 2),
        )

        # Only import and create experiment if not dry run
        if self.dry_run:
            # For dry run, return a mock object with the config
            class MockExperiment:
                def __init__(self, config):
                    self.config = config

            return MockExperiment(experiment_config)
        else:
            from .experiment import EmotionMemoryExperiment

            experiment = EmotionMemoryExperiment(experiment_config)
            return experiment

    def _clean_cuda_memory(self) -> None:
        """Clean up CUDA memory after an experiment

        This function attempts to free up CUDA memory by:
        1. Running Python's garbage collector
        2. Emptying CUDA cache if PyTorch is available
        3. Running a system command to check CUDA memory usage
        """
        try:
            # Run Python's garbage collector
            gc.collect()

            # Try to empty CUDA cache if PyTorch is available
            try:
                import torch

                if torch.cuda.is_available():
                    self.logger.info("Clearing CUDA cache...")
                    torch.cuda.empty_cache()
                    # Get and log memory stats
                    allocated = torch.cuda.memory_allocated() / (1024**3)
                    max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
                    reserved = torch.cuda.memory_reserved() / (1024**3)
                    max_reserved = torch.cuda.max_memory_reserved() / (1024**3)
                    self.logger.info(
                        f"CUDA memory stats after cleanup: allocated={allocated:.2f}GB, "
                        f"max_allocated={max_allocated:.2f}GB, reserved={reserved:.2f}GB, "
                        f"max_reserved={max_reserved:.2f}GB"
                    )
            except ImportError:
                self.logger.info("PyTorch not available for CUDA memory cleanup")

            # Try to run nvidia-smi to check memory usage
            try:
                self.logger.info("Running nvidia-smi to check GPU memory...")
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,memory.free",
                        "--format=csv",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.logger.info(f"NVIDIA-SMI report:\n{result.stdout}")
            except (subprocess.SubprocessError, FileNotFoundError) as e:
                self.logger.info(f"Could not run nvidia-smi: {str(e)}")

        except Exception as e:
            self.logger.warning(f"Error during CUDA memory cleanup: {str(e)}")

    def run_single_experiment(
        self, benchmark_config: Dict, model_name: str, exp_id: str
    ) -> bool:
        """Run a single memory experiment with the specified benchmark and model"""
        self.logger.info(
            f"Starting memory experiment with benchmark: {benchmark_config['name']}, model: {model_name}"
        )

        # Start timing
        start_time = datetime.now()

        # Update experiment status
        self.report.update_experiment(
            exp_id, status=ExperimentStatus.RUNNING, start_time=start_time.isoformat()
        )

        try:
            experiment = self.setup_experiment(benchmark_config, model_name)

            # Record output directory for later reference
            output_dir = str(experiment.output_dir)
            self.report.update_experiment(exp_id, output_dir=output_dir)

            # Run the experiment
            if self.base_config.get("run_sanity_check", False):
                experiment.run_sanity_check()
            else:
                experiment.run_experiment()

            # End timing
            end_time = datetime.now()
            time_cost = (end_time - start_time).total_seconds()

            # Format time cost for logging
            time_cost_str = str(timedelta(seconds=int(time_cost)))

            # Update experiment status
            self.report.update_experiment(
                exp_id,
                status=ExperimentStatus.COMPLETED,
                end_time=end_time.isoformat(),
                time_cost_seconds=time_cost,
            )

            self.logger.info(
                f"Memory experiment completed: {benchmark_config['name']}, {model_name}, Time cost: {time_cost_str}"
            )

            # Clean up CUDA memory after experiment
            self.logger.info("Cleaning up CUDA memory...")
            self._clean_cuda_memory()

            return True

        except Exception as e:
            # End timing even if experiment failed
            end_time = datetime.now()
            time_cost = (end_time - start_time).total_seconds()

            error_trace = traceback.format_exc()
            self.logger.error(
                f"Memory experiment failed: {benchmark_config['name']}, {model_name}\nError: {str(e)}\n{error_trace}"
            )

            # Update experiment status
            self.report.update_experiment(
                exp_id,
                status=ExperimentStatus.FAILED,
                end_time=end_time.isoformat(),
                time_cost_seconds=time_cost,
                error=f"{str(e)}\n{error_trace}",
            )

            # Try to clean up CUDA memory even after failure
            self.logger.info(
                "Attempting to clean up CUDA memory after failed experiment..."
            )
            self._clean_cuda_memory()

            return False

    def expand_benchmark_configs(self, benchmarks: List[Dict]) -> List[Dict]:
        """
        Expand benchmark configurations that have task_type='all' into individual configs.

        Args:
            benchmarks: List of benchmark configuration dictionaries

        Returns:
            Expanded list with task_type='all' configs replaced by individual task configs
        """
        expanded_benchmarks = []

        for benchmark_config in benchmarks:
            task_type = benchmark_config.get("task_type", "")

            # Check if task_type is a pattern (contains wildcards or regex characters)
            if self._is_pattern_task_type(task_type):
                # Create a temporary BenchmarkConfig to discover datasets
                temp_benchmark = BenchmarkConfig(
                    name=benchmark_config["name"], task_type=task_type
                )

                # Discover task types matching the pattern
                base_data_dir = self.base_config.get(
                    "base_data_dir", "data/memory_benchmarks"
                )
                discovered_tasks = temp_benchmark.discover_datasets_by_pattern(
                    base_data_dir
                )

                if not discovered_tasks:
                    self.logger.warning(
                        f"No datasets found for benchmark '{benchmark_config['name']}' "
                        f"in directory '{base_data_dir}'. Skipping."
                    )
                    continue

                self.logger.info(
                    f"Discovered {len(discovered_tasks)} task types for benchmark '{benchmark_config['name']}'"
                )

                # Create individual configs for each discovered task
                for task_type in discovered_tasks:
                    expanded_config = copy.deepcopy(benchmark_config)
                    expanded_config["task_type"] = task_type
                    expanded_benchmarks.append(expanded_config)
            else:
                # Keep existing non-'all' configs as-is
                expanded_benchmarks.append(benchmark_config)

        return expanded_benchmarks

    def _is_pattern_task_type(self, task_type: str) -> bool:
        """
        Check if task_type is a pattern that needs expansion.

        Args:
            task_type: The task type string to check

        Returns:
            True if task_type contains pattern characters, False otherwise

        Examples:
            - "*" -> True (all files)
            - "narrative*" -> True (starts with narrative)
            - "*qa" -> True (ends with qa)
            - "pass.*" -> True (regex pattern)
            - "narrativeqa" -> False (literal task name)
        """
        if not task_type:
            return False

        # Check for common pattern characters
        pattern_chars = [
            "*",
            "?",
            "[",
            "]",
            "{",
            "}",
            "(",
            ")",
            "^",
            "$",
            "+",
            ".",
            "|",
            "\\",
        ]
        return any(char in task_type for char in pattern_chars)

    def dry_run_series(self) -> None:
        """Dry run to validate configuration without running experiments"""
        self.logger.info("🚀 Starting DRY RUN - Memory Experiment Series Validation")
        self.logger.info("=" * 60)

        # Get lists of benchmarks and models from config
        original_benchmarks = self.base_config["benchmarks"]
        models = self.base_config["models"]

        # Expand benchmarks with regex patterns
        benchmarks = self.expand_benchmark_configs(original_benchmarks)

        self.logger.info(f"📊 Original benchmarks: {len(original_benchmarks)}")
        self.logger.info(f"📈 Expanded benchmarks: {len(benchmarks)}")
        self.logger.info(f"🤖 Models: {len(models)}")
        self.logger.info(f"😊 Emotions: {len(self.base_config['emotions'])}")
        self.logger.info(f"📈 Intensities: {len(self.base_config['intensities'])}")

        # Calculate experiment combinations
        total_combinations = len(benchmarks) * len(models)
        total_with_emotions = (
            total_combinations
            * len(self.base_config["emotions"])
            * len(self.base_config["intensities"])
        )

        self.logger.info(f"🧮 Total experiment combinations: {total_combinations}")
        self.logger.info(
            f"🎯 Total runs with emotions/intensities: {total_with_emotions}"
        )

        # Test creating experiment configurations
        self.logger.info("\n🔬 Testing experiment configuration creation...")
        test_count = min(3, len(benchmarks), len(models))  # Test first 3 combinations

        for i, (benchmark_config, model_name) in enumerate(
            zip(benchmarks[:test_count], models[:test_count])
        ):
            try:
                experiment = self.setup_experiment(benchmark_config, model_name)
                self.logger.info(
                    f"   ✅ Config {i+1}: {benchmark_config['name']}_{benchmark_config['task_type']} + {model_name}"
                )
                self.logger.info(f"      📁 Output: {experiment.config.output_dir}")
                self.logger.info(
                    f"      🎯 Data path: {experiment.config.benchmark.get_data_path()}"
                )
            except Exception as e:
                self.logger.error(f"   ❌ Config {i+1} failed: {e}")

        self.logger.info("\n🎉 DRY RUN COMPLETED SUCCESSFULLY!")
        self.logger.info("✅ Configuration is valid and ready for execution")
        self.logger.info(
            f"✅ Would run {len(benchmarks)} benchmark(s) × {len(models)} model(s)"
        )

    def run_experiment_series(self) -> None:
        """Run the full series of memory experiments with all benchmark/model combinations"""
        # Check if this is a dry run
        if self.dry_run:
            self.dry_run_series()
            return

        # Record series start time
        series_start_time = datetime.now()

        # Get lists of benchmarks and models from config
        original_benchmarks = self.base_config["benchmarks"]
        models = self.base_config["models"]

        # Expand benchmarks with task_type='all'
        benchmarks = self.expand_benchmark_configs(original_benchmarks)

        self.logger.info(
            f"Starting memory experiment series with {len(benchmarks)} benchmarks "
            f"(expanded from {len(original_benchmarks)} original) and {len(models)} models"
        )

        # Pre-check and download models if needed
        self.logger.info("Checking model availability...")
        for model_name in models:
            # For model checking and downloading, use the original model name
            model_exists = self._check_model_existence(model_name)
            if not model_exists:
                self.logger.warning(
                    f"Model {model_name} could not be verified or downloaded. Experiments with this model may fail."
                )

        # Generate experiment IDs and initialize report
        for benchmark_config in benchmarks:
            benchmark_name = benchmark_config["name"]
            task_type = benchmark_config["task_type"]
            for model_name in models:
                # For folder names, use the formatted version
                model_folder_name = self._format_model_name_for_folder(model_name)
                exp_id = f"{benchmark_name}_{task_type}_{model_folder_name.replace('/', '_')}"

                # Only add if not resuming or not already in report
                if not self.resume or exp_id not in self.report.experiments:
                    self.report.add_experiment(
                        f"{benchmark_name}_{task_type}", model_name, exp_id
                    )

        # Get pending experiments
        pending_experiments = self.report.get_pending_experiments()
        total_experiments = len(pending_experiments)
        self.logger.info(f"Total pending experiments: {total_experiments}")

        # Run each experiment
        for i, exp in enumerate(pending_experiments):
            if self.shutdown_requested:
                self.logger.info("Shutdown requested. Stopping experiment series.")
                break

            self.logger.info(
                f"Running experiment {i+1}/{total_experiments}: {exp['benchmark_name']}, {exp['model_name']}"
            )

            # Verify model exists before running the experiment
            model_name = exp["model_name"]
            if not self._check_model_existence(model_name):
                self.logger.error(
                    f"Skipping experiment with {exp['benchmark_name']} and {model_name} due to missing model."
                )
                self.report.update_experiment(
                    exp["exp_id"],
                    status=ExperimentStatus.FAILED,
                    error=f"Model {model_name} could not be found or downloaded.",
                )
                continue

            # Find the benchmark config
            # exp["benchmark_name"] is now in format "benchmark_tasktype"
            benchmark_config: Optional[Dict[str, Any]] = None
            for bench in benchmarks:
                bench_identifier = f"{bench['name']}_{bench['task_type']}"
                if bench_identifier == exp["benchmark_name"]:
                    benchmark_config = bench
                    break

            if not benchmark_config:
                self.logger.error(
                    f"Benchmark config not found for {exp['benchmark_name']}"
                )
                continue

            self.run_single_experiment(
                benchmark_config, exp["model_name"], exp["exp_id"]
            )

            # Print summary after each experiment
            summary = self.report.get_summary()
            self.logger.info(f"Memory experiment series progress: {summary}")

        # Final summary
        summary = self.report.get_summary()

        # Calculate and log total series time
        series_end_time = datetime.now()
        series_time_cost = (series_end_time - series_start_time).total_seconds()
        series_time_str = str(timedelta(seconds=int(series_time_cost)))

        self.logger.info(f"Memory experiment series completed. Final status: {summary}")
        self.logger.info(f"Total series time: {series_time_str}")
        self.logger.info(f"Average experiment time: {summary['formatted_avg_time']}")

        if summary["failed"] > 0:
            self.logger.info("Failed experiments:")
            for exp in self.report.get_failed_experiments():
                # Get time cost for failed experiment if available
                time_cost = exp.get("time_cost_seconds")
                time_info = (
                    f", time: {str(timedelta(seconds=int(time_cost)))}"
                    if time_cost
                    else ""
                )

                self.logger.info(
                    f"  - {exp['benchmark_name']}, {exp['model_name']}{time_info}: {exp.get('error', 'Unknown error')[:100]}..."
                )


def main():
    """Run the memory experiment series from command line"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a memory experiment series with multiple benchmarks and models"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to experiment config file"
    )
    parser.add_argument(
        "--name", type=str, default=None, help="Custom name for the experiment series"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume interrupted experiment series"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running experiments",
    )

    args = parser.parse_args()

    repe_pipeline_registry()
    runner = MemoryExperimentSeriesRunner(
        args.config, args.name, args.resume, args.dry_run
    )
    runner.run_experiment_series()


if __name__ == "__main__":
    main()
