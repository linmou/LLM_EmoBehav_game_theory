#!/usr/bin/env python3
"""
Runner script for emotion memory experiments.
Loads configuration from YAML files and runs complete emotion memory experiments.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuro_manipulation.repe import repe_pipeline_registry 
from emotion_memory_experiments.dataset_factory import (
    create_dataset_from_config,
    create_vllm_config_from_dict,
    create_benchmark_config_from_dict,
    create_experiment_config_from_dict,
)
from emotion_memory_experiments.data_models import (
    BenchmarkConfig,
    BenchmarkItem,
    ExperimentConfig,
)
from emotion_memory_experiments.experiment import EmotionMemoryExperiment


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load experiment configuration from YAML file"""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"✅ Loaded configuration from: {config_path}")
    return config


def create_experiment_config(config_dict: Dict[str, Any]) -> ExperimentConfig:
    """Convert YAML config to ExperimentConfig object"""

    # Extract single benchmark configuration (use first benchmark if multiple provided)
    benchmarks = config_dict.get("benchmarks", {})
    if not benchmarks:
        raise ValueError("No benchmarks specified in configuration")

    # Take the first benchmark and create configs using factory functions
    benchmark_name, benchmark_data = next(iter(benchmarks.items()))
    model_path = config_dict["model"]["model_path"]
    
    # Use factory functions for consistent config creation
    benchmark_config = create_benchmark_config_from_dict(benchmark_data, config_dict)
    loading_config = create_vllm_config_from_dict(config_dict, model_path)

    # Create main experiment config
    exp_config = ExperimentConfig(
        model_path=model_path,
        emotions=config_dict["emotions"]["target_emotions"],
        benchmark=benchmark_config,
        output_dir=config_dict["output"]["results_dir"],
        # Optional parameters with defaults
        intensities=config_dict["emotions"].get("intensities", [1.0]),
        batch_size=config_dict["execution"].get("batch_size", 4),
        generation_config=config_dict.get("generation", {}),
        loading_config=loading_config,  # Add loading config
        repe_eng_config=config_dict.get("repe_eng_config", {}),
        max_evaluation_workers=config_dict["execution"].get(
            "max_evaluation_workers", 4
        ),
        pipeline_queue_size=config_dict["execution"].get("batch_size", 4) * 2,
    )

    return exp_config


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration"""
    log_level = config.get("output", {}).get("log_level", "INFO")
    log_file = config.get("output", {}).get("log_file")

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(file_handler)
        print(f"📝 Logging to file: {log_path}")

    return logging.getLogger(__name__)


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration before running experiment"""
    print("🔍 Validating configuration...")

    errors = []

    # Check required sections
    required_sections = ["model", "emotions", "benchmarks", "execution", "output"]
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")

    # Check model path
    model_path = Path(config.get("model", {}).get("path", ""))
    if not model_path.exists() and not str(model_path).startswith("/mock"):
        errors.append(f"Model path does not exist: {model_path}")

    # Check benchmark data files
    for benchmark_name, benchmark_config in config.get("benchmarks", {}).items():
        data_path = Path(benchmark_config.get("data_path", ""))
        if not data_path.exists():
            errors.append(
                f"Benchmark data file not found: {data_path} (for {benchmark_name})"
            )

    # Check output directory is writable
    output_dir = Path(config.get("output", {}).get("results_dir", "results"))
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        errors.append(f"Cannot create output directory {output_dir}: {e}")

    if errors:
        print("❌ Configuration validation failed:")
        for error in errors:
            print(f"  - {error}")
        return False

    print("✅ Configuration validation passed")
    return True


def run_experiment(
    config_path: Path, dry_run: bool = False, debug: bool = False
) -> bool:
    """Run emotion memory experiment from configuration file"""

    try:
        # Load and validate configuration
        config_dict = load_config(config_path)

        if not validate_config(config_dict):
            return False

        # Setup logging
        logger = setup_logging(config_dict)

        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")

        # Create experiment configuration
        exp_config = create_experiment_config(config_dict)

        print(f"\n🚀 EMOTION MEMORY EXPERIMENT")
        print(f"Model: {exp_config.model_path}")
        print(f"Emotions: {exp_config.emotions}")
        print(
            f"Benchmark: {exp_config.benchmark.name} ({exp_config.benchmark.task_type})"
        )
        print(f"Output: {exp_config.output_dir}")
        print("=" * 60)

        if dry_run:
            print("🔍 DRY RUN - Configuration validated successfully")
            print("Would run experiment with:")

            # Show what would be tested
            try:
                from .dataset_factory import create_dataset_from_config
                dataset = create_dataset_from_config(exp_config.benchmark)
                items = len(dataset)
                conditions = (
                    len(exp_config.emotions) * len(exp_config.intensities) * items
                )

                print(
                    f"  📊 {exp_config.benchmark.name}: {items} items × {len(exp_config.emotions)} emotions × {len(exp_config.intensities)} intensities = {conditions} conditions"
                )
                print(f"\n📈 Total experimental conditions: {conditions}")
            except Exception as e:
                print(f"  ❌ {exp_config.benchmark.name}: Error loading - {e}")

            print(f"Dry run complete!")

            return True

        repe_pipeline_registry()
        # Create and run experiment
        experiment = EmotionMemoryExperiment(exp_config)
        logger.info(f"Starting emotion memory experiment")

        # Run the experiment
        results_df = experiment.run_experiment()

        if results_df is not None and len(results_df) > 0:
            print(f"\n✅ Experiment completed successfully!")
            print(f"📊 Generated {len(results_df)} results")
            print(f"📁 Results saved to: {exp_config.output_dir}")

            # Show summary statistics
            print(f"\n📈 Results Summary:")
            print(f"  Emotions tested: {results_df['emotion'].nunique()}")
            print(f"  Benchmarks tested: {results_df['benchmark'].nunique()}")
            print(f"  Average score: {results_df['score'].mean():.3f}")
            print(
                f"  Score range: {results_df['score'].min():.3f} - {results_df['score'].max():.3f}"
            )

            return True
        else:
            print("❌ Experiment failed - no results generated")
            return False

    except KeyboardInterrupt:
        print("\n⚠️ Experiment interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Experiment failed with error: {e}")
        if debug:
            import traceback

            traceback.print_exc()
        return False


def create_sample_config(output_path: Path):
    """Create a sample configuration file based on the working test config"""

    # Load the working test configuration as template
    template_path = (
        Path(__file__).parent.parent / "config" / "test_downloaded_data.yaml"
    )

    try:
        with open(template_path, "r") as f:
            sample_config = yaml.safe_load(f)
    except FileNotFoundError:
        # Fallback to minimal config if template not found
        sample_config = {
            "experiment_name": "sample_emotion_memory_experiment",
            "description": "Sample configuration for emotion memory experiments",
            "version": "1.0.0",
            "model": {
                "model_path": "/path/to/your/model/Qwen2.5-0.5B-Instruct",
            },
            "emotions": {
                "target_emotions": ["anger", "happiness"],
                "include_neutral": True,
                "intensities": [1.0],
            },
            "benchmarks": {
                "infinitebench_passkey": {
                    "name": "infinitebench",
                    "task_type": "passkey",
                    "data_path": "test_data/real_benchmarks/infinitebench_passkey.jsonl",
                    "evaluation_method": "get_score_one_passkey",
                    "sample_limit": 5,
                }
            },
            "generation": {
                "temperature": 0.1,
                "max_new_tokens": 50,
                "do_sample": False,
                "top_p": 0.9,
                "repetition_penalty": 1.0,
            },
            "execution": {
                "batch_size": 4,
                "max_evaluation_workers": 2,
            },
            "output": {
                "results_dir": "results/emotion_memory",
                "save_intermediate": True,
                "formats": ["csv"],
                "log_level": "INFO",
            },
        }

    # Customize the template for sample config
    sample_config["experiment_name"] = "sample_emotion_memory_experiment"
    sample_config["description"] = "Sample configuration for emotion memory experiments"

    # Update model path to placeholder
    if "model" not in sample_config:
        sample_config["model"] = {}
    sample_config["model"]["model_path"] = "/path/to/your/model/Qwen2.5-0.5B-Instruct"

    # Update output directory
    sample_config["output"]["results_dir"] = "results/emotion_memory"

    with open(output_path, "w") as f:
        yaml.dump(sample_config, f, default_flow_style=False, indent=2)

    print(f"✅ Sample configuration created: {output_path}")
    print("📝 Based on working test configuration")
    print("📝 Edit the model path and data paths before running the experiment.")
    print(
        f"📝 Test with: python scripts/run_emotion_memory_experiment.py {output_path} --dry-run"
    )


def main():
    """Main entry point for the experiment runner"""
    parser = argparse.ArgumentParser(
        description="Run emotion memory experiments from configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run experiment from config
  python scripts/run_emotion_memory_experiment.py config/emotion_memory_test.yaml
  
  # Validate config without running
  python scripts/run_emotion_memory_experiment.py config/test.yaml --dry-run
  
  # Create sample config
  python scripts/run_emotion_memory_experiment.py --create-sample config/sample.yaml
  
  # Run with debug logging
  python scripts/run_emotion_memory_experiment.py config/test.yaml --debug
        """,
    )

    parser.add_argument(
        "config", nargs="?", type=Path, help="Path to YAML configuration file"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and show what would be run without executing",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    parser.add_argument(
        "--create-sample",
        type=Path,
        metavar="OUTPUT_PATH",
        help="Create a sample configuration file",
    )

    args = parser.parse_args()

    # Handle sample config creation
    if args.create_sample:
        create_sample_config(args.create_sample)
        return 0

    # Require config file for other operations
    if not args.config:
        parser.error("Config file required (or use --create-sample)")

    # Run experiment
    success = run_experiment(args.config, dry_run=args.dry_run, debug=args.debug)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
