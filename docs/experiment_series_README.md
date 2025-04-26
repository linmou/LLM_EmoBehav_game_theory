# Experiment Series Pipeline

This document explains how to use the experiment series pipeline for running multiple experiments with different combinations of games and models.

## Overview

The experiment series pipeline allows you to run a permutation and combination of games and models in a single command. The pipeline has the following features:

1. Creates a new result base folder for the series of experiments.
2. Maintains the same structure for individual experiment results as the current system.
3. Handles failures in sub-experiments gracefully, continuing with the next experiment.
4. Generates a completeness report that is updated in real-time.
5. Supports graceful shutdown (Ctrl+C) and experiment resumption.

## Configuration

Create a configuration file similar to `config/experiment_series_config.yaml`. The key sections to configure are:

```yaml
experiment:
  name: "Multi_Game_Model_Experiment_Series"
  
  # List of games to run experiments with
  games:
    - "Prisoners_Dilemma"
    - "Escalation_Game"
    - "Ultimatum_Game"
  
  # List of models to run experiments with
  models:
    - "meta-llama/Llama-3.1-8B-Instruct"
    - "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
  # Rest of the configuration...
```

The pipeline will run experiments for all combinations of games and models specified.

## Running the Pipeline

To run the experiment series:

```bash
# First activate the conda environment
conda activate llm

# Run the experiment series
python -m neuro_manipulation.run_experiment_series --config config/experiment_series_config.yaml
```

Optional arguments:
- `--name`: Custom name for the experiment series
- `--resume`: Resume a previously interrupted experiment series

## Output Structure

The results will be organized in the following structure:

```
results/
└── ExperimentSeries/
    └── Multi_Game_Model_Experiment_Series_20250210_163355/
        ├── experiment_report.json
        ├── Prisoners_Dilemma_Llama-3.1-8B-Instruct_20250210_163355/
        │   ├── exp_config.yaml
        │   ├── exp_results.csv
        │   ├── exp_results.json
        │   └── stats_analysis.json
        ├── Escalation_Game_Llama-3.1-8B-Instruct_20250210_163455/
        │   └── ...
        └── ...
```

## Experiment Report

The `experiment_report.json` file contains the status of all experiments in the series:

```json
{
  "last_updated": "2025-02-10T16:34:55",
  "experiments": {
    "Prisoners_Dilemma_Llama-3.1-8B-Instruct": {
      "game_name": "Prisoners_Dilemma",
      "model_name": "meta-llama/Llama-3.1-8B-Instruct",
      "status": "completed",
      "start_time": "2025-02-10T16:33:55",
      "end_time": "2025-02-10T16:34:55",
      "error": null,
      "output_dir": "results/ExperimentSeries/Prisoners_Dilemma_Llama-3.1-8B-Instruct_20250210_163355",
      "exp_id": "Prisoners_Dilemma_Llama-3.1-8B-Instruct"
    },
    "Escalation_Game_Llama-3.1-8B-Instruct": {
      "status": "running",
      "...": "..."
    }
  }
}
```

## Resuming Experiments

If an experiment series is interrupted (either by an error or by user intervention with Ctrl+C), you can resume it:

```bash
python -m neuro_manipulation.run_experiment_series --config config/experiment_series_config.yaml --resume
```

This will skip completed experiments and retry failed or pending ones. 