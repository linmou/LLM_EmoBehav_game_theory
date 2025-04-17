# Neuro Manipulation Framework Documentation

## Overview

This repository contains tools and experiments for neural representation engineering and game theory experiments.

## Modules

- [Experiment Configuration](experiment_configuration.md)
- [Game Scenarios](game_scenarios.md)
- [GPU Optimization](gpu_optimization.md) - Tools for optimizing batch size and GPU utilization
- [Statistical Analysis](statistical_analysis.md)

## Running Experiments

### Basic Usage

To run a game theory experiment:

```bash
python -m neuro_manipulation.game_theory_exp_0205
```

The experiment will use the default configuration path (`config/escalGame_repEng_experiment_config.yaml`). You can modify this path in the script if needed.

### Optimizing GPU Utilization

For maximum performance, add these settings to your config file:

```yaml
experiment:
  # ... other settings ...
  
  auto_batch_size: true  # Automatically find optimal batch size
  batch_size_safety_margin: 0.9  # Percentage of GPU memory to target (0.9 = 90%)
```

This will:
1. Start with a small batch size
2. Incrementally test larger batch sizes
3. Find the maximum size that fits in GPU memory
4. Run your experiment with the optimal batch size

See [GPU Optimization](gpu_optimization.md) for more details.

### Quick Sanity Checks

To run a small validation experiment before committing to a full run, add this to your config:

```yaml
experiment:
  # ... other settings ...
  run_sanity_check: true
```