# Option Probability Experiment Configuration Guide

## Overview

The Option Probability Experiment requires three main configuration components that work together to define the experimental setup. This guide explains all configuration options and provides examples.

## Configuration Files

### 1. Main Configuration File

**Location**: `config/option_probability_experiment_config.yaml`

This is the master configuration file that combines all settings:

```yaml
# Experiment settings
experiment:
  name: "emotion_context_option_probability"
  emotions: ["neutral", "angry"]
  emotion_intensities: [0.0, 1.0]
  context_conditions: ["with_description", "without_description"]
  sample_num: 100
  batch_size: 4
  output:
    base_dir: "experiments/option_probability_results"

# Model and representation settings  
repe_config:
  model_name_or_path: "meta-llama/Llama-3.1-8B-Instruct"
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.85

# Game settings
game_config:
  game_name: "prisoners_dilemma"
  data_path: "path/to/scenarios.json"
  decision_class: "binary_choice"
```

### 2. Quick Test Configuration

**Location**: `config/option_probability_quick_test.yaml`

Minimal setup for fast testing:

```yaml
experiment:
  name: "quick_test"
  emotions: ["neutral", "angry"]
  sample_num: 5  # Small sample for testing
  batch_size: 2

repe_config:
  model_name_or_path: "/path/to/small/model"
  gpu_memory_utilization: 0.6
```

## Configuration Sections

### Experiment Configuration

Controls the experimental design and execution:

```yaml
experiment:
  # Basic experiment info
  name: "experiment_name"
  description: "Experiment description"
  
  # Experimental factors
  emotions:
    - "neutral"      # Baseline condition
    - "angry"        # Target emotion
    - "happy"        # Additional emotions (optional)
    - "sad"
  
  emotion_intensities:
    - 0.0           # Neutral intensity  
    - 1.0           # Standard intensity
    - 1.5           # High intensity (optional)
  
  context_conditions:
    - "with_description"     # Full context provided
    - "without_description"  # Minimal context
  
  # Sampling settings
  sample_num: 100           # Scenarios per condition (null = all)
  batch_size: 4             # Processing batch size
  
  # Output settings
  output:
    base_dir: "experiments/results"
    save_raw_results: true
    save_statistical_analysis: true
    save_summary_report: true
  
  # Statistical analysis options
  statistical_analysis:
    enable_interaction_tests: true
    enable_pairwise_comparisons: true
    significance_level: 0.05
    multiple_comparison_correction: "bonferroni"
```

### Representation Engineering Configuration

Controls the neural representation aspects:

```yaml
repe_config:
  # Model settings
  model_name_or_path: "meta-llama/Llama-3.1-8B-Instruct"
  
  # vLLM engine settings
  tensor_parallel_size: 1      # Number of GPUs for tensor parallelism
  gpu_memory_utilization: 0.85 # GPU memory usage fraction
  max_num_seqs: 16            # Maximum concurrent sequences
  enforce_eager: true          # Use eager execution
  trust_remote_code: true      # Allow custom model code
  
  # Representation control (for future extensions)
  control_method: "reading_vec"
  block_name: "model.layers.{}.self_attn"
  coeffs: [0.0, 1.0]
  control_layers: "middle_third"  # Layer selection strategy
```

### Game Configuration

Defines the game-theoretic scenario:

```yaml
game_config:
  # Game identification
  game_name: "prisoners_dilemma"
  data_path: "path/to/scenario/data.json"
  
  # Decision format
  decision_class: "binary_choice"    # or "multiple_choice", "text_generation"
  
  # Prompt settings
  include_payoff_matrix: true
  include_context_description: true
  
  # Game-specific parameters
  payoff_matrix:
    both_cooperate: [3, 3]
    cooperate_defect: [0, 5] 
    defect_cooperate: [5, 0]
    both_defect: [1, 1]
  
  options: ["Cooperate", "Defect"]
  description: "Classic Prisoner's Dilemma"
```

### Logging Configuration

Controls logging behavior:

```yaml
logging:
  level: "INFO"                    # DEBUG, INFO, WARNING, ERROR
  log_to_file: true
  log_file_prefix: "experiment"
  include_timestamps: true
  include_model_outputs: false     # Set true for debugging
```

## Configuration Options Reference

### Emotion Options

| Emotion | Description | Typical Intensity |
|---------|-------------|-------------------|
| `neutral` | Baseline emotional state | 0.0 |
| `angry` | Anger/frustration | 1.0 - 1.5 |
| `happy` | Joy/satisfaction | 1.0 - 1.5 |
| `sad` | Sadness/disappointment | 1.0 - 1.5 |
| `fear` | Fear/anxiety | 1.0 - 1.5 |

### Context Conditions

| Condition | Description |
|-----------|-------------|
| `with_description` | Full scenario context provided |
| `without_description` | Minimal context, focus on choice |

### Model Options

| Model Path | Description | Use Case |
|------------|-------------|----------|
| `meta-llama/Llama-3.1-8B-Instruct` | Large model, high quality | Production experiments |
| `Qwen/Qwen2.5-0.5B-Instruct` | Small model, fast | Quick testing |
| `/path/to/local/model` | Custom local model | Specialized experiments |

### Game Types

| Game | Data Path | Description |
|------|-----------|-------------|
| `prisoners_dilemma` | `Prisoners_Dilemma_*.json` | Classic cooperation dilemma |
| `stag_hunt` | `Stag_Hunt_*.json` | Coordination game |
| `trust_game` | `Trust_Game_*.json` | Trust and reciprocity |
| `ultimatum_game` | `Ultimatum_Game_*.json` | Fairness and rejection |

## Usage Examples

### 1. Using YAML Configuration

```python
import yaml
from neuro_manipulation.experiments.option_probability_experiment import OptionProbabilityExperiment

# Load config
with open('config/option_probability_experiment_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Extract configurations
repe_eng_config = config['repe_config']
exp_config = {'experiment': config['experiment']}
game_config = config['game_config']

# Run experiment
experiment = OptionProbabilityExperiment(
    repe_eng_config=repe_eng_config,
    exp_config=exp_config,
    game_config=game_config
)
results = experiment.run_experiment()
```

### 2. Using Python Dictionaries

```python
# Define configs directly in code
repe_eng_config = {
    'model_name_or_path': 'meta-llama/Llama-3.1-8B-Instruct',
    'tensor_parallel_size': 1,
    'gpu_memory_utilization': 0.85
}

exp_config = {
    'experiment': {
        'name': 'test_experiment',
        'emotions': ['neutral', 'angry'],
        'emotion_intensities': [0.0, 1.0],
        'output': {'base_dir': 'experiments/results'}
    }
}

game_config = get_game_config(GameNames.PRISONERS_DILEMMA)

# Run experiment
experiment = OptionProbabilityExperiment(
    repe_eng_config=repe_eng_config,
    exp_config=exp_config,
    game_config=game_config,
    batch_size=4,
    sample_num=50
)
```

### 3. Custom Game Configuration

```python
custom_game_config = {
    'game_name': 'custom_game',
    'data_path': 'path/to/custom/scenarios.json',
    'decision_class': 'multiple_choice',
    'options': ['Option A', 'Option B', 'Option C'],
    'payoff_matrix': {
        'option_a_payoff': [10, 5, 0],
        'option_b_payoff': [5, 10, 5], 
        'option_c_payoff': [0, 5, 10]
    }
}
```

## Best Practices

### 1. Performance Optimization

- **Small Models for Testing**: Use smaller models (0.5B-1B parameters) for initial testing
- **Batch Size**: Start with small batch sizes (2-4) and increase based on GPU memory
- **Sample Size**: Use small samples (5-10) for quick validation

### 2. Reproducibility

- **Fixed Seeds**: Set random seeds in your experiment runner
- **Version Control**: Track configuration files with git
- **Documentation**: Include experiment descriptions in config files

### 3. Resource Management

- **GPU Memory**: Monitor `gpu_memory_utilization` setting
- **Tensor Parallel**: Use multiple GPUs only for large models
- **Output Storage**: Ensure sufficient disk space for results

### 4. Debugging

- **Logging Level**: Set to "DEBUG" when troubleshooting
- **Small Samples**: Use `sample_num: 2` for quick debugging
- **Model Outputs**: Enable `include_model_outputs: true` for detailed inspection

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `batch_size` or `gpu_memory_utilization`
2. **Model Not Found**: Check `model_name_or_path` and model availability
3. **Data Path Error**: Verify `data_path` points to valid scenario file
4. **Permission Error**: Ensure write access to output directory

### Validation

Before running large experiments:

1. Test with quick config (`sample_num: 2`)
2. Verify all output files are generated
3. Check statistical analysis results
4. Review log files for warnings

## File Locations

- **Main Config**: `config/option_probability_experiment_config.yaml`
- **Quick Test**: `config/option_probability_quick_test.yaml`
- **Example Runner**: `examples/run_option_probability_experiment.py`
- **Code Example**: `examples/option_probability_code_example.py`
- **Documentation**: `docs/code_readme/neuro_manipulation/experiments/` 