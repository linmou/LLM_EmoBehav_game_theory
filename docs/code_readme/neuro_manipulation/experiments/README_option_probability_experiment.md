# Option Probability Experiment

## Overview

The `OptionProbabilityExperiment` class implements a 2x2 factorial experiment design to measure how different emotional states and contextual factors influence decision-making probabilities in game-theoretic scenarios.

## Experimental Design

### Factors
- **Emotion Factor**: neutral vs angry
- **Context Factor**: with_description vs without_description

### Combinations
The experiment tests all 4 combinations:
1. neutral + with_description
2. neutral + without_description  
3. angry + with_description
4. angry + without_description

## Implementation Details

### CombinedVLLMHook Integration

The experiment uses `CombinedVLLMHook` for sequence probability measurement:

```python
self.sequence_prob_hook = CombinedVLLMHook(
    model=self.model,
    tokenizer=self.tokenizer,
    tensor_parallel_size=self.repe_eng_config.get('tensor_parallel_size', 1),
    enable_sequence_prob=True,
    enable_rep_control=False,
    enable_layer_logit_recording=False
)
```

### Probability Measurement Process

1. **Option Extraction**: Extract option texts from formatted choices (removing "Option 1. " prefixes)
2. **Probability Calculation**: Use `CombinedVLLMHook.get_log_prob()` to measure sequence probabilities
3. **Normalization**: Normalize probabilities to sum to 1.0 across all options
4. **Storage**: Store both raw log probabilities and normalized probabilities

### Data Format

The experiment returns results in the following format:

```python
{
    'condition_emotion': 'angry',
    'condition_context': 'with_description', 
    'emotion_intensity': 1.0,
    'include_description': True,
    'prompt': '...',
    'scenario': '...',
    'behavior_choices': '...',
    'option_probabilities': {'Cooperate': 0.6, 'Defect': 0.4},
    'log_probabilities': {'Cooperate': -0.5, 'Defect': -0.9},
    'options': ['Option 1. Cooperate', 'Option 2. Defect'],
    'batch_idx': 0
}
```

## Statistical Analysis

The experiment automatically performs:

- **Summary Statistics**: Mean, std, count by condition and option
- **Interaction Effects**: Chi-square tests for emotion × context interactions
- **Pairwise Comparisons**: Mann-Whitney U tests between conditions

## Usage Example

```python
from neuro_manipulation.experiments.option_probability_experiment import OptionProbabilityExperiment

# Setup configurations
repe_eng_config = {
    'model_name_or_path': 'meta-llama/Llama-3.1-8B-Instruct',
    'tensor_parallel_size': 1,
    'coeffs': [0.0, 1.0],
    'block_name': 'model.layers.{}.self_attn',
    'control_method': 'reading_vec'
}

exp_config = {
    'experiment': {
        'name': 'emotion_context_probability_test',
        'emotions': ['neutral', 'angry'],
        'emotion_intensities': [0.0, 1.0],
        'output': {'base_dir': 'experiments/option_probability'}
    }
}

# Run experiment
experiment = OptionProbabilityExperiment(
    repe_eng_config=repe_eng_config,
    exp_config=exp_config,
    game_config=game_config,
    batch_size=4,
    sample_num=100
)

results_file = experiment.run_experiment()
```

## Output Files

- `option_probability_results.json`: Raw experimental results
- `option_probability_results.csv`: Structured data for analysis
- `statistical_analysis.json`: Detailed statistical test results
- `experiment_summary_report.txt`: Human-readable summary

## Key Features

1. **Modular Design**: Easy to extend with new emotions or context conditions
2. **Robust Probability Measurement**: Uses vLLM's sequence probability capabilities
3. **Comprehensive Analysis**: Automatic statistical testing and reporting
4. **Error Handling**: Graceful handling of probability calculation failures
5. **Reproducible**: Structured configuration and logging

## Core Components

### 1. ContextManipulationDataset

A specialized dataset class that extends `GameScenarioDataset` to support controlled context manipulation experiments.

**Key Features:**
- Conditional inclusion/exclusion of scenario descriptions
- Support for 2x2 factorial design studies
- Maintains compatibility with existing game scenario infrastructure
- Custom collate function for batch processing

**Usage Example:**
```python
from neuro_manipulation.datasets.context_manipulation_dataset import ContextManipulationDataset

# Create dataset with descriptions
dataset_with_context = ContextManipulationDataset(
    game_config=game_config,
    prompt_wrapper=prompt_wrapper,
    include_description=True,
    sample_num=100
)

# Create dataset without descriptions
dataset_without_context = ContextManipulationDataset(
    game_config=game_config,
    prompt_wrapper=prompt_wrapper,
    include_description=False,
    sample_num=100
)
```

### 2. OptionProbabilityExperiment

The main experiment class that orchestrates the complete 2x2 factorial study measuring behavioral choice probabilities across all conditions.

**Key Features:**
- Integrates SequenceProbVLLMHook for accurate probability measurement
- Implements 2x2 factorial design (Emotion × Context)
- Automated statistical analysis with interaction effects
- Comprehensive result storage and reporting

**Usage Example:**
```python
from neuro_manipulation.experiments.option_probability_experiment import OptionProbabilityExperiment

experiment = OptionProbabilityExperiment(
    repe_eng_config=repe_config,
    exp_config=experiment_config,
    game_config=game_config,
    batch_size=4,
    sample_num=200
)

results_file = experiment.run_experiment()
```

## Experimental Design

### 2x2 Factorial Structure

The experiment implements a complete 2×2 factorial design:

| Factor | Level 1 | Level 2 |
|--------|---------|---------|
| **Emotion** | Neutral | Angry |
| **Context** | Without Description | With Description |

This creates four experimental conditions:
1. **Neutral + Without Description**: Baseline condition
2. **Neutral + With Description**: Context effect only
3. **Angry + Without Description**: Emotion effect only  
4. **Angry + With Description**: Combined emotion and context effects

### Measured Variables

For each condition, the experiment measures:
- **Choice Probabilities**: Probability distribution across all available behavioral options
- **Log Probabilities**: Raw log probability values from the language model
- **Condition Metadata**: Emotion type, context presence, scenario details

## Statistical Analysis

The framework automatically performs comprehensive statistical analysis:

### Summary Statistics
- Mean and standard deviation of probabilities by condition
- Sample sizes for each experimental cell
- Descriptive statistics across all factors

### Interaction Effects Testing
- Chi-square tests for independence between factors
- Contingency table analysis for high/low probability choices
- Effect size calculations

### Pairwise Comparisons
- Mann-Whitney U tests between all condition pairs
- Multiple comparison corrections
- Effect size estimates for significant differences

## Configuration

### Experiment Configuration

```yaml
experiment:
  name: "emotion_context_probability_study"
  emotions: ["neutral", "angry"]
  emotion_intensities: [0.0, 1.0]
  output:
    base_dir: "experiments/option_probability"
```

### REPE Engine Configuration

```yaml
model_name_or_path: "meta-llama/Llama-3.1-8B-Instruct"
tensor_parallel_size: 1
coeffs: [0.0, 1.0]
block_name: "model.layers.{}.self_attn"
control_method: "reading_vec"
```

## Output Files

The experiment generates several output files:

### Primary Results
- **`option_probability_results.csv`**: Structured data for analysis
- **`option_probability_results.json`**: Raw experimental results

### Analysis Results
- **`statistical_analysis.json`**: Complete statistical analysis results
- **`experiment_summary_report.txt`**: Human-readable summary

### Log Files
- **`logs/option_probability_experiment_*.log`**: Detailed execution logs

## Data Structure

### CSV Output Structure

| Column | Description |
|--------|-------------|
| `condition_emotion` | Emotion condition (neutral/angry) |
| `condition_context` | Context condition (with/without description) |
| `option_id` | Option identifier (1, 2, ...) |
| `option_text` | Text of the behavioral choice |
| `probability` | Normalized probability [0, 1] |
| `log_probability` | Raw log probability |
| `scenario` | Scenario identifier |
| `behavior_choices` | Available choices |

## Testing

Comprehensive unit tests are provided for both main components:

### Run Dataset Tests
```bash
python -m unittest neuro_manipulation.tests.test_context_manipulation_dataset
```

### Run Experiment Tests
```bash
python -m unittest neuro_manipulation.tests.test_option_probability_experiment
```

### Run All Tests
```bash
python -m unittest discover neuro_manipulation/tests/ -p "test_*probability*.py"
```

## Integration Example

Complete example integrating both components:

```python
from neuro_manipulation.experiments.option_probability_experiment import OptionProbabilityExperiment
from games.game_configs import get_game_config
from constants import GameNames

# Setup configurations
game_config = get_game_config(GameNames.PRISONERS_DILEMMA)
repe_config = {
    'model_name_or_path': 'meta-llama/Llama-3.1-8B-Instruct',
    'tensor_parallel_size': 1
}
exp_config = {
    'experiment': {
        'name': 'pd_emotion_context_study',
        'emotions': ['neutral', 'angry'],
        'output': {'base_dir': 'results/'}
    }
}

# Run experiment
experiment = OptionProbabilityExperiment(
    repe_eng_config=repe_config,
    exp_config=exp_config, 
    game_config=game_config,
    batch_size=8,
    sample_num=500
)

results_file = experiment.run_experiment()
print(f"Results saved to: {results_file}")
```

## Research Applications

This framework enables investigation of:

1. **Emotion-Context Interactions**: How emotional states modify the effect of contextual information
2. **Decision Probability Modeling**: Quantitative modeling of choice probabilities under different conditions
3. **Behavioral Intervention Studies**: Testing interventions that modify decision-making patterns
4. **Cross-Game Generalization**: Studying whether emotion-context effects generalize across different game types

## Troubleshooting

### Common Issues

**vLLM Model Required**: The experiment requires vLLM models for sequence probability measurement
```python
# Ensure vLLM model is used
if not isinstance(self.model, LLM):
    raise ValueError("OptionProbabilityExperiment requires vLLM model")
```

**Memory Management**: Large batch sizes may cause memory issues
```python
# Reduce batch size for large models
experiment = OptionProbabilityExperiment(
    batch_size=2,  # Reduce if memory issues occur
    sample_num=100
)
```

**Statistical Analysis Requirements**: Ensure scipy is installed for statistical tests
```bash
pip install scipy>=1.7.0
```

## Future Extensions

Potential enhancements to the framework:

1. **Multi-Emotion Support**: Extend beyond angry/neutral to full emotion spectrum
2. **Dynamic Context Generation**: AI-generated context descriptions
3. **Longitudinal Studies**: Track probability changes over time
4. **Cross-Model Comparisons**: Compare probability patterns across different LLMs
5. **Real-Time Adaptation**: Adaptive experimental designs based on ongoing results

## References

- [SequenceProbVLLMHook Documentation](../repe/README_sequence_prob_vllm_hook.md)
- [Game Theory Framework](../../../games/README.md)
- [Statistical Engine](../../../statistical_engine.md) 