# Option Probability Experiment

## Overview

The Option Probability Experiment framework provides a comprehensive solution for studying emotion-context interactions in behavioral decision-making through a 2x2 factorial design. This framework measures the probabilities of all available behavioral choices using advanced sequence probability techniques with vLLM models.

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

## Implementation Details

### Sequence Probability Measurement

The experiment uses `SequenceProbVLLMHook` to accurately measure choice probabilities:

```python
# Hook integration for probability measurement
self.sequence_prob_hook = SequenceProbVLLMHook(
    model=self.model,
    tokenizer=self.tokenizer,
    tensor_parallel_size=tensor_parallel_size
)

# Measure probabilities for all options
prob_results = self.sequence_prob_hook.get_log_prob(
    text_inputs=[prompt],
    target_sequences=option_texts
)
```

### Context Manipulation

Context manipulation is achieved through conditional prompt modification:

```python
# With description
if self.include_description:
    event_with_context = f"{event}\n\nContext: {description}"
else:
    event_with_context = event
```

### Emotion Intervention

Emotion states are induced through user message modifications:

```python
if condition.emotion == "neutral":
    user_message = "You are participating in this scenario."
else:
    user_message = f"You are participating in this scenario. You are feeling {condition.emotion}."
```

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