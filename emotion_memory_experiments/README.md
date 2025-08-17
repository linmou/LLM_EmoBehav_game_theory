# Emotion Memory Experiment Framework

A comprehensive framework for testing the effects of emotional states on long-context memory tasks using neural representation control.

## Overview

This framework enables researchers to study how induced emotional states affect LLM performance on memory benchmarks. It builds on the existing emotion manipulation framework used for game theory experiments and extends it to long-context memory tasks.

### Key Features

- **Emotion-Aware Evaluation**: Apply different emotional states (anger, happiness, sadness, etc.) during memory task inference
- **Multi-Benchmark Support**: Works with InfiniteBench, LoCoMo, and other memory benchmarks
- **Native Evaluation**: Uses each benchmark's original evaluation methods for accurate scoring
- **Comprehensive Testing**: Full test suite with mock data and integration tests
- **Simple Configuration**: YAML-based configuration following existing project patterns

## Architecture

```
emotion_memory_experiments/
├── __init__.py
├── data_models.py          # Configuration and result data structures
├── benchmark_adapters.py   # Adapters for different benchmark formats
├── experiment.py           # Main experiment orchestration class
├── example_usage.py        # Usage examples and demonstrations
├── tests/                  # Comprehensive test suite
│   ├── test_data_models.py
│   ├── test_benchmark_adapters.py
│   ├── test_experiment.py
│   ├── test_integration.py
│   ├── test_utils.py
│   └── run_all_tests.py
└── README.md
```

## Quick Start

### 1. Basic Usage

```python
from emotion_memory_experiments.experiment import EmotionMemoryExperiment
from emotion_memory_experiments.data_models import ExperimentConfig, BenchmarkConfig

# Configure benchmark
benchmark_config = BenchmarkConfig(
    name="infinitebench",
    data_path=Path("passkey_data.jsonl"),
    task_type="passkey",
    evaluation_method="get_score_one_passkey"
)

# Configure experiment
exp_config = ExperimentConfig(
    model_path="/path/to/qwen/model",
    emotions=["anger", "happiness", "sadness"],
    intensities=[0.5, 1.0, 1.5],
    benchmark=benchmark_config,
    output_dir="results/emotion_memory",
    batch_size=4
)

# Run experiment
experiment = EmotionMemoryExperiment(exp_config)
results_df = experiment.run_experiment()
```

### 2. Quick Sanity Check

```python
# Run a quick test with limited samples
results_df = experiment.run_sanity_check(sample_limit=5)
```

## Supported Benchmarks

### InfiniteBench Tasks
- **Passkey Retrieval**: Find hidden keys in long contexts
- **Key-Value Retrieval**: Locate values for specific keys
- **Number String**: Find repeated number sequences
- **Reading Comprehension**: Answer questions about long texts
- **Code Tasks**: Debug and execution simulation
- **Math Tasks**: Arithmetic and pattern finding

### LoCoMo Tasks
- **Conversational QA**: Answer questions about multi-session conversations
- **Event Summarization**: Summarize events across conversation sessions

## Configuration

### Benchmark Configuration

```python
BenchmarkConfig(
    name="infinitebench",           # Benchmark suite name
    data_path=Path("data.jsonl"),   # Path to benchmark data
    task_type="passkey",            # Specific task type
    evaluation_method="get_score_one_passkey",  # Evaluation function
    sample_limit=100                # Optional: limit number of samples
)
```

### Experiment Configuration

```python
ExperimentConfig(
    model_path="/path/to/model",
    emotions=["anger", "happiness"],     # Emotions to test
    intensities=[0.5, 1.0],             # Intensity levels
    benchmark=benchmark_config,
    output_dir="results",
    batch_size=4,
    generation_config={                  # Optional: custom generation settings
        "temperature": 0.1,
        "max_new_tokens": 100,
        "do_sample": False,
        "top_p": 0.9
    }
)
```

## Data Format

### Input Data Format (InfiniteBench)
```jsonl
{"id": 0, "context": "long context...", "input": "What is the passkey?", "answer": "12345"}
{"id": 1, "context": "long context...", "input": "What is the passkey?", "answer": "67890"}
```

### Output Results Format
```csv
emotion,intensity,item_id,task_name,response,ground_truth,score,benchmark
anger,1.0,0,passkey,"The passkey is 12345",12345,1.0,infinitebench
happiness,0.5,1,passkey,"I think it's 67890",67890,1.0,infinitebench
neutral,0.0,0,passkey,"12345",12345,1.0,infinitebench
```

## Testing

### Run All Tests
```bash
cd emotion_memory_experiments/tests
python run_all_tests.py
```

### Run Specific Test Module
```bash
python run_all_tests.py data_models    # Test data models
python run_all_tests.py adapters       # Test benchmark adapters
python run_all_tests.py experiment     # Test main experiment class
python run_all_tests.py integration    # Test full workflow
```

### Test Coverage
- **Unit Tests**: All components tested in isolation
- **Integration Tests**: Full workflow validation
- **Mock Data Tests**: Controlled test scenarios
- **Error Handling**: Exception and edge case testing

## Examples

### Run Example Experiments
```bash
python example_usage.py
```

This will demonstrate:
1. Passkey retrieval experiment
2. Key-value retrieval experiment  
3. Sanity check workflow
4. Configuration examples

## Integration with Existing Framework

This framework seamlessly integrates with the existing emotion manipulation system:

- **Reuses Emotion Readers**: Uses the same RepE emotion extraction
- **Same Model Setup**: Compatible with existing model loading utilities
- **Consistent Patterns**: Follows emotion_game_experiment.py patterns
- **Shared Dependencies**: Uses the same vLLM and RepE pipelines

## Performance Considerations

- **Batch Processing**: Configurable batch sizes for memory efficiency
- **Lazy Loading**: Benchmark data loaded only when needed
- **Memory Management**: Careful model cleanup between operations
- **Parallel Evaluation**: Threaded response processing where possible

## Extending the Framework

### Adding New Benchmarks

1. Create a new adapter class inheriting from `BenchmarkAdapter`
2. Implement the required methods:
   - `load_data()`: Parse benchmark data format
   - `create_prompt()`: Generate prompts from items
   - `evaluate_response()`: Score responses using benchmark method
3. Register in the `get_adapter()` factory function

```python
class CustomBenchmarkAdapter(BenchmarkAdapter):
    def load_data(self) -> List[BenchmarkItem]:
        # Load and parse custom format
        pass
    
    def create_prompt(self, item: BenchmarkItem) -> str:
        # Create task-specific prompt
        pass
    
    def evaluate_response(self, response: str, ground_truth: Any, task_name: str) -> float:
        # Use benchmark's evaluation method
        pass
```

### Adding New Task Types

Simply specify the new task type in your `BenchmarkConfig` and ensure the corresponding evaluation method is available.

## Dependencies

- **Core Framework**: Uses existing neuro_manipulation components
- **Model Support**: Compatible with vLLM and Transformers
- **Evaluation**: Integrates with InfiniteBench compute_scores.py
- **Data Processing**: pandas, numpy for result analysis

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure InfiniteBench path is added to sys.path
2. **Model Loading**: Verify model path and RepE setup
3. **Memory Issues**: Reduce batch_size for large models
4. **Evaluation Errors**: Check task_type matches evaluation method

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Research Applications

This framework enables research questions such as:

- **Memory Type Effects**: How do emotions affect different types of memory (working, episodic, semantic)?
- **Context Length Interactions**: Do emotional effects scale with context length?
- **Task Complexity**: How do emotions impact simple retrieval vs. complex reasoning?
- **Emotion Specificity**: Which emotions help or hinder specific memory tasks?

## Citation

When using this framework, please cite:
- The original emotion manipulation work
- Relevant memory benchmark papers (InfiniteBench, LoCoMo, etc.)
- Any specific models or datasets used

## License

This framework inherits the license of the parent project.

## Contributing

1. Follow existing code patterns and style
2. Add comprehensive tests for new features
3. Update documentation for any changes
4. Ensure compatibility with existing emotion framework