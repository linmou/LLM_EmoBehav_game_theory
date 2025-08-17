# Emotion Memory Experiment Configuration

This directory contains YAML configuration files for emotion memory experiments.

## Available Configurations

### Basic Configurations
- `emotion_memory_passkey.yaml` - Passkey retrieval task configuration
- `emotion_memory_kv_retrieval.yaml` - Key-value retrieval task configuration  
- `emotion_memory_comprehensive.yaml` - Full-featured configuration with all options

## Configuration Structure

```yaml
experiment:
  name: "Experiment_Name"
  model_path: "/path/to/model"
  
  emotions:
    - "anger"
    - "happiness"
    - "sadness"
  
  intensities:
    - 0.5
    - 1.0
    - 1.5
  
  benchmark:
    name: "infinitebench"
    data_path: "/path/to/data.jsonl"
    task_type: "passkey"
    evaluation_method: "get_score_one_passkey"
    sample_limit: 100
  
  generation_config:
    temperature: 0.1
    max_new_tokens: 50
    do_sample: false
    top_p: 0.9
  
  batch_size: 4
  
  output:
    base_dir: "results/emotion_memory"
```

## Usage

```python
from emotion_memory_experiments.config_loader import load_emotion_memory_config
from emotion_memory_experiments import EmotionMemoryExperiment

# Load configuration
config = load_emotion_memory_config("config/emotion_memory_passkey.yaml")

# Run experiment
experiment = EmotionMemoryExperiment(config)
results = experiment.run_experiment()
```

## Configuration Options

### Required Fields
- `model_path`: Path to the model directory
- `benchmark.data_path`: Path to benchmark data file

### Optional Fields
- `emotions`: List of emotions to test (default: ["anger", "happiness"])
- `intensities`: List of intensity values (default: [1.0])
- `benchmark.name`: Benchmark adapter name (default: "infinitebench")
- `benchmark.task_type`: Task type (default: "passkey")
- `benchmark.evaluation_method`: Evaluation function (default: "auto")
- `benchmark.sample_limit`: Limit samples for testing (default: None)
- `generation_config`: Model generation parameters
- `batch_size`: Processing batch size (default: 4)
- `output.base_dir`: Results directory (default: "results/emotion_memory")

## Supported Benchmarks

### InfiniteBench Tasks
- `passkey`: Passkey retrieval from long context
- `kv_retrieval`: Key-value pair retrieval
- `longbook_qa_eng`: Long book reading comprehension
- `longbook_sum_qa`: Long book summarization QA
- `longbook_choice_eng`: Long book multiple choice
- `longbook_qa_chn`: Chinese long book QA
- `math_find`: Mathematical pattern finding
- `math_calc`: Mathematical calculation
- `code_run`: Code execution prediction
- `code_debug`: Code debugging

### Data Paths
Data should be located in `/data/home/jjl7137/memory_benchmarks/InfiniteBench/data/`

## Example Commands

```bash
# Run with YAML config
python -c "
from emotion_memory_experiments.config_loader import load_emotion_memory_config
from emotion_memory_experiments import EmotionMemoryExperiment
config = load_emotion_memory_config('config/emotion_memory_passkey.yaml')
experiment = EmotionMemoryExperiment(config)
results = experiment.run_experiment()
"

# Run sanity check
python emotion_memory_experiments/example_usage.py
```