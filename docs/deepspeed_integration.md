# DeepSpeed Integration for Neural Representation Control

This document explains how to use DeepSpeed with our neural representation control pipeline to accelerate inference and enable processing of larger models through pipeline parallelism.

## Overview

DeepSpeed provides several optimizations for large language model inference:

1. **Pipeline Parallelism**: Splits model layers across multiple GPUs
2. **Memory Optimization**: More efficient memory usage during inference
3. **Faster Inference**: Optimized kernels for better performance

Our integration uses DeepSpeed's pipeline parallelism while preserving the ability to apply neural representation control through forward hooks.

## Setup and Installation

### Requirements

- DeepSpeed (`pip install deepspeed`)
- PyTorch
- Transformers
- Our neural manipulation package

### Basic Configuration

Create a DeepSpeed configuration JSON file:

```json
{
  "train_batch_size": 1,  // Not used for inference
  "train_micro_batch_size_per_gpu": 1, // Not used for inference
  "steps_per_print": 1,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.001,
      "weight_decay": 0
    }
  },
  "pipeline": {
    "stages": 2,  // Split model into 2 pipeline stages
    "activation_checkpoint_interval": 0  // Disable activation checkpointing for inference
  },
  "zero_optimization": {
    "stage": 0  // Disable ZeRO for inference pipeline
  }
}
```

## Usage

### Basic Example

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from neuro_manipulation.repe.pipelines import get_pipeline
import torch

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define DeepSpeed config
ds_config = {
    "train_batch_size": 1,
    "pipeline": {
        "stages": 2,
        "activation_checkpoint_interval": 0
    },
    "zero_optimization": {
        "stage": 0
    }
}

# Create pipeline with DeepSpeed
pipeline = get_pipeline(
    "rep-control-deepspeed",
    model=model,
    tokenizer=tokenizer,
    layers=[-1, -2, -3],  # Control last three layers
    deepspeed_config=ds_config
)

# Define some activation vectors
activations = {
    -1: torch.randn(1, 1, 4096),  # Example for layer -1
    -2: torch.randn(1, 1, 4096),  # Example for layer -2
    -3: torch.randn(1, 1, 4096),  # Example for layer -3
}

# Generate text with representation control
outputs = pipeline(
    "Write a story about a robot that learns to love:",
    activations=activations,
    token_pos="all",  # Apply to all tokens
    max_new_tokens=100
)

print(outputs[0]["generated_text"])
```

### Using with Emotion Game Experiment

To use DeepSpeed with the existing emotion game experiment framework:

```python
from neuro_manipulation.experiments.emotion_game_experiment import EmotionGameExperiment
from neuro_manipulation.repe.pipelines import get_pipeline

# Your experiment setup code
repe_eng_config = {...}
exp_config = {...}
game_config = {...}

# Initialize experiment
experiment = EmotionGameExperiment(repe_eng_config, exp_config, game_config)

# Replace the default pipeline with DeepSpeed pipeline
deepspeed_config = {
    "train_batch_size": 1,
    "pipeline": {
        "stages": 2,
        "activation_checkpoint_interval": 0
    },
    "zero_optimization": {
        "stage": 0
    }
}

experiment.rep_control_pipeline = get_pipeline(
    "rep-control-deepspeed",
    model=experiment.model,
    tokenizer=experiment.tokenizer,
    layers=experiment.repe_eng_config['control_layer_id'],
    block_name=experiment.repe_eng_config['block_name'],
    control_method=experiment.repe_eng_config['control_method'],
    deepspeed_config=deepspeed_config
)

# Run the experiment as usual
results = experiment.run_experiment()
```

## Best Practices

1. **Pipeline Stages**: Choose the number of pipeline stages based on:
   - Your model size
   - Number of available GPUs
   - Memory per GPU

2. **Layer Distribution**: Be aware of which layers are on which pipeline stage. The `pipeline_rank` attribute can help determine this.

3. **Memory Management**: Monitor GPU memory usage. DeepSpeed will show memory statistics during initialization.

## Troubleshooting

### Common Issues

1. **Layer Not Found**: If you're getting warnings about layers not found, check the pipeline rank assignment.

2. **CUDA Out of Memory**: Try increasing the number of pipeline stages to distribute the model across more GPUs.

3. **Slow Initial Runtime**: The first run with DeepSpeed involves compilation time; subsequent runs will be faster.

## Performance Tuning

For optimal performance:

1. Experiment with different pipeline configurations
2. Consider using DeepSpeed's inference optimization features
3. Benchmark with different batch sizes

## Limitations

- Currently focused on inference only (not training)
- Layer access is more complex with pipeline parallelism
- Some models might require custom layer detection logic 