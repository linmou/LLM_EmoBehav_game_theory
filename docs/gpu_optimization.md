# GPU Optimization Guide

This document explains how to use the automatic batch size optimization features to maximize GPU utilization for your experiments.

## Overview

The batch size is a critical parameter that affects both memory usage and computational efficiency. Too small, and you'll underutilize your GPU. Too large, and you'll run out of memory.

To address this, the `neuro_manipulation.gpu_optimization` module provides a function (`find_optimal_batch_size_for_experiment`) which is automatically called by `EmotionGameExperiment` when configured. This function:
- Detects the optimal batch size for your specific GPU hardware
- Considers model architecture
- Accounts for input data characteristics
- Adapts to generation parameters

## Configuration Options

To enable automatic batch size optimization, set the `batch_size` parameter to the string `'auto_batch_size'` in your experiment configuration file:

```yaml
experiment:
  # ... other settings ...
  
  # Use string 'auto_batch_size' to enable automatic batch size detection
  batch_size: 'auto_batch_size'
  batch_size_safety_margin: 0.9  # Percentage of GPU memory to target (0.9 = 90%)
```

These settings control:
- Setting `batch_size` to `'auto_batch_size'` enables automatic detection within `EmotionGameExperiment`.
- `batch_size_safety_margin`: How much of the GPU memory to target (0.9 = 90%).

## How It Works

When `EmotionGameExperiment` is initialized and `batch_size` is set to `'auto_batch_size'`:

1. The experiment class calls `find_optimal_batch_size_for_experiment` from the `gpu_optimization` module.
2. This function creates a minimal sample dataset based on the experiment's configuration.
3. It uses binary search to find the maximum batch size that fits within the specified `batch_size_safety_margin` of the GPU memory.
4. The function measures memory usage by running a single generation step with `model.generate()` for each tested batch size.
5. It applies a small additional safety reduction (e.g., 5%) to the discovered batch size.
6. This optimized batch size is returned to `EmotionGameExperiment` and used for the actual experiment runs.

The binary search algorithm:

1. Starts with a batch size range (e.g., 1 to 2048).
2. Tests the middle batch size value.
3. If it fits in memory, it tries a larger size (adjusts the lower bound).
4. If it causes an Out-of-Memory (OOM) error, it tries a smaller size (adjusts the upper bound).
5. Repeats until the optimal size within the memory constraints is found.

## Best Practices

- Use `batch_size: 'auto_batch_size'` for the first run on new hardware or after significant changes to the model/data.
- Adjust the `batch_size_safety_margin` based on stability needs:
  - Higher values (e.g., 0.95) for more stability, leaving more memory headroom.
  - Lower values (e.g., 0.85) for more aggressive utilization, potentially faster but higher OOM risk.
- For models with highly variable memory usage per input, consider a slightly higher safety margin.

## Manually Setting Batch Size

If you prefer to specify a batch size directly, provide an integer value in the config:

```yaml
experiment:
  # ... other settings ...
  batch_size: 128  # Use exactly 128 as the batch size
```

## Troubleshooting

If you encounter OOM errors even with automatic batch sizing:

1. Verify your model configuration hasn't changed unexpectedly.
2. Check if the complexity or length of your input data has significantly increased.
3. Try increasing the `batch_size_safety_margin` (e.g., to 0.95 or even higher).
4. Ensure there are no other processes consuming significant GPU memory.
5. Check for potential memory leaks elsewhere in the code.

## Memory Utilization Metrics

The `measure_memory_usage` function returns detailed memory statistics:

- `start_memory`: Memory allocated before processing
- `peak_memory`: Maximum memory usage during processing
- `current_memory`: Memory usage after processing
- `memory_per_sample`: Memory used per sample in the batch 

# GPU Optimization in Multi-GPU Environments

## Overview

The `BatchSizeFinder` and related utilities in the `neuro_manipulation.gpu_optimization` module now support multi-GPU setups, including models loaded with `device_map="auto"`. This document explains how these optimizations work and how to use them effectively.

## Key Features

- **Multi-GPU Detection**: Automatically detects available GPUs and models distributed across multiple devices
- **Aggregate Memory Tracking**: Measures and tracks memory usage across all GPUs
- **Smart Input Handling**: Properly handles inputs for models split across multiple devices
- **Balanced Memory Utilization**: Finds the optimal batch size considering the total memory across all GPUs

## How It Works

### Device Detection

The system automatically detects:
1. The number of available GPUs
2. Whether your model is distributed across multiple devices (via `device_map="auto"`)
3. The total available memory across all devices

```python
# Inside the finder
num_gpus = torch.cuda.device_count()
for gpu_id in range(num_gpus):
    gpu_mem = torch.cuda.get_device_properties(gpu_id).total_memory
    total_gpu_mem += gpu_mem
```

### Memory Measurement

For proper multi-GPU support, memory is measured across all available devices:

```python
# Reset stats on all devices
for device_id in range(num_gpus):
    with torch.cuda.device(device_id):
        torch.cuda.reset_peak_memory_stats()

# Run model (forward or generate)
...

# Measure peak memory across all devices
total_peak_mem = 0
for device_id in range(num_gpus):
    with torch.cuda.device(device_id):
        peak_mem = torch.cuda.max_memory_allocated()
        total_peak_mem += peak_mem
```

### Input Handling

The system is designed to work with models that use `device_map="auto"` by:
1. Detecting the device of the first model parameter
2. Placing inputs on that device
3. Letting the model's internal mechanisms handle the distribution of computation across devices

## Usage Examples

### Finding Optimal Batch Size for a Multi-GPU Model

```python
from neuro_manipulation.gpu_optimization import find_optimal_batch_size_for_llm

# Load model with device_map="auto"
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Find optimal batch size
batch_size = find_optimal_batch_size_for_llm(
    model=model,
    tokenizer=tokenizer,
    sample_text="Sample input for testing batch size",
    max_length=512,
    generation_kwargs={"max_new_tokens": 100}
)

print(f"Optimal batch size: {batch_size}")
```

### For Experiments

```python
from neuro_manipulation.gpu_optimization import find_optimal_batch_size_for_experiment

# Assuming model, tokenizer, prompt_wrapper, game_config, and exp_config are defined
batch_size = find_optimal_batch_size_for_experiment(
    model=model,
    tokenizer=tokenizer,
    prompt_wrapper=prompt_wrapper,
    game_config=game_config,
    exp_config=exp_config,
    safety_margin=0.9
)
```

## Considerations

- The optimal batch size is determined based on the total memory across all GPUs
- A safety margin (default 90%) and reduction factor (default 95%) are applied to prevent OOM errors
- For heavily unbalanced models (e.g., much larger layers on one GPU), the memory distribution may not be optimal
- Always test the suggested batch size before running large-scale experiments

## Limitations

- The implementation assumes that PyTorch's `device_map="auto"` or similar mechanisms handle the distribution of model layers across devices
- Memory tracking may not be perfectly accurate for some advanced model parallelism techniques
- External factors (like CUDA caching) might affect the actual usable memory 