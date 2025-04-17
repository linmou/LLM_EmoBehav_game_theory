# Multi-GPU Optimization for LLM Experiments

This document explains how to effectively utilize multiple GPUs in your language model experiments using our automatic batch size detection and optimization features.

## Overview

Our codebase now fully supports multi-GPU setups, including models loaded with `device_map="auto"` for automatic model sharding across available GPUs. This allows you to run experiments with larger models that would not fit on a single GPU.

The key components are:
1. Automatic batch size detection that works across multiple GPUs
2. Configuration options for model loading with device maps
3. Memory optimization across all available GPU devices

## Configuration

### Experiment Configuration

To enable automatic batch size detection in your experiment config:

```yaml
experiment:
  # ... other settings ...
  
  # Use string 'auto_batch_size' to enable automatic detection
  batch_size: "auto_batch_size"
  
  # Optional: Adjust safety margin (default: 0.9)
  batch_size_safety_margin: 0.9
  
  # ... other settings ...
```

### Model Loading Configuration

To enable multi-GPU model loading, add these settings to your model configuration:

```yaml
model_loading:
  # Set to "auto" to enable model sharding across available GPUs
  device_map: "auto"
  
  # Optional: Memory optimization techniques
  load_in_8bit: false  # Set to true for 8-bit quantization
  load_in_4bit: false  # Set to true for 4-bit quantization (or use 8-bit)
  
  # Optional: Advanced 4-bit settings if using 4-bit quantization
  quantization_config:
    bnb_4bit_compute_dtype: "float16"
    bnb_4bit_use_double_quant: true
    bnb_4bit_quant_type: "nf4"
```

## Usage Examples

### Running an Experiment with Multi-GPU Support

```python
from neuro_manipulation.experiments.emotion_game_experiment import EmotionGameExperiment

# Load configurations (yaml files)
repe_eng_config = load_config('config/repe_eng_config.yaml')
exp_config = load_config('config/multi_gpu_example_config.yaml')
game_config = load_config('config/game_theory_config.yaml')

# Initialize experiment with auto batch size
experiment = EmotionGameExperiment(
    repe_eng_config=repe_eng_config,
    exp_config=exp_config,
    game_config=game_config,
    batch_size=None,  # None will use the config setting (auto_batch_size)
    sample_num=None,  # Use all samples
    repeat=1
)

# Run the experiment
experiment.run_experiment()
```

## How It Works

When automatic batch size detection is enabled:

1. The system detects all available GPUs and their total memory
2. The model is analyzed to detect if it's distributed across multiple devices
3. Batch size finding starts with a small value and gradually increases
4. Memory usage is measured across all GPUs in aggregate
5. The largest batch size that fits within the safety margin is selected
6. A small reduction factor (0.95) is applied for extra safety

## Debugging and Monitoring

The system logs detailed information about GPU detection and memory usage:

```
INFO - Found 2 GPUs
INFO - GPU 0: NVIDIA A100-SXM4-40GB with 40.00 GB
INFO - GPU 1: NVIDIA A100-SXM4-40GB with 40.00 GB
INFO - Total GPU memory across all devices: 80.00 GB
INFO - Target memory usage (with 90.0% safety margin): 72.00 GB
INFO - Power phase - Trial 1: Testing batch size 1
INFO - Batch size 1 uses 5.23 GB
...
INFO - Found optimal batch size: 32, using 30 for safety
```

## Tips for Multi-GPU Usage

1. **Balance Safety Margin**: Adjust `batch_size_safety_margin` based on your workload. Lower values (0.7-0.8) are safer but less efficient.

2. **Memory Optimization**: For very large models, combine automatic batch size with quantization options like `load_in_8bit: true`.

3. **Benchmark Performance**: The optimal batch size for throughput may be different from the maximum that fits in memory. Consider running manual benchmarks.

4. **Monitor GPU Usage**: Use tools like `nvidia-smi` to monitor actual GPU memory usage during execution.

5. **Checkpoint Frequency**: With larger batch sizes across multiple GPUs, consider more frequent checkpointing as more computation is at risk in case of failures.

## Troubleshooting

If you encounter OOM (Out of Memory) errors despite using automatic batch size:

1. Decrease the `batch_size_safety_margin` to 0.7 or 0.8
2. Enable quantization with `load_in_8bit: true`
3. Reduce the `max_new_tokens` in your generation config
4. Check if your model has unbalanced weight distribution (some GPUs may be more loaded than others)

## Further Reading

For more details on how GPU memory optimization works, see:
- [GPU Optimization Documentation](./gpu_optimization.md)
- The implementation in `neuro_manipulation/gpu_optimization.py` 