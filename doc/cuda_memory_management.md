# CUDA Memory Management

## Overview

The experiment series runner includes memory management functionality to clean up CUDA memory between experiments. This is important for preventing memory leaks and out-of-memory errors when running multiple experiments in sequence, especially when using large language models that consume significant GPU resources.

## Implementation

The CUDA memory cleanup is implemented in the `_clean_cuda_memory` method of the `ExperimentSeriesRunner` class. This method:

1. Runs Python's garbage collector to remove any unreferenced objects
2. Uses PyTorch's `torch.cuda.empty_cache()` to free up CUDA memory allocated by PyTorch
3. Collects and logs memory statistics to monitor GPU usage
4. Runs `nvidia-smi` to report current GPU memory usage

## When Memory Cleanup Happens

CUDA memory cleanup is triggered:

- After each successful experiment completes
- After an experiment fails with an exception
- Before the next experiment in the series begins

## Memory Usage Reporting

After cleanup, the system reports:
- PyTorch's CUDA memory stats:
  - Currently allocated memory
  - Maximum allocated memory
  - Currently reserved memory
  - Maximum reserved memory
- NVIDIA-SMI output showing used and free memory

## Troubleshooting GPU Memory Issues

If you encounter CUDA out-of-memory errors:

1. Check the memory usage logs to identify which experiments are consuming the most memory
2. Consider reducing model size or batch size for memory-intensive experiments
3. Ensure the cleanup function is being called properly
4. For persistent memory issues, you may need to restart the Python process or the system

## Example Log Output

```
2023-06-01 10:15:30,123 - neuro_manipulation.experiment_series_runner - INFO - Cleaning up CUDA memory...
2023-06-01 10:15:30,234 - neuro_manipulation.experiment_series_runner - INFO - Clearing CUDA cache...
2023-06-01 10:15:30,456 - neuro_manipulation.experiment_series_runner - INFO - CUDA memory stats after cleanup: allocated=0.00GB, max_allocated=15.42GB, reserved=16.00GB, max_reserved=16.00GB
2023-06-01 10:15:30,567 - neuro_manipulation.experiment_series_runner - INFO - Running nvidia-smi to check GPU memory...
2023-06-01 10:15:30,789 - neuro_manipulation.experiment_series_runner - INFO - NVIDIA-SMI report:
memory.used [MiB], memory.free [MiB]
16384, 24576
``` 