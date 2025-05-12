# Model Download and Management

## Overview

The ExperimentSeriesRunner has been enhanced with automatic model checking and downloading capabilities. This feature ensures that models required for experiments are downloaded before the experiment starts, improving the reliability of batch experiments.

## How it Works

The system checks for model existence in two locations:
1. HuggingFace's default cache location (`~/.cache/huggingface/hub/`)
2. An alternative location in the parent directory (`../huggingface_models`)

If a model is not found in either location, it will be downloaded to the alternative location using the `huggingface-cli` command-line tool, which is more reliable and efficient than the Python API, especially for large models.

## Implementation Details

### Model Checking Logic

The core functionality is implemented in the `_check_model_existence` method:

```python
def _check_model_existence(self, model_name: str) -> bool:
    """
    Check if the model exists in either ~/.cache/huggingface/hub/ or ../huggingface_models.
    If not, download it to ../huggingface_models.
    
    Args:
        model_name: The name of the model to check
        
    Returns:
        bool: True if model exists or was successfully downloaded, False otherwise
    """
    # Implementation details...
```

### Features:

1. **Local Path Support**: Models specified with absolute paths (starting with `/`) are assumed to be local and skipped from checking.
2. **Path Detection**: For HuggingFace models, it correctly handles the path structure used by HuggingFace, including organization and model name.
3. **Command-line Downloads**: Uses `huggingface-cli download` with `HF_HUB_ENABLE_HF_TRANSFER=1` for faster, more reliable downloads.
4. **Real-time Progress**: Streams download progress to the log in real time.
5. **Failure Handling**: If a model fails to download, the experiment with that model is marked as failed but doesn't stop the entire series.

### Integration in Experiment Series Runner

The model checking is integrated at two key points:

1. **Pre-check all models**: At the start of the experiment series, all models are checked to pre-download them.
2. **Per-experiment check**: Before each individual experiment, the model is checked again to ensure it's available.

## Usage

No additional configuration is needed. The ExperimentSeriesRunner will automatically check for models and download them as needed.

### Download Command

The system uses this command format for downloads:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download [MODEL_NAME] --local-dir [TARGET_DIR]
```

For example:
```bash
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ../huggingface_models/Qwen/Qwen2.5-7B-Instruct
```

## Testing

Unit tests for this functionality are available in `neuro_manipulation/tests/test_model_check.py`. The tests cover:

1. Local model path detection
2. Existing model detection
3. Model download process
4. Handling of download failures

## Notes

- The download process uses the optimized Hugging Face transfer protocol via the environment variable `HF_HUB_ENABLE_HF_TRANSFER=1`.
- This approach is generally faster and more reliable than using the Python API directly, especially for large models.
- If a model fails to download, the experiment involving that model will be marked as failed, but other experiments in the series will continue running. 