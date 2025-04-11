# Model Layer Detector

## Overview

The `ModelLayerDetector` is a utility class that automatically detects transformer layers in any model architecture without hardcoding model-specific paths. It's especially useful when working with multiple different model architectures or when you want to build model-agnostic tools.

## Features

- Works with any transformer-based model architecture:
  - HuggingFace Transformers models (GPT-2, OPT, Llama, etc.)
  - vLLM-hosted models
  - HuggingFace Custom transformer architectures, like RWKV, Mamaba, xLSTM
- Uses breadth-first search to efficiently traverse the model's module hierarchy
- Identifies transformer layers based on common architectural patterns
- Handles deeply nested model structures
- No configuration or hardcoded paths required

## API Reference

### `ModelLayerDetector.get_model_layers(model)`

Find transformer layers in a model using breadth-first search traversal.

**Parameters:**
- `model` (torch.nn.Module): Any PyTorch model

**Returns:**
- `torch.nn.ModuleList`: The detected transformer layers

**Raises:**
- `ValueError`: If no transformer layers could be detected

### `ModelLayerDetector.print_model_structure(model, max_depth=3)`

Print the structure of a PyTorch model to help with debugging.

**Parameters:**
- `model` (torch.nn.Module): Any PyTorch model
- `max_depth` (int, optional): Maximum depth to print. Defaults to 3.

## Examples

### Basic Usage

```python
from transformers import AutoModel
from neuro_manipulation.model_layer_detector import ModelLayerDetector

# Load any model
model = AutoModel.from_pretrained("gpt2")

# Automatically detect its layers
layers = ModelLayerDetector.get_model_layers(model)

print(f"Found {len(layers)} transformer layers")
```

### Debugging Model Structure

```python
# Print the model structure to understand its hierarchy
ModelLayerDetector.print_model_structure(model)
```

### Working with Different Model Architectures

```python
# Works with any model architecture
from transformers import AutoModel, AutoModelForCausalLM

# ChatGLM models
chatglm = AutoModel.from_pretrained("THUDM/glm-4-9b", trust_remote_code=True)
chatglm_layers = ModelLayerDetector.get_model_layers(chatglm)

# RWKV models
rwkv = AutoModelForCausalLM.from_pretrained("BlinkDL/rwkv-4-raven", trust_remote_code=True)
rwkv_layers = ModelLayerDetector.get_model_layers(rwkv)
```

### Integration with vLLM

The `ModelLayerDetector` is designed to work with vLLM's nested model structure:

```python
# Example usage with vLLM
from model_layer_detector import ModelLayerDetector
from vllm import LLM

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
model = llm.llm_engine.model_executor.driver_worker.model_runner.model
model_layers = ModelLayerDetector.get_model_layers(model)

# Now model_layers contains the transformer layers from the vLLM model
# which should match model.model.layers for Llama models
```

This feature is particularly important for representation engineering techniques that modify model layers in real-time.

## Implementation Details

The detector uses a breadth-first search algorithm that:

1. Traverses the model structure level by level
2. Identifies transformer layers by checking for attention components and other transformer features
3. Prioritizes modules named "layers" and with shorter paths
4. Falls back to finding any `ModuleList` with consistent module types if no transformer layers are detected

This approach makes the detector robust to different model architectures and naming conventions.

## Testing

For detailed information about testing the `ModelLayerDetector` with different models, see the [test documentation](../neuro_manipulation/tests/README.md). The tests verify compatibility with:

1. **Small Models** - Tests with lightweight models like GPT-2 and OPT-125M
2. **ChatGLM Models** - Specific test for ChatGLM architecture
3. **RWKV Models** - Specific test for RWKV architecture
4. **Custom Models** - Tests with custom-built transformer architectures
5. **vLLM Models** - Tests with vLLM-hosted models like Llama-3.1-8B-Instruct 