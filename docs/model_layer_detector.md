# Model Layer Detector

## Overview

The `ModelLayerDetector` is a utility class that automatically detects transformer layers in any model architecture without hardcoding model-specific paths. It's especially useful when working with multiple different model architectures or when you want to build model-agnostic tools.

## Features

- Works with any transformer-based model architecture (GPT, Llama, Mistral, ChatGLM, RWKV, etc.)
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

## Implementation Details

The detector uses a breadth-first search algorithm that:

1. Traverses the model structure level by level
2. Identifies transformer layers by checking for attention components and other transformer features
3. Prioritizes modules named "layers" and with shorter paths
4. Falls back to finding any `ModuleList` with consistent module types if no transformer layers are detected

This approach makes the detector robust to different model architectures and naming conventions. 