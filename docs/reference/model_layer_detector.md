# Model Layer Detector

## Overview

The `ModelLayerDetector` is a utility class that automatically detects transformer layers in any model architecture without hardcoding model-specific paths. It's especially useful when working with multiple different model architectures or when you want to build model-agnostic tools.

## Key Features

- **Model-Agnostic**: Works with any transformer-based model architecture (e.g., GPT-2, OPT, Llama, Mistral, ChatGLM, RWKV, Mamba, xLSTM, and other HuggingFace or custom transformer architectures).
- **Zero Configuration**: No need to specify model-specific paths or patterns.
- **Flexible Detection**: Uses intelligent heuristics and breadth-first search to identify transformer layers.
- **Handles Nesting**: Efficiently traverses deeply nested model structures, common in frameworks like vLLM.
- **Robust Fallbacks**: Includes fallback mechanisms if standard patterns are not found.

## API Reference

### `ModelLayerDetector.get_model_layers(model)`

Find transformer layers in a model using breadth-first search traversal.

**Parameters:**
- `model` (torch.nn.Module): Any PyTorch model.

**Returns:**
- `torch.nn.ModuleList`: The detected transformer layers.

**Raises:**
- `ValueError`: If no transformer layers could be detected.

### `ModelLayerDetector.num_layers(model)`

Returns the number of transformer layers detected in the model. This is a convenience method that calls `len(ModelLayerDetector.get_model_layers(model))`.

**Parameters:**
- `model` (torch.nn.Module): Any PyTorch model.

**Returns:**
- `int`: The number of detected transformer layers.

### `ModelLayerDetector.print_model_structure(model, max_depth=3)`

Print the structure of a PyTorch model to help with debugging.

**Parameters:**
- `model` (torch.nn.Module): Any PyTorch model.
- `max_depth` (int, optional): Maximum depth to print. Defaults to 3.

## How It Works / Implementation Details

The `ModelLayerDetector` employs a sophisticated strategy to identify transformer layers:

1.  **Breadth-First Search (BFS) Traversal**: The detector explores the model's module hierarchy level by level using BFS. This ensures that layers closer to the model root are considered first.
2.  **Transformer Layer Identification**:
    *   It defines characteristics of a transformer layer, such as the presence of attention components (e.g., attributes named `attention`, `self_attn`, `attn`) or other common transformer parts like `mlp`, `ffn`, `layernorm`.
    *   It checks `torch.nn.ModuleList` instances to see if they contain a sequence of such transformer-like modules.
3.  **Candidate Prioritization**:
    *   The detector prioritizes `nn.ModuleList` instances that are explicitly named `layers` (e.g., `model.layers`).
    *   Among candidates, those with shorter module paths (i.e., less deeply nested) are preferred.
4.  **Fallback Mechanism**: If the primary heuristics don't identify layers, the detector falls back to searching for any `nn.ModuleList` where all contained modules are of the same class type. This can help with less conventionally structured models.
5.  **Visited Module Tracking**: To handle potential circular references in model architectures and improve efficiency, the detector keeps track of already visited modules during traversal.

This multi-faceted approach makes the detector robust to diverse model architectures and common naming conventions.

## Examples

### Basic Usage

```python
from transformers import AutoModel
from neuro_manipulation.model_layer_detector import ModelLayerDetector

# Load any model
model = AutoModel.from_pretrained("gpt2")

# Automatically detect its layers
layers = ModelLayerDetector.get_model_layers(model)
num_layers = ModelLayerDetector.num_layers(model)

print(f"Found {len(layers)} transformer layers (also confirmed by num_layers: {num_layers})")
```

### Debugging Model Structure

```python
# Print the model structure to understand its hierarchy
ModelLayerDetector.print_model_structure(model)
```

### Working with Different Model Architectures

The detector is designed to be model-agnostic:
```python
from transformers import AutoModel, AutoModelForCausalLM
from neuro_manipulation.model_layer_detector import ModelLayerDetector # Ensure correct import

# Example with a Llama-like model (replace with actual loading if needed)
# model_llama = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf") 
# layers_llama = ModelLayerDetector.get_model_layers(model_llama)
# print(f"Llama-like model: {len(layers_llama)} layers")

# ChatGLM models
# Assuming 'THUDM/glm-4-9b' is available and loads correctly
# chatglm = AutoModel.from_pretrained("THUDM/glm-4-9b", trust_remote_code=True)
# chatglm_layers = ModelLayerDetector.get_model_layers(chatglm)
# print(f"ChatGLM model: {len(chatglm_layers)} layers")

# RWKV models
# rwkv = AutoModelForCausalLM.from_pretrained("BlinkDL/rwkv-4-raven", trust_remote_code=True)
# rwkv_layers = ModelLayerDetector.get_model_layers(rwkv)
# print(f"RWKV model: {len(rwkv_layers)} layers")
```
*(Note: The above examples for specific architectures might require the respective models to be downloaded and accessible in your environment.)*

### Integration with vLLM

The `ModelLayerDetector` is also designed to work with vLLM's potentially nested model structure:

```python
# Example usage with vLLM (conceptual, requires vLLM setup)
# from neuro_manipulation.model_layer_detector import ModelLayerDetector
# from vllm import LLM

# llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
# Ensure you access the correct underlying PyTorch model from the vLLM LLM object
# This path might vary based on vLLM version and setup
# model_vllm_runner = llm.llm_engine.model_executor.driver_worker.model_runner.model 
# model_layers_vllm = ModelLayerDetector.get_model_layers(model_vllm_runner)

# print(f"vLLM (Llama-3.1-8B-Instruct) has: {len(model_layers_vllm)} layers")
# For Llama models, this often corresponds to model.model.layers
```
This capability is crucial for representation engineering techniques that need to dynamically access or modify model layers within a vLLM serving environment.

## When to Use

This utility is particularly useful when:

1.  Working with multiple different model architectures.
2.  Building model-agnostic tools and pipelines for layer manipulation or analysis.
3.  Exploring new or custom model architectures without prior knowledge of their structure.
4.  Avoiding hardcoded model-specific layer paths that can break with model updates or variations.

## Testing

For detailed information about testing the `ModelLayerDetector` with various models, including standard HuggingFace models, custom architectures, and vLLM-hosted models, please refer to the [test documentation in the `neuro_manipulation/tests/` directory](./../code_readme/neuro_manipulation/tests/README.md). The tests verify compatibility with:

1.  **Small Standard Models**: e.g., GPT-2, OPT-125M.
2.  **Specialized Architectures**: e.g., ChatGLM, RWKV.
3.  **Custom Transformer Models**: Demonstrating adaptability.
4.  **vLLM Integration**: Specifically testing layer detection within models loaded via vLLM.
5.  **Hook Registration Compatibility**: Ensuring layers found can be used for PyTorch hook registration, crucial for some representation engineering tasks. 