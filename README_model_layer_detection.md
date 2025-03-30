# Automatic Model Layer Detection

This utility provides a general-purpose solution for automatically detecting transformer layers in any model architecture without hardcoding model-specific paths.

## Key Features

- **Model-Agnostic**: Works with any transformer-based model architecture (GPT, Llama, Mistral, ChatGLM, RWKV, etc.)
- **Zero Configuration**: No need to specify model-specific paths or patterns
- **Flexible Detection**: Uses intelligent heuristics to identify transformer layers
- **Breadth-First Search**: Efficiently traverses the model tree structure

## How It Works

The layer detection algorithm works using these key techniques:

1. **BFS Traversal**: Explores the model structure level by level to find layer modules
2. **Transformer Layer Identification**: Detects modules that have attention components and other transformer features
3. **Candidate Prioritization**: Prioritizes modules named "layers" and with shorter paths
4. **Fallback Mechanism**: Falls back to finding any ModuleList with consistent module types

## Usage Example

```python
from model_layer_detector import ModelLayerDetector

# Load any transformer model
model = AutoModel.from_pretrained("your-model-name")

# Detect transformer layers automatically
layers = ModelLayerDetector.get_model_layers(model)

# Use the detected layers
print(f"Found {len(layers)} layers")
```

## Testing

The included test script verifies the functionality across multiple model architectures:

- Standard models (GPT-2, OPT, Pythia)
- ChatGLM models (THUDM/glm-4-9b)
- RWKV models (BlinkDL/rwkv-4-raven)

Run the tests using:

```bash
python test_model_layer_detection.py
```

## When to Use

This utility is particularly useful when:

1. Working with multiple different model architectures
2. Building model-agnostic tools and pipelines
3. Exploring new or custom model architectures
4. Avoiding hardcoded model-specific paths that break with model updates 