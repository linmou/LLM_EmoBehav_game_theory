# Enhanced Sequence Probability vLLM Hook

## Overview

The enhanced `sequence_prob_vllm_hook.py` provides a comprehensive solution for:

1. **Sequence Probability Calculation**: Calculate exact log probabilities for specific target sequences
2. **Representation Control**: Modify hidden states at specific layers during inference (similar to `rep_control_vllm_hook.py`)
3. **Layer-wise Logit Recording**: Optionally record activations from specific layers for analysis

This implementation unifies all functionality into a single, tensor parallel-aware system via the `CombinedVLLMHook` class. **Backward compatibility classes are no longer present; all features are accessed through `CombinedVLLMHook`.**

## Testing

Comprehensive unit tests are provided to validate all functionality:

### Test Files

1. **`test_combined_vllm_hook.py`**: Complete test suite with comprehensive coverage
   - Tests each feature individually and in combination
   - Tests error handling and edge cases
   - Tests different operators and token position controls

2. **`test_combined_functionality_simple.py`**: Lightweight integration test
   - Quick validation of core functionality
   - Uses minimal memory requirements
   - Can be run standalone for quick checks

### Running Tests

```bash
# Run comprehensive test suite
python -m unittest neuro_manipulation.repe.tests.test_combined_vllm_hook

# Run simple integration test
python neuro_manipulation/repe/tests/test_combined_functionality_simple.py --simple

# Run simple test as unittest
python -m unittest neuro_manipulation.repe.tests.test_combined_functionality_simple

# Run all tests in the directory
python -m unittest discover neuro_manipulation/repe/tests/ -p "test_*.py"
```

### Test Coverage

The test suite covers:
- ✅ Sequence probability calculation accuracy
- ✅ Representation control with different operators (`linear_comb`, `piecewise_linear`)
- ✅ Layer-wise logit recording functionality
- ✅ Combined feature usage
- ✅ Error handling for invalid configurations
- ✅ Token position controls (`start`, `end`, specific positions)
- ✅ Memory management and cleanup
- ✅ Tensor parallel compatibility (single GPU tests)

## Key Class

### `CombinedVLLMHook`
The main and only class that provides all functionality. Features can be enabled/disabled as needed:

```python
# Enable all features
hook = CombinedVLLMHook(
    model=llm, 
    tokenizer=tokenizer,
    layers=[10, 15, 20],  # Layers for control/recording
    block_name="decoder_block",  # Target block within layers
    enable_sequence_prob=True,
    enable_rep_control=True, 
    enable_layer_logit_recording=True
)

# Sequence probability only
hook = CombinedVLLMHook(
    model=llm, 
    tokenizer=tokenizer,
    enable_sequence_prob=True,
    enable_rep_control=False,
    enable_layer_logit_recording=False
)
```

## Core Functionality

### 1. Sequence Probability Calculation

Calculate exact log probabilities for target sequences:

```python
# Initialize hook for sequence probability
hook = CombinedVLLMHook(llm, tokenizer, enable_sequence_prob=True)

# Calculate probabilities
results = hook.get_log_prob(
    text_inputs=["The capital of France is"],
    target_sequences=["Paris", "London", "Berlin"]
)

for result in results:
    print(f"'{result['sequence']}': prob={result['prob']:.6f}")
```

### 2. Representation Control

Modify hidden states at specific layers during generation:

```python
# Initialize hook for representation control
hook = CombinedVLLMHook(
    llm, tokenizer, 
    layers=[10, 15], 
    enable_rep_control=True
)

# Create control vectors (reading vectors)
control_activations = {
    10: torch.randn(4096) * 0.1,  # Layer 10 control vector
    15: torch.randn(4096) * 0.1   # Layer 15 control vector  
}

# Generate with control
outputs = hook.generate_with_control(
    text_inputs=["The capital of France is"],
    activations=control_activations,
    operator='linear_comb',  # or 'piecewise_linear'
    normalize=False,
    token_pos=None,  # Apply to all tokens
    max_new_tokens=10
)
```

### 3. Layer-wise Logit Recording

Record activations from specific layers during generation:

```python
# Initialize hook for layer recording
hook = CombinedVLLMHook(
    llm, tokenizer,
    layers=[5, 10, 15],
    enable_layer_logit_recording=True
)

# Generate with recording enabled
outputs = hook.generate_with_control(
    text_inputs=["The capital of France is"],
    record_layer_logits=True,
    max_new_tokens=5
)

# Retrieve recorded activations
layer_logits = hook.get_layer_logits()
for layer_id, recordings in layer_logits.items():
    print(f"Layer {layer_id}: {len(recordings)} recordings")
    if recordings:
        print(f"  Shape: {recordings[0]['shape']}")
```

### 4. Combined Usage

Use all features together:

```python
# Initialize with all features
hook = CombinedVLLMHook(
    llm, tokenizer,
    layers=[10, 15],
    enable_sequence_prob=True,
    enable_rep_control=True,
    enable_layer_logit_recording=True
)

# Calculate sequence probabilities
prob_results = hook.get_log_prob(
    ["The capital is"], 
    ["Paris", "London"]
)

# Generate with control and recording
control_outputs = hook.generate_with_control(
    text_inputs=["The capital is"],
    activations={10: control_vector},
    record_layer_logits=True,
    max_new_tokens=5
)

# Retrieve layer recordings
layer_data = hook.get_layer_logits()
```

## Hook Function: `hook_fn_combined`

The core hook function combines three types of processing:

1. **Representation Control**: Modifies hidden states using control vectors
2. **Logit Capture**: Captures logits from language model head for probability calculation  
3. **Layer Recording**: Records activations from specified layers

The hook checks for different state types attached to modules:
- `_rep_control_state`: For representation control
- `_sequence_prob_state`: For sequence probability calculation
- `_layer_logit_state`: For layer-wise recording

## Tensor Parallel Support

The implementation is designed to work with vLLM's tensor parallel execution:

- **Control Vector Slicing**: Automatically slices control vectors across tensor parallel ranks
- **Logit Aggregation**: Properly handles logit capture across multiple GPUs
- **State Synchronization**: Uses RPC calls to coordinate state across workers

## Configuration Options

### Representation Control
- `operator`: How to combine control vectors (`'linear_comb'`, `'piecewise_linear'`)
- `token_pos`: Which tokens to modify (`int`, `list`, `'start'`, `'end'`, `None`)
- `normalize`: Whether to normalize activations after modification
- `masks`: Optional masks for selective application

### Sequence Probability
- Automatically uses vLLM's built-in `logprobs` for efficient calculation
- Supports multiple tokenization variants (with/without leading space)
- Returns detailed probability metrics (log_prob, prob, perplexity)

### Layer Recording
- Records raw activations from specified layers
- Supports tensor parallel aggregation
- Configurable per-layer recording

## Error Handling

The implementation includes robust error handling:

- **State Validation**: Checks for required state before processing
- **Shape Compatibility**: Validates tensor shapes for tensor parallel consistency
- **Graceful Degradation**: Continues operation if individual components fail
- **Resource Cleanup**: Automatically resets states after generation

## Performance Considerations

- **Memory Efficient**: Only captures/stores data when explicitly requested
- **Parallel Execution**: Designed for efficient tensor parallel operation
- **Selective Features**: Can disable unused functionality to reduce overhead

## Example Use Cases

1. **Emotion Intervention Study**: Use representation control to inject emotion vectors while measuring sequence probabilities for different behavioral choices

2. **Layer Analysis**: Record activations from multiple layers during controlled generation to understand how interventions propagate

3. **Probability Comparison**: Compare sequence probabilities before and after applying representation control to quantify behavioral changes

4. **Debugging**: Record layer activations to debug representation control effectiveness

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Disable unused features, reduce batch size, or use gradient checkpointing
2. **Tensor Parallel Inconsistency**: Ensure control vectors match the model's hidden dimension
3. **Hook Registration Failure**: Check that the model has the expected layer structure

### Debug Logging

Enable debug logging to troubleshoot:

```python
import logging
logging.getLogger('neuro_manipulation.repe.sequence_prob_vllm_hook').setLevel(logging.DEBUG)
```

## Implementation Notes

This enhanced implementation addresses the original limitations by:
- Combining multiple hook functionalities efficiently
- Providing better tensor parallel support
- Including comprehensive error handling and logging
- Offering flexible feature selection for different use cases 