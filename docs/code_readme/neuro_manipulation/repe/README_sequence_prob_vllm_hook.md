# [DEPRECATED] SequenceProbVLLMHook Documentation

> **⚠️ DEPRECATED**: This hook-based approach does not work with vLLM v1.x due to engine optimizations that bypass PyTorch forward hooks. See [README_sequence_prob_fix.md](./README_sequence_prob_fix.md) for the current working implementation.

## Migration Notice
The hook-based sequence probability calculation has been replaced with a direct logprobs extraction approach. The new implementation:
- Works with vLLM v1.x optimized execution paths  
- Provides more reliable results
- Handles tensor parallel configurations correctly
- Fixes token encoding issues (space-prefixed tokens)
- Properly handles vLLM parameter limits

**For current usage, please refer to the new documentation: [README_sequence_prob_fix.md](./README_sequence_prob_fix.md)**

---

## Historical Documentation (Preserved for Reference)

### Overview

The `SequenceProbVLLMHook` class provides a tensor parallel-aware hook system for capturing sequence probabilities from vLLM language models. It hooks into the language model head to capture logits and calculate log probabilities for target sequences.

## Features

- **Tensor Parallel Support**: Automatically detects and handles tensor parallel configurations
- **Logit Aggregation**: Properly aggregates logits across tensor parallel ranks
- **RPC Communication**: Uses vLLM's collective RPC system for distributed hook management
- **Probability Calculation**: Computes log probabilities, probabilities, and perplexity for target sequences
- **Clean State Management**: Automatic state setup and cleanup for reliable operation

## Architecture

### Hook Function
- `hook_fn_sequence_prob()`: Captures logits from the language model head output
- Stores logits with rank information for tensor parallel aggregation
- Handles both single tensor and tuple outputs

### RPC Functions
- `_register_lm_head_hook_rpc()`: Registers hooks on language model head across workers
- `_set_sequence_prob_state_rpc()`: Sets state for sequence probability capture
- `_reset_sequence_prob_state_rpc()`: Cleans up state after capture
- `_get_captured_logits_rpc()`: Retrieves captured logits from workers

### Main Class
- `SequenceProbVLLMHook`: Main interface for sequence probability calculation
- Handles tensor parallel detection and hook management
- Provides `get_log_prob()` method for calculating sequence probabilities

## Usage

### Basic Usage

```python
from vllm import LLM
from transformers import AutoTokenizer
from neuro_manipulation.repe.sequence_prob_vllm_hook import SequenceProbVLLMHook

# Initialize model and tokenizer
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model=model_name, tensor_parallel_size=1)

# Create hook
seq_prob_hook = SequenceProbVLLMHook(llm, tokenizer)

# Calculate probabilities
prompt = "The capital of France is"
target_sequences = ["Paris", "London", "Berlin"]
results = seq_prob_hook.get_log_prob([prompt], target_sequences)

# View results
for result in results:
    print(f"Sequence: '{result['sequence']}'")
    print(f"Log Probability: {result['log_prob']:.4f}")
    print(f"Probability: {result['prob']:.6f}")
    print(f"Perplexity: {result['perplexity']:.4f}")
```

### Advanced Usage with Multiple Prompts

```python
# Multiple prompts
prompts = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "The chemical symbol for gold is"
]

target_sequences = ["Paris", "Jupiter", "Au"]

# Calculate probabilities for each prompt-sequence pair
for i, prompt in enumerate(prompts):
    results = seq_prob_hook.get_log_prob([prompt], [target_sequences[i]])
    print(f"Prompt: '{prompt}' -> '{target_sequences[i]}'")
    print(f"Log Probability: {results[0]['log_prob']:.4f}")
```

## API Reference

### SequenceProbVLLMHook Class

#### Constructor
```python
SequenceProbVLLMHook(model: LLM, tokenizer, tensor_parallel_size: int = 1)
```

**Parameters:**
- `model`: vLLM LLM instance
- `tokenizer`: Transformers tokenizer object
- `tensor_parallel_size`: Tensor parallel size (auto-detected if available)

#### Methods

##### get_log_prob()
```python
get_log_prob(text_inputs: List[str], target_sequences: List[str], **kwargs) -> List[Dict[str, float]]
```

Calculates log probabilities for target sequences given input prompts.

**Parameters:**
- `text_inputs`: List of input prompts
- `target_sequences`: List of target sequences to calculate probabilities for
- `**kwargs`: Additional arguments for vLLM SamplingParams

**Returns:**
List of dictionaries with keys:
- `sequence`: The target sequence
- `log_prob`: Log probability of the sequence
- `prob`: Probability of the sequence (exp(log_prob))
- `perplexity`: Perplexity of the sequence (exp(-log_prob))
- `num_tokens`: Number of tokens in the sequence

## Implementation Details

### Tensor Parallel Handling

The hook automatically detects tensor parallel configurations and handles logit aggregation:

1. **Single Rank (TP=1)**: Uses logits directly from the single worker
2. **Multi Rank (TP>1)**: Concatenates logits along vocabulary dimension from all ranks

### Language Model Head Detection

The hook searches for language model head modules using common naming patterns:
- `lm_head`
- `language_model_head`
- `head`
- `output_projection`
- `embed_out`

### State Management

Each hook operation follows this lifecycle:
1. Set state with target sequences and configuration
2. Run generation to trigger hook and capture logits
3. Retrieve captured logits from all workers
4. Aggregate logits across tensor parallel ranks
5. Calculate sequence probabilities
6. Clean up state

### Error Handling

- Graceful handling of missing language model heads
- Robust error recovery in RPC operations
- Proper state cleanup even on exceptions
- Detailed logging for debugging

## Limitations

- Currently supports only forward pass probability calculation
- Requires models with accessible language model heads
- Hook removal functionality is placeholder (manual cleanup required)
- Memory usage scales with vocabulary size for logit storage

## Testing

The module includes a comprehensive example in the `__main__` section that demonstrates:
- Model loading and initialization
- Hook setup and registration
- Probability calculation for multiple sequences
- Proper cleanup and resource management

## Dependencies

- PyTorch
- vLLM
- Transformers
- NumPy

## Performance Considerations

- **Memory**: Logits are stored temporarily during calculation
- **Computation**: Probability calculation scales with sequence length
- **Communication**: RPC overhead for tensor parallel setups
- **GPU Memory**: Ensure sufficient VRAM for model + logit storage

## Troubleshooting

### Common Issues

1. **Hook not registered**: Check if language model head is found
2. **RPC failures**: Verify vLLM version supports collective_rpc
3. **Memory errors**: Reduce batch size or sequence length
4. **Tensor parallel issues**: Ensure consistent TP configuration

### Debug Logging

Enable debug logging for detailed operation information:
```python
import logging
logging.getLogger('sequence_prob_vllm_hook').setLevel(logging.DEBUG)
```

## Integration with Existing Code

The `SequenceProbVLLMHook` is designed to work alongside other hooks like `RepControlVLLMHook`:

```python
# Can use both hooks on the same model
rep_control = RepControlVLLMHook(llm, tokenizer, layers=[10, 15], block_name="decoder_block", control_method="reading_vec")
seq_prob = SequenceProbVLLMHook(llm, tokenizer)

# Use them independently
controlled_output = rep_control(prompts, activations=control_vectors)
probabilities = seq_prob.get_log_prob(prompts, target_sequences)
``` 