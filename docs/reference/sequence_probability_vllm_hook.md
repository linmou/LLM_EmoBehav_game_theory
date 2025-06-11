# Sequence Probability vLLM Hook

## Current Working Implementation

- **New Approach**: Direct logprobs extraction (no hooks)
- **Documentation**: [README_sequence_prob_fix.md](../code_readme/neuro_manipulation/repe/README_sequence_prob_fix.md) 
- **Tests**: `neuro_manipulation/repe/tests/test_sequence_prob_tp_consistency.py`

# This will return empty results [] instead of probabilities
seq_prob_hook = SequenceProbVLLMHook(llm, tokenizer)
results = seq_prob_hook.get_log_prob([prompt], target_sequences)
# Returns: [] (empty list)
```


For the current working implementation that actually returns results with vLLM v1.x, see the [Sequence Probability Fix documentation](../code_readme/neuro_manipulation/repe/README_sequence_prob_fix.md).

## Architecture Components

### Hook Function
- `hook_fn_sequence_prob()`: Captures logits from language model head output
- Stores logits with rank information for tensor parallel aggregation
- Handles both single tensor and tuple outputs

### RPC Functions
- `_register_lm_head_hook_rpc()`: Registers hooks on language model head across workers
- `_set_sequence_prob_state_rpc()`: Sets state for sequence probability capture
- `_reset_sequence_prob_state_rpc()`: Cleans up state after capture
- `_get_captured_logits_rpc()`: Retrieves captured logits from workers

### Main Class Methods
- `__init__()`: Initializes hook and registers on language model head
- `get_log_prob()`: Main method for calculating sequence probabilities
- `_aggregate_logits_across_ranks()`: Handles tensor parallel logit aggregation
- `_calculate_sequence_probabilities()`: Computes final probability metrics

## Tensor Parallel Handling

The hook automatically handles different tensor parallel configurations:

1. **Single Rank (TP=1)**: Uses logits directly from the single worker
2. **Multi Rank (TP>1)**: Concatenates logits along vocabulary dimension from all ranks

## Return Format

The `get_log_prob()` method returns a list of dictionaries, each containing:

- `sequence`: The target sequence string
- `log_prob`: Log probability of the sequence
- `prob`: Probability of the sequence (exp(log_prob))
- `perplexity`: Perplexity of the sequence (exp(-log_prob))
- `num_tokens`: Number of tokens in the sequence

## Integration with Neural Manipulation

This hook complements the existing neural manipulation capabilities:

- Works alongside `RepControlVLLMHook` for comprehensive model analysis
- Can be used to measure the effect of neural interventions on sequence probabilities
- Provides quantitative metrics for evaluating manipulation effectiveness

## Performance Considerations

- **Memory**: Logits are stored temporarily during calculation
- **Computation**: Scales with sequence length and vocabulary size
- **Communication**: RPC overhead for tensor parallel setups
- **GPU Memory**: Ensure sufficient VRAM for model + logit storage

## Testing

Comprehensive unit tests are provided covering:

- Hook function behavior with various inputs
- RPC function communication
- Tensor parallel logit aggregation
- Main class functionality and error handling
- Edge cases and error conditions

Run tests with:
```bash
cd neuro_manipulation/repe/tests
python -m unittest test_sequence_prob_vllm_hook.py
```

## Related Documentation

- [vLLM Hook Implementation](vllm_hook_implementation.md)
- [Model Layer Detector](model_layer_detector.md)
- [vLLM Compatibility](vllm_compatibility.md)

## Dependencies

- PyTorch
- vLLM
- Transformers
- NumPy
- Python 3.8+

## Limitations

- Currently supports only forward pass probability calculation
- Requires models with accessible language model heads
- Hook removal functionality is placeholder (manual cleanup required)
- Memory usage scales with vocabulary size for logit storage 