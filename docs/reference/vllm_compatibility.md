# vLLM Compatibility with Representation Engineering

## Problem: Hidden States Access in vLLM

When using the `vllm.LLM` model for generating text, we encountered a compatibility issue with the representation engineering (RepE) pipeline, specifically `rep-reading`. The issue was that vLLM models do not expose hidden states in the same way that HuggingFace models do, which was required for the `rep-reading` pipeline to build emotion readers.

The original implementation in `model_utils.py` was loading the model twice:

1. First as a `vllm.LLM` instance for efficient generation
2. Then as a `transformers.AutoModelForCausalLM` for accessing hidden states

This approach worked but led to doubled VRAM usage, as the model was effectively loaded twice in GPU memory.

## Current Workaround: Temporary Model for RepE (Further Development Planned)

> **Note:** This section describes our current approach to enable Representation Engineering with vLLM. We are actively exploring more integrated solutions, as outlined in the "Future Improvements" section below.

Our solution keeps the primary vLLM model for generation but creates a temporary HuggingFace model only when needed for the representation reading step:

1. We use the primary vLLM model for all generation tasks
2. Only when building emotion readers (in `load_emotion_readers`), we:
   - Create a temporary HuggingFace model with aggressive memory optimization
   - Use it to extract the necessary hidden states information
   - Delete the temporary model and clear CUDA cache to free memory
   - Cache the results to avoid repeating this step in future runs

This approach provides:
- Memory efficiency - the temporary model is loaded with minimal memory footprint options
- Performance - we keep the highly optimized vLLM for inference
- Compatibility - we can still use the representation engineering techniques

## Implementation Details

The implementation applies several VRAM optimization techniques to the temporary model:

```python
hf_model = AutoModelForCausalLM.from_pretrained(
    config['model_name_or_path'], 
    torch_dtype=torch.float16,  # Use half precision
    device_map="auto",
    token=True, 
    trust_remote_code=True,
    max_memory={0: "2GiB"},     # Limit memory usage
    offload_folder="offload_folder",  # Enable weight offloading
    offload_state_dict=True,    # Offload weights not in use
    low_cpu_mem_usage=True      # Optimize CPU memory usage
).eval()
```

After the emotion readers are built, we explicitly free the memory:

```python
# Clean up the temporary model to free memory
del hf_model
torch.cuda.empty_cache()
```

## Future Improvements

For a more complete solution, we could:

1. Implement a full `RepReadingVLLM` class that directly extracts hidden states from vLLM models without needing the temporary HuggingFace model.
2. Add more robust caching for the emotion readers to avoid rebuilding them when the model is restarted.
3. Investigate direct hooks into the vLLM model architecture to access hidden states in a more memory-efficient way. 