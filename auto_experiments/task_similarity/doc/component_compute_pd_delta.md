# Component: `compute_pd_delta.py`

## Purpose and Responsibility

`compute_pd_delta.py` measures how a PD defection vector changes **mean hidden activations** on a fixed set of generic text probes. It is a diagnostic tool rather than a core part of the PD training pipeline.

The main responsibilities are:

- Collect baseline mean hidden states per layer on generic probes.  
- Apply the defection vector (at a chosen layer or across the middle third of layers) via a forward hook.  
- Collect steered mean hidden states and compute deltas.  
- Save `baseline.npz`, `steered.npz`, and `delta.npz` for offline analysis.

## Core Helper: `_collect_hidden`

```python
def _collect_hidden(
    model,
    tokenizer,
    prompts: Sequence[str],
    layers: Sequence[int],
    batch_size: int = 8,
    max_length: int = 256,
) -> Dict[int, np.ndarray]:
    ...
```

Algorithm:

1. Iterate over prompts in batches.  
2. Tokenize with padding/truncation, `add_special_tokens=False`.  
3. Run the model with `output_hidden_states=True`.  
4. For each target layer:  
   - Obtain `hidden_states[layer + 1]` (skipping the embedding layer).  
   - Apply the attention mask to zero out padding tokens.  
   - Mean-pool over the sequence dimension for each example:
     ```python
     pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1)
     ```  
   - Append pooled vectors.  
5. At the end, concatenate per-batch pooled vectors and average across prompts:
   ```python
   return {k: torch.cat(v, dim=0).mean(dim=0).numpy() for k, v in out.items()}
   ```

The result is a mapping `layer -> (hidden_dim,)` of mean activations over all prompts.

## Steering Hook: `_register_control_hook`

```python
def _register_control_hook(layer_module, vec: np.ndarray, intensity: float):
    vec_t = torch.tensor(vec * intensity, device=next(layer_module.parameters()).device)

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            ctrl = vec_t.to(hidden.dtype).view(1, 1, -1)
            hidden = hidden + ctrl
            return (hidden,) + output[1:]
        ctrl = vec_t.to(output.dtype).view(1, 1, -1)
        return output + ctrl

    return layer_module.register_forward_hook(hook)
```

Same pattern as in `run_pd_defection_experiment`, but tailored for this module.

## Control Layer Resolution

```python
def resolve_control_layers(num_layers: int, layer: Union[int, None], use_middle_third: bool) -> List[int]:
    if use_middle_third:
        return select_middle_third_layers(num_layers)
    if layer is None:
        raise ValueError("Specify layer or enable use_middle_third")
    return [int(layer)]
```

- Uses `delta_activation_engine.backends.hf.select_middle_third_layers` when `use_middle_third` is `True`.  
- Otherwise, expects a single `layer` index.

## Orchestration: `run_delta`

```python
def run_delta(
    model_path: str,
    vector_path: Path,
    layer: Union[int, None],
    use_middle_third: bool,
    intensity: float,
    output_dir: Path,
    max_length: int = 256,
    batch_size: int = 8,
    seed: int = 0,
) -> Dict:
    ...
```

Steps:

1. Seed NumPy and Torch and configure threading for reproducibility.  
2. Call `get_generic_probes()` to obtain a fixed list of text prompts.  
3. Load tokenizer and model from `model_path`.  
4. Determine control layers via `resolve_control_layers`.  
5. Collect baseline activations via `_collect_hidden`.  
6. Load the defection vector from `vector_path`.  
7. Register hooks on each control layer with `_register_control_hook`.  
8. Collect steered activations via `_collect_hidden`.  
9. Remove hooks.  
10. Compute per-layer deltas with:
    ```python
    delta = {k: compute_delta(baseline[k], steered[k]) for k in control_layers}
    ```
11. Create a timestamped directory under `output_dir` and save:
    - `baseline.npz`, `steered.npz`, `delta.npz` with keys being the layer indices (as strings).  
    - `metadata.json` containing model path, vector path, control layers, intensity, seed, timestamp, and a hash of the prompt set.

The function returns this metadata dict.

## CLI Entry Point: `main`

```bash
python -m auto_experiments.task_similarity.compute_pd_delta \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --vector_path auto_experiments/task_similarity/results/.../best_vector.npy \
  --layer 8 \
  --intensity 1.5 \
  --output_dir auto_experiments/task_similarity/results/delta
```

Options:

- `--model`: HF model path (required).  
- `--vector_path`: path to the steering vector (required).  
- `--layer`: target layer (required unless `--middle_third` is set).  
- `--middle_third`: steer all middle-third layers instead of a single one.  
- `--intensity`: vector scaling before application.  
- `--max_length`, `--batch_size`, `--seed`: standard hyperparameters.

## Dependencies and Interactions

- **Upstream**: The steering vector is typically `best_vector.npy` from a PD training run.  
- **External**:  
  - HuggingFace Transformers for model and tokenizer.  
  - `delta_activation_engine` for probes and middle-third layer selection.

This module does not depend on PD data or the benchmark engine and can be used in isolation as a generic "vector delta" analyzer.

