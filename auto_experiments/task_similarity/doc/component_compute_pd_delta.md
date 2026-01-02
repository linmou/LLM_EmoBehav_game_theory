# Component: `compute_pd_delta.py`

## Purpose and Responsibility

`compute_pd_delta.py` measures how a PD defection vector changes **final-token hidden activations** on a fixed set of generic probe prompts. It is a diagnostic tool rather than a core part of the PD training pipeline.

The main responsibilities are:

- Collect baseline **final-token** hidden states for **all layers** on generic probes.  
- Apply the defection vector (at a chosen layer or across the middle third of layers) via a forward hook.  
- Collect steered final-token hidden states and compute deltas.  
- Save `baseline.npz`, `steered.npz`, and `delta.npz` for offline analysis.

## Probe “Dataset”

The inputs are taken from `delta_activation_engine/prompts/probes_texts.py:get_generic_probes()`.

Important: the probe strings are used **as-is** (no `{task}` / `{input}` formatting is applied here). If you want fully-rendered prompts, the dataset layer should produce them.

## Core Helper: `_collect_final_token_hidden`

This helper returns a dict mapping `layer_idx -> (hidden_dim,)` by:

1. Running the model with `output_hidden_states=True`.
2. For each prompt, selecting the **last non-pad token** (using `attention_mask.sum(1) - 1`).
3. For each layer, taking that token’s hidden state.
4. Averaging across prompts.

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
5. Set `measurement_layers = list(range(num_layers))` (all layers).  
6. Collect baseline activations via `_collect_final_token_hidden` for all measurement layers.  
7. Load the defection vector(s) from `vector_path`.  
8. Register hooks on each control layer with `_register_control_hook`.  
9. Collect steered activations via `_collect_final_token_hidden` for all measurement layers.  
9. Remove hooks.  
10. Compute per-layer deltas for all measurement layers:
    `delta[layer] = steered[layer] - baseline[layer]`.
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
