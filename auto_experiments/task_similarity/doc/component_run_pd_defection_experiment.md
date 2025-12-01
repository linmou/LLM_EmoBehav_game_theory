# Component: `run_pd_defection_experiment.py`

## Purpose and Responsibility

This module is the **main contrastive defection vector training entry point** (Prisoner's Dilemma is currently the default dataset). It is responsible for:

- Loading game-theory scenarios (currently Prisoner's Dilemma) and building paired prompts.  
- Training per-layer defection vectors using span-based hidden representations.  
- Evaluating the model's defection behavior on held-out test prompts, with and without steering.  
- Writing all artifacts (split manifest, vectors, metrics, summary JSON) under `auto_experiments/task_similarity/results/`.

It is typically invoked via:

```bash
python -m auto_experiments.task_similarity.run_pd_defection_experiment \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --output_dir auto_experiments/task_similarity/results \
  --max_length 256 \
  --batch_size 8 \
  --seed 0 \
  --intensity 1.0 \
  --middle_third_only
```

## Key Helpers

### `_token_id`

```python
def _token_id(tokenizer, token_str: str) -> int:
    ids = tokenizer(token_str, add_special_tokens=False).input_ids
    if len(ids) != 1:
        raise ValueError(f"Token '{token_str}' splits into {ids}")
    return ids[0]
```

- Ensures that option labels like `"A"` and `"B"` correspond to **single tokens** for logit comparison.  
- Raises a `ValueError` if that assumption fails, avoiding silent misalignment.

### `_decision_rate`

```python
def _decision_rate(
    model,
    tokenizer,
    pairs: Sequence[PromptPair],
    label_to_token: Dict[str, int],
    batch_size: int = 8,
    max_length: int = 256,
) -> float:
    ...
```

Behavior:

- Builds inference prompts for each `PromptPair` using `build_inference_prompt`.  
- Tokenizes prompts and runs the model to obtain logits.  
- For each example:  
  - Compares final logits for token IDs corresponding to `"A"` vs `"B"`.  
  - Counts a "defect decision" if the higher-logit label matches `pair.meta.defect_label`.

Returns the fraction of test pairs where the model prefers the defection label.

This metric is used both for baseline and steered behavior evaluation.

### `_register_control_hook`

```python
def _register_control_hook(
    layer: nn.Module, vec: np.ndarray, intensity: float
):
    vec_t = torch.tensor(vec * intensity, device=next(layer.parameters()).device)

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            ctrl = vec_t.to(hidden.dtype).view(1, 1, -1)
            hidden = hidden + ctrl
            return (hidden,) + output[1:]
        ctrl = vec_t.to(output.dtype).view(1, 1, -1)
        return output + ctrl

    return layer.register_forward_hook(hook)
```

- Adds a constant steering vector to a transformer block's output on every forward pass.  
- Handles both:
  - Layers that return a single tensor.  
  - Layers that return tuples `(hidden, ...)` (common in HF transformer implementations).

This is the central mechanism for applying the PD defection direction.

## Core Training Function: `train_pd_repreader`

```python
def train_pd_repreader(
    model: Any,
    tokenizer: Any,
    train_data: Dict[str, Any],
    test_data: Dict[str, Any],
    hidden_layers: Sequence[int],
    batch_size: int,
    max_length: int,
    span_mode: str = "assistant",
) -> Tuple[Any, Dict[int, float], Dict[int, np.ndarray]]:
    ...
```

High-level algorithm per layer:

1. **Representation extraction**  
   - Call `collect_answer_means` on `train_data["data"]` and `test_data["data"]` using the specified `hidden_layers` and `span_mode` (usually `"option"`).  
   - Produces `train_hiddens[layer]` and `test_hiddens[layer]` with shape `(N, hidden_dim)`.

2. **Sanity checks**  
   - Number of hidden vectors equals number of prompts.  
   - The flattening of `train_data["labels"]` matches the number of prompts.  
   - The number of examples is even (so `[pos, neg, pos, neg, ...]` pairs can be formed).

3. **PCA on pairwise differences**  
   - Construct `diffs = pos_train - neg_train`, shape `(num_pairs, hidden_dim)`.  
   - Center: `diffs_centered = diffs - diffs.mean(axis=0, keepdims=True)`.  
   - Compute SVD: `U, S, Vt = np.linalg.svd(diffs_centered, full_matrices=False)`.  
   - Set `direction = Vt[0]`, the first principal component.

4. **Orientation and accuracy**  
   - Project test features: `scores = H_test @ direction`.  
   - Group into `[pos_score, neg_score]` pairs.  
   - Compute accuracy for both orientations:
     - `acc_plus = mean(pos_score > neg_score)`  
     - `acc_minus = mean(pos_score < neg_score)`  
   - Choose the sign (`+1` or `-1`) that yields higher accuracy.  
   - Store `layer_vectors[layer] = direction * sign` and `layer_acc[layer] = acc`.

The function returns:

- `rep_reader`: currently `None`; kept only for API compatibility.  
- `layer_acc`: per-layer validation accuracy on PD test data.  
- `layer_vectors`: per-layer defection vectors.

## Orchestration: `run`

```python
def run(
    model_path: str,
    output_dir: Path,
    max_length: int = 256,
    batch_size: int = 8,
    seed: int = 0,
    intensity: float = 1.0,
    max_pairs: int | None = None,
    middle_third_only: bool = False,
    behavior_intensities: Sequence[float] | None = None,
) -> Dict:
    ...
```

Main steps:

1. **Data loading and split**  
   - Build contrastive pairs via `build_pd_pair_bundle(dataset_path, seed=seed)` using the Prisoner's Dilemma JSON.  
   - Optionally truncate to `max_pairs`.  
   - Convert to RepReader datasets with `build_repreader_dataset`.  
   - Record the **train/test indices** relative to the underlying JSON file and compute integrity hashes:

     - `dataset_sha256`: SHA-256 hash of the entire JSON file.  
     - `entry_hashes[idx]`: SHA-256 of the JSON for entry `idx` (train or test).

   - Persist these in `split_manifest.json` under a model-specific root:

     ```text
     auto_experiments/task_similarity/results/{model_name}/split_manifest.json
     ```

     with structure:

     ```json
     {
       "dataset_path": "data_creation/.../Prisoners_Dilemma_all_data_samples.json",
       "dataset_sha256": "...",
       "split_seed": 0,
       "train_ratio": 0.5,
       "max_pairs": null,
       "train_indices": [0,3,5,...],
       "test_indices": [1,2,4,...],
       "entry_hashes": {
         "0": "hash_of_entry_0",
         "1": "hash_of_entry_1"
       }
     }
     ```

2. **Model loading**  
   - Load tokenizer and model from `model_path`, with `torch_dtype=torch.float16` and `device_map="auto"`.

3. **Layer selection**  
   - Determine the number of hidden layers from `model.config.num_hidden_layers`.  
   - If `middle_third_only`:
     - Use the middle third `range(num_layers // 3, (2 * num_layers) // 3)`.  
   - Else:
     - Use all layers.

4. **Train defection vectors**  
   - Call `train_pd_repreader` with the chosen `control_layers`.  
   - Select `best_layer` by max `layer_acc`.  
   - Save all `layer_vectors` under a model-specific directory:

     ```text
     auto_experiments/task_similarity/results/{model_name}/layer_vectors/layer_{L}.npy
     ```
   - Save `layer_accuracies` and `best_layer` / `best_accuracy` to `layer_metrics.json`.

5. **Baseline PD behavior**  
   - Build `label_to_token` for `"A"` and `"B"` using `_token_id`.  
   - Compute `base_rate = _decision_rate(model, tokenizer, test_pairs, label_to_token, ...)`.

6. **Per-layer behavior evaluation**  
   - For each layer and each intensity in `behavior_intensities` (defaults to `[0.5, 1.0, 1.5, 2.0]`):  
     - Register a hook via `_register_control_hook(layer_module, vec, inten)`.  
     - Compute defection rate via `_decision_rate(...)`.  
     - Remove the hook.  
     - Record rates in `per_layer_behavior[layer][intensity]`.

7. **Result writing**  
   - Create `run_dir = output_dir / f"{model_name}_{timestamp}"`.  
   - Save:
     - `result.json` with configuration, metrics, and behavior results.  
     - `best_vector.npy` containing the chosen layer's defection vector.  
   - At this point, downstream behavior runners can rely on **two stable artifacts**:
     - `split_manifest.json` for the exact train/test split.  
     - `layer_vectors/` for per-layer directions.

The returned dict mirrors `result.json`.

## Dependencies and Interactions

- **Internal**:
  - `pd_data.build_pd_pair_bundle`, `build_repreader_dataset`.  
  - `pd_prompt_builder.build_inference_prompt`.  
  - `pd_hidden_extractor.collect_answer_means`.

- **External**:
  - HuggingFace Transformers for model and tokenizer.  
  - NumPy and Torch for numerical work.

- **Downstream**:
  - `run_pd_defection_pd_behavior` expects:
    - `split_manifest.json` under `auto_experiments/task_similarity/results/{model_name}` to recover the test split, and  
    - per-layer vectors in `layer_vectors/` for steering.  
  - `compute_pd_delta` and future delta-activation tooling can reuse `best_vector.npy` or any `layer_{L}.npy` as steering vectors.

## Potential Issues and Improvements

- The code assumes that `model.model.layers` exists and matches `num_hidden_layers`; this is true for Qwen and many HF models but may need adaptation for others.

- Behavior evaluation on PD prompts currently uses a *single-token* comparison on `"A"` vs `"B"`. If tokenization changes (for example, multi-token labels), `_token_id` will fail loudly, but `_decision_rate` logic would need a redesign.

- The current implementation assumes that `BenchmarkItem.id` in game-theory benchmarks matches the **index** of the original scenario in the JSON file. This is true for Prisoner's Dilemma today, but other datasets should follow the same convention (or expose an explicit `source_index` in metadata) to allow reuse of `split_manifest.json`.

## Usage Example (Programmatic)

```python
from pathlib import Path
from auto_experiments.task_similarity.run_pd_defection_experiment import run

result = run(
    model_path="/data/home/.../Qwen2.5-0.5B-Instruct",
    output_dir=Path("auto_experiments/task_similarity/results"),
    max_length=256,
    batch_size=8,
    seed=0,
    intensity=1.0,
    max_pairs=None,
    middle_third_only=True,
    behavior_intensities=[0.5, 1.0, 1.5, 2.0],
)

print("Best PD layer:", result["best_layer"], "accuracy:", result["best_accuracy"])
print("Base defection rate:", result["base_defect_rate"])
```

This mirrors the CLI entry point but is suitable for use in notebooks or higher-level orchestration scripts.
