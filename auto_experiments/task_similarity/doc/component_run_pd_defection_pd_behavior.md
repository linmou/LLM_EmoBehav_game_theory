# Component: `run_pd_defection_pd_behavior.py`

## Purpose and Responsibility

This module evaluates how a previously trained **contrastive defection direction**
(currently trained on Prisoner's Dilemma) affects behavior in the
`game_theory` benchmark (task type `Prisoners_Dilemma` by default).

It:

- Loads an activation spec that points to a specific vector training run.  
- Builds a `GameTheoryDataset` using `GameBenchmarkPromptWrapper`.  
- Restricts evaluation strictly to the **test split** recorded by the training run.  
- Applies the learned directions at selected layers (single layer or middle third).  
- Measures defection ratios across steering intensities.

This module is the bridge between contrastive vector training and game-theory behavior evaluation.

## Activation Spec Handling (Dataset-Agnostic)

### Dataclass: `ActivationSpec`

```python
@dataclass
class ActivationSpec:
    train_run_dir: Path
    span_mode: str
    best_layer: int
    best_layer_accuracy: float
    split_seed: int
    max_pairs: int | None
```

Fields:

- `train_run_dir`: One specific training run directory produced by
  `run_pd_defection_experiment.py` (or future contrastive trainers).  
- `span_mode`: The representation span used during training
  (e.g., `"option"` for option-text-only mean).  
- `best_layer`: Layer index with the highest validation accuracy.  
- `best_layer_accuracy`: The corresponding accuracy, for reporting.  
- `split_seed`: Seed used for the original dataset split.  
- `max_pairs`: Optional cap on number of pairs used during training.

### `_load_activation_spec`

```python
def _load_activation_spec(path: Path) -> ActivationSpec:
    raw = json.loads(path.read_text())
    train_run_dir = Path(raw["train_run_dir"])
    span_mode = str(raw.get("span_mode", "option"))
    best_layer = int(raw.get("best_layer", 0))
    best_layer_accuracy = float(raw.get("best_layer_accuracy", 0.0))
    split_seed = int(raw.get("split_seed", 0))
    max_pairs = raw.get("max_pairs")
    if max_pairs is not None:
        try:
            max_pairs = int(max_pairs)
        except Exception:
            max_pairs = None
    return ActivationSpec(
        train_run_dir=train_run_dir,
        span_mode=span_mode,
        best_layer=best_layer,
        best_layer_accuracy=best_layer_accuracy,
        split_seed=split_seed,
        max_pairs=max_pairs,
    )
```

Example spec (Prisoner's Dilemma, Qwen2.5‑0.5B, Iter 8):

```json
{
  "train_run_dir": "auto_experiments/task_similarity/results/Qwen2.5-0.5B-Instruct_20251129_211403",
  "span_mode": "option",
  "best_layer": 8,
  "best_layer_accuracy": 0.5515055467511886,
  "split_seed": 0,
  "max_pairs": null
}
```

The behavior module does not need to know how the vectors were trained; it relies
on the training run's artifacts (split manifest + layer vectors).

## Dataset Alignment with the Training Test Split

Instead of reconstructing a split from the raw JSON, the behavior runner honors
the split recorded by the training module.

### Split Manifest

`run_pd_defection_experiment.py` writes a manifest under:

```text
auto_experiments/task_similarity/results/{model_name}/split_manifest.json
```

with structure:

```json
{
  "dataset_path": "data_creation/scenario_creation/langgraph_creation/Prisoners_Dilemma_all_data_samples.json",
  "dataset_sha256": "...",
  "split_seed": 0,
  "train_ratio": 0.5,
  "max_pairs": null,
  "train_indices": [0, 3, 5, ...],
  "test_indices": [1, 2, 4, ...],
  "entry_hashes": {
    "0": "hash_of_entry_0",
    "1": "hash_of_entry_1"
  }
}
```

Key idea: `train_indices` and `test_indices` are indices into the original
dataset JSON list; game-theory datasets must preserve this indexing in
`BenchmarkItem.id`.

### Restricting the Benchmark Dataset

In `run(...)`, after constructing `GameTheoryDataset`, we restrict it:

```python
output_root = activation.train_run_dir.parent
model_name = Path(model_path).name
model_root = output_root / model_name
split_manifest_path = model_root / "split_manifest.json"
manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
test_indices = set(int(i) for i in manifest.get("test_indices", []))

filtered_items = []
for item in dataset.items:
    try:
        idx = int(item.id)
    except Exception:
        continue
    if idx in test_indices:
        filtered_items.append(item)
dataset.items = filtered_items
split_info = {
    "n_test_indices": len(test_indices),
    "n_dataset_items": len(filtered_items),
}
```

Assumptions:

- `GameTheoryDataset` sets `BenchmarkItem.id` to the original scenario index.  
- The raw dataset used by the benchmark matches `dataset_path` from the manifest.

If these hold, behavior evaluation is guaranteed to use the same test split as
vector training.

## Behavior Evaluation Logic

### `_compute_defect_ratio`

```python
def _compute_defect_ratio(
    model: Any,
    tokenizer: Any,
    dataset: GameTheoryDataset,
    max_length: int,
    batch_size: int,
    generation_config: Dict[str, Any],
) -> float:
    ...
```

Algorithm:

1. Iterate through the filtered `dataset` in batches.
2. Tokenize `prompt` strings and call `model.generate` with the supplied
   `generation_config`.  
3. Decode responses using `tokenizer.batch_decode`.  
4. For each entry:
   - Extract options via `GameTheoryDataset._extract_options_from_prompt(prompt)`.  
   - Extract a choice ID via
     `GameTheoryDataset._extract_option_from_response(response, options)`.  
   - If extraction fails, skip.  
   - Otherwise, increment `valid_count` and `defect_count` if `choice_id == 2`
     (the defection option in the Prisoner's Dilemma benchmark).

Returns:

- `defect_count / valid_count`, or `NaN` if no valid choices were parsed.

## Layer Selection and Steering Vectors

### Control Layers

```python
num_layers = getattr(model.config, "num_hidden_layers", None)
if middle_third:
    start = num_layers // 3
    end = (2 * num_layers) // 3
    control_layers = list(range(start, end))
else:
    control_layers = [activation.best_layer]
```

- Single-layer mode: steer only at the best layer from training.  
- Middle-third mode: steer at all middle-third layers.

### Loading Layer Vectors

Vectors are stored under:

```text
auto_experiments/task_similarity/results/{model_name}/layer_vectors/layer_{L}.npy
```

In `run(...)`:

```python
vec_dir = model_root / "layer_vectors"
if middle_third:
    layer_vectors = {
        lyr: np.load(vec_dir / f"layer_{lyr}.npy") for lyr in control_layers
    }
else:
    base_vec = np.load(vec_dir / f"layer_{activation.best_layer}.npy")
```

Steering:

- Middle-third: each layer uses its own vector `layer_{lyr}.npy`.  
- Single-layer: all steering uses `base_vec` at `best_layer`.

## Orchestration: `run`

```python
def run(
    model_path: str,
    benchmark_config_path: Path,
    activation_spec_path: Path,
    output_dir: Path,
    intensities: Sequence[float],
    max_length: int = 256,
    batch_size: int = 8,
    seed: int = 0,
    middle_third: bool = False,
) -> Dict[str, Any]:
    ...
```

High-level flow:

1. Seed Torch and NumPy.  
2. Load `ActivationSpec`, `BenchmarkConfig`, and generation config.  
3. Load HF model and tokenizer.  
4. Determine `control_layers` (single or middle-third).  
5. Build `GameTheoryDataset` + `GameBenchmarkPromptWrapper`.  
6. Restrict dataset using `split_manifest.json` test indices.  
7. Compute baseline defection ratio at intensity `0.0`.  
8. For each non-zero intensity:
   - Register per-layer hooks via `_register_control_hook`.  
   - Compute defection ratio via `_compute_defect_ratio`.  
   - Remove hooks.  
9. Write a summary JSON under `output_dir` with:

   ```json
   {
     "model_path": "...",
     "benchmark_config": "...",
     "activation_spec": "...",
     "timestamp": "...",
     "seed": 0,
     "benchmark_name": "game_theory",
     "task_type": "Prisoners_Dilemma",
     "train_run_dir": "...",
     "best_layer": 8,
     "best_layer_accuracy": 0.55,
     "span_mode": "option",
     "control_layers": [...],
     "intensities": [0.0, 0.5, 1.0, 1.5, 2.0],
     "defect_ratio": {...},
     "n_items": 1262,
     "split_info": {
       "n_test_indices": 1262,
       "n_dataset_items": 1262
     }
   }
   ```

The returned dict matches this JSON.

## Usage Example (CLI)

```bash
python -m auto_experiments.task_similarity.run_pd_defection_pd_behavior \
  --model /data/home/.../Qwen2.5-0.5B-Instruct \
  --benchmark_config auto_experiments/task_similarity/config/pd_behavior_game_theory.yaml \
  --activation_spec auto_experiments/task_similarity/config/pd_defection_iter8_qwen2.5_0.5B.json \
  --intensities 0.0,0.5,1.0,1.5,2.0 \
  --batch_size 100 \
  --max_length 256 \
  --middle_third \
  --output_dir auto_experiments/task_similarity/results/pd_behavior
```

This is the recommended way to quantify how much a trained defection vector
shifts behavior in the game-theory benchmark, while ensuring the **behavior
test set matches the original training split**.

