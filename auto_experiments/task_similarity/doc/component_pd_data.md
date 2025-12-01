# Component: `pd_data.py`

## Purpose and Responsibility

`pd_data.py` bridges raw PD scenario JSON into structures suitable for:

- Prompt construction (`PromptPair` objects via `pd_prompt_builder`).  
- Deterministic train/test splits.  
- RepReader-style datasets used in representation learning.

It is deliberately small and deterministic: all randomness is explicit and controlled by seeds.

## Key Data Structures

```python
@dataclass
class PDPairBundle:
    pairs: List[PromptPair]
    train_pairs: List[PromptPair]
    test_pairs: List[PromptPair]
```

- `pairs`: all PD prompt pairs constructed from the JSON.  
- `train_pairs`: subset used for learning defection vectors.  
- `test_pairs`: subset used for validation and behavior evaluation.

## Core Functions

### `load_pairs`

```python
def load_pairs(json_path: Path, seed: int = 0) -> List[PromptPair]:
    rng = random.Random(seed)
    data = json.loads(json_path.read_text())
    pairs: List[PromptPair] = []
    for entry in data:
        pairs.append(build_pair(entry, rng))
    return pairs
```

Behavior:

- Reads a PD scenario JSON file (list of dicts).  
- Uses a local `random.Random` seeded with `seed` to:
  - Feed `build_pair`, which randomizes whether defection is option A or B.  
- Returns a list of `PromptPair` objects with reproducible option orderings.

### `split_pairs`

```python
def split_pairs(
    pairs: Sequence[PromptPair],
    train_ratio: float = 0.5,
    seed: int = 0,
) -> Tuple[List[PromptPair], List[PromptPair]]:
    rng = random.Random(seed)
    idxs = list(range(len(pairs)))
    rng.shuffle(idxs)
    cut = int(len(idxs) * train_ratio)
    train_idx = idxs[:cut]
    test_idx = idxs[cut:]
    train_pairs = [pairs[i] for i in train_idx]
    test_pairs = [pairs[i] for i in test_idx]
    return train_pairs, test_pairs
```

Behavior:

- Randomly shuffles indices using a local RNG with its own `seed`.  
- Splits indices into train and test at `train_ratio`.  
- Returns disjoint lists (`train_pairs`, `test_pairs`) that partition the input pairs.

Tests (`tests/test_pd_data.py`) enforce:

- Deterministic splits for the same seed.  
- No overlap between train and test sets.  
- Union of train and test equals the original set.

### `build_repreader_dataset`

```python
def build_repreader_dataset(pairs: Sequence[PromptPair]) -> dict:
    data: List[str] = []
    labels: List[List[int]] = []
    for pair in pairs:
        data.extend([pair.positive, pair.negative])
        labels.append([1, 0])
    return {"data": data, "labels": labels}
```

Behavior:

- For each `PromptPair`, appends its positive and negative prompts to `data`.  
- Adds a label vector `[1, 0]` indicating "defect (positive), cooperate (negative)".  
- Returns a dict format compatible with the RepReading pipeline expected by `neuro_manipulation`.

The tests verify:

- `data` length is exactly `2 * len(pairs)`.  
- `labels` length equals `len(pairs)` and all labels are `[1, 0]`.  
- The positive prompt appears before the negative one for each pair.

### `build_pd_pair_bundle`

```python
def build_pd_pair_bundle(json_path: Path, seed: int = 0) -> PDPairBundle:
    pairs = load_pairs(json_path, seed=seed)
    train_pairs, test_pairs = split_pairs(pairs, seed=seed)
    return PDPairBundle(pairs=pairs, train_pairs=train_pairs, test_pairs=test_pairs)
```

Behavior:

- Single convenience entry point used by higher-level code.  
- Uses the same `seed` for both `load_pairs` and `split_pairs`, ensuring:  
  - Randomized option ordering per scenario.  
  - A reproducible train/test split aligned with that ordering.

## Dependencies and Interactions

- **Inputs**:
  - PD scenario JSON at `data_creation/scenario_creation/langgraph_creation/Prisoners_Dilemma_all_data_samples.json`.

- **Outputs / Consumers**:
  - `run_pd_defection_experiment.run`:
    - Calls `build_pd_pair_bundle` to get train/test pairs.  
    - Calls `build_repreader_dataset` to construct RepReader datasets.  
    - Uses `train_pairs` and `test_pairs` for behavior evaluation on PD prompts.
  - `run_pd_defection_pd_behavior._restrict_dataset_to_pd_test_split`:
    - Calls `build_pd_pair_bundle` with the same `seed` and uses `test_pairs`' descriptions to filter the benchmark dataset.

This coupling via `seed` and descriptions is how PD training and benchmark transfer are aligned.

## Potential Issues and Improvements

- The JSON schema (`description`, `behavior_choices["defect"]`, `["cooperate"]`) is assumed but not validated here. If the upstream data format changes, failures may appear in `build_pair` or downstream code.

- Train/test splits are purely random. If stratification by difficulty or scenario type becomes important, `split_pairs` would need to be extended (keeping determinism via seed).

## Usage Example

```python
from pathlib import Path
from auto_experiments.task_similarity.pd_data import build_pd_pair_bundle, build_repreader_dataset

json_path = Path("data_creation/scenario_creation/langgraph_creation/Prisoners_Dilemma_all_data_samples.json")
bundle = build_pd_pair_bundle(json_path, seed=0)

train_ds = build_repreader_dataset(bundle.train_pairs)
test_ds = build_repreader_dataset(bundle.test_pairs)

print(len(bundle.pairs), "total pairs")
print(len(train_ds["data"]), "train prompts", len(test_ds["data"]), "test prompts")
```

This is essentially how `run_pd_defection_experiment.run` initializes its data.
