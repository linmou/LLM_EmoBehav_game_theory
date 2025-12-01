# Component: `pd_vector_extractor.py`

## Purpose and Responsibility

`pd_vector_extractor.py` provides a simple, self-contained implementation of:

- Per-layer **defection vectors** obtained via a diff-of-means approach.  
- Associated per-layer validation accuracies.

While `run_pd_defection_experiment.train_pd_repreader` currently uses a PCA-based approach, this module is useful as:

- A baseline implementation.  
- A reference for unit tests or exploratory analysis.  
- A potential drop-in if you want a simpler training pipeline.

## Key Data Structure

```python
@dataclass
class LayerVectorResult:
    vector: np.ndarray
    accuracy: float
```

Represents, for one layer:

- `vector`: the learned defection direction.  
- `accuracy`: held-out validation accuracy based on projection sign.

## Core Functions

### `compute_vectors_and_accuracy`

```python
def compute_vectors_and_accuracy(
    layer_hidden: Dict[int, np.ndarray],
    test_layer_hidden: Dict[int, np.ndarray],
) -> Dict[int, LayerVectorResult]:
    ...
```

Inputs:

- `layer_hidden`: mapping `layer -> (2N, hidden_dim)` for training data.  
  - Layout: `[pos0, neg0, pos1, neg1, ...]`.  
- `test_layer_hidden`: same mapping for test data, layout `[pos0, neg0, ...]`.

Algorithm for each `layer`:

1. Split training features into positive and negative:
   ```python
   pos = feats[::2]
   neg = feats[1::2]
   diff = pos - neg
   vec = diff.mean(axis=0)
   ```
2. For the test set:
   ```python
   t_pos = test_feats[::2]
   t_neg = test_feats[1::2]
   t_diff = t_pos - t_neg
   scores = np.dot(t_diff, vec)
   accuracy = float((scores > 0).mean())
   ```
   - If the direction perfectly separates defection from cooperation, `scores` will be positive for all test pairs.

3. Store `LayerVectorResult(vector=vec, accuracy=accuracy)` for the layer.

The function returns a dict `layer -> LayerVectorResult`.

### `select_best_layer`

```python
def select_best_layer(results: Dict[int, LayerVectorResult]) -> Tuple[int, LayerVectorResult]:
    if not results:
        raise ValueError("No layer results provided")
    best_layer = max(results.items(), key=lambda kv: kv[1].accuracy)[0]
    return best_layer, results[best_layer]
```

Simple helper to:

- Guard against empty input.  
- Select the layer with maximum `accuracy`.

## Dependencies and Interactions

- This module is currently not called from the main PD training pipeline, which uses the PCA-based logic in `run_pd_defection_experiment.train_pd_repreader`.  
- It is, however, aligned in expectations:
  - Assumes alternating positive/negative ordering.  
  - Expects consistent shapes across train and test hidden states.

If you need a simpler, less compute-heavy alternative to PCA, you can reuse this module by plugging its output where `train_pd_repreader` currently returns `layer_vectors`.

## Potential Issues and Improvements

- Accuracy uses a strict `scores > 0` threshold. In noisy settings, you might want to normalize the vector or introduce a margin, but that would complicate the simple baseline.

- The module does not perform any centering of `diff` vectors beyond the implicit diff; if you want to mimic PCA's centering, you could subtract the mean before computing the final vector, at the cost of additional code.

## Usage Example

```python
from auto_experiments.task_similarity.pd_vector_extractor import (
    compute_vectors_and_accuracy,
    select_best_layer,
)

train_hidden = {8: train_layer8, 9: train_layer9}  # each (2N, hidden_dim)
test_hidden = {8: test_layer8, 9: test_layer9}

results = compute_vectors_and_accuracy(train_hidden, test_hidden)
best_layer, best_result = select_best_layer(results)

print("Best layer:", best_layer, "accuracy:", best_result.accuracy)
vec = best_result.vector
```

This is analogous to the logic in `run_pd_defection_experiment.train_pd_repreader`, but uses a diff-of-means rather than PCA.

