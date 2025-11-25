"""
Responsible: auto_experiments/task-similarity/pd_vector_extractor.py
Purpose: Compute per-layer defection vectors and accuracies from hidden states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class LayerVectorResult:
    vector: np.ndarray
    accuracy: float


def compute_vectors_and_accuracy(
    layer_hidden: Dict[int, np.ndarray],
    test_layer_hidden: Dict[int, np.ndarray],
) -> Dict[int, LayerVectorResult]:
    """
    Args:
        layer_hidden: mapping layer -> (2N, hidden) train features (pos, neg alternating)
        test_layer_hidden: mapping layer -> (2M, hidden) test features (pos, neg alternating)
    Returns:
        Per-layer vector and validation accuracy (projection > 0)
    """
    results: Dict[int, LayerVectorResult] = {}
    for layer, feats in layer_hidden.items():
        pos = feats[::2]
        neg = feats[1::2]
        diff = pos - neg
        vec = diff.mean(axis=0)
        test_feats = test_layer_hidden[layer]
        t_pos = test_feats[::2]
        t_neg = test_feats[1::2]
        t_diff = t_pos - t_neg
        scores = np.dot(t_diff, vec)
        acc = float((scores > 0).mean()) if len(scores) else 0.0
        results[layer] = LayerVectorResult(vector=vec, accuracy=acc)
    return results


def select_best_layer(results: Dict[int, LayerVectorResult]) -> Tuple[int, LayerVectorResult]:
    if not results:
        raise ValueError("No layer results provided")
    best_layer = max(results.items(), key=lambda kv: kv[1].accuracy)[0]
    return best_layer, results[best_layer]
