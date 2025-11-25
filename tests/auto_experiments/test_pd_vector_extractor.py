"""Tests: auto_experiments/task-similarity/pd_vector_extractor.py
Purpose: verify vector computation and best-layer selection."""

import numpy as np

from auto_experiments.task_similarity.pd_vector_extractor import (
    compute_vectors_and_accuracy,
    select_best_layer,
)


def test_vector_accuracy_and_best_layer():
    # Layer 0: perfect separation
    feats0 = np.array(
        [
            [2.0, 0.0],  # pos
            [0.0, 0.0],  # neg
            [1.0, 0.0],  # pos
            [0.0, 0.0],  # neg
        ]
    )
    test0 = np.array([[1.0, 0.0], [0.0, 0.0]])

    # Layer 1: weaker separation
    feats1 = np.array(
        [
            [0.5, 0.5],
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.0],
        ]
    )
    test1 = np.array([[0.1, 0.0], [0.0, 0.0]])

    results = compute_vectors_and_accuracy(
        {0: feats0, 1: feats1}, {0: test0, 1: test1}
    )
    best_layer, best = select_best_layer(results)

    assert best_layer == 0
    assert best.accuracy == 1.0
    assert results[1].accuracy > 0.0
