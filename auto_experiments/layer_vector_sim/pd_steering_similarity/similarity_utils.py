"""
Similarity utilities for PD steering similarity.
"""

import numpy as np


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors. Returns 0.0 if either is zero.
    """
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def similarity_delta(baseline: float, steered: float) -> float:
    """
    Compute steered - baseline similarity delta.
    """
    return float(steered - baseline)
