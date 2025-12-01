"""Tests: auto_experiments/task-similarity/compute_pd_delta.py helpers."""

import numpy as np

from auto_experiments.task_similarity.compute_pd_delta import compute_delta


def test_compute_delta_basic():
    base = np.array([1.0, 2.0], dtype=np.float32)
    steered = np.array([1.5, 1.0], dtype=np.float32)
    delta = compute_delta(base, steered)
    np.testing.assert_allclose(delta, np.array([0.5, -1.0], dtype=np.float32))


def test_compute_delta_shape_mismatch():
    base = np.zeros((2,), dtype=np.float32)
    steered = np.zeros((3,), dtype=np.float32)
    try:
        compute_delta(base, steered)
    except ValueError as exc:
        assert "shape mismatch" in str(exc)
    else:  # pragma: no cover
        assert False, "Expected ValueError on shape mismatch"
