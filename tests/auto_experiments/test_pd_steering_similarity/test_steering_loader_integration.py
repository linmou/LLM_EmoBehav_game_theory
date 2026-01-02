#!/usr/bin/env python3
# Tests for steering_loader: ensure it returns normalized float32 arrays.

from pathlib import Path

import numpy as np

from auto_experiments.layer_vector_sim.pd_steering_similarity import steering_loader


def test_load_layer_vectors_normalizes_vectors(tmp_path: Path) -> None:
    vec_dir = tmp_path / "layer_vectors"
    vec_dir.mkdir(parents=True)
    np.save(vec_dir / "layer_0.npy", np.array([3.0, 0.0], dtype=np.float32))

    loaded = steering_loader.load_layer_vectors(vec_dir)

    assert np.allclose(np.linalg.norm(loaded[0]), 1.0)
