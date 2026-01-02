#!/usr/bin/env python3
# Tests for auto_experiments.layer_vector_sim.pd_steering_similarity.steering_loader: load per-layer steering vectors.

from pathlib import Path

import numpy as np
import pytest

from auto_experiments.layer_vector_sim.pd_steering_similarity import steering_loader


def test_load_layer_vectors_reads_all_layers(tmp_path: Path) -> None:
    vec_dir = tmp_path / "layer_vectors"
    vec_dir.mkdir(parents=True)
    np.save(vec_dir / "layer_0.npy", np.ones(3, dtype=np.float32))
    np.save(vec_dir / "layer_1.npy", np.ones(3, dtype=np.float32) * 2)

    loaded = steering_loader.load_layer_vectors(vec_dir)

    assert set(loaded.keys()) == {0, 1}
    assert np.allclose(np.linalg.norm(loaded[0]), 1.0)
    assert np.allclose(np.linalg.norm(loaded[1]), 1.0)


def test_load_layer_vectors_raises_on_empty_dir(tmp_path: Path) -> None:
    vec_dir = tmp_path / "empty_vectors"
    vec_dir.mkdir(parents=True)

    with pytest.raises(ValueError):
        steering_loader.load_layer_vectors(vec_dir)
