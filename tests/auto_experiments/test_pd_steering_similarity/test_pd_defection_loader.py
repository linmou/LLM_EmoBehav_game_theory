#!/usr/bin/env python3
# Tests for auto_experiments.layer_vector_sim.pd_steering_similarity.pd_defection_loader.

from pathlib import Path

import numpy as np
import pytest

from auto_experiments.layer_vector_sim.pd_steering_similarity import pd_defection_loader


def test_load_pd_defection_vectors_reads_layers(tmp_path: Path) -> None:
    vec_dir = tmp_path / "pd_vectors"
    vec_dir.mkdir(parents=True)
    np.save(vec_dir / "layer_0.npy", np.array([1.0, 0.0], dtype=np.float32))
    np.save(vec_dir / "layer_2.npy", np.array([0.0, 2.0], dtype=np.float32))

    loaded = pd_defection_loader.load_pd_defection_vectors(vec_dir)

    assert set(loaded.keys()) == {0, 2}
    assert np.allclose(np.linalg.norm(loaded[0]), 1.0)
    assert np.allclose(np.linalg.norm(loaded[2]), 1.0)


def test_load_pd_defection_vectors_raises_on_missing(tmp_path: Path) -> None:
    vec_dir = tmp_path / "missing_vectors"
    vec_dir.mkdir(parents=True)

    with pytest.raises(ValueError):
        pd_defection_loader.load_pd_defection_vectors(vec_dir)
