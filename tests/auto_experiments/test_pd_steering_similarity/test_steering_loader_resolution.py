#!/usr/bin/env python3
# Tests for steering_loader directory resolution helpers.

from pathlib import Path

import numpy as np

from auto_experiments.layer_vector_sim.pd_steering_similarity import steering_loader


def test_load_emotion_vectors_uses_nested_layer_vectors(tmp_path: Path) -> None:
    root = tmp_path / "run_root"
    seed_dir = root / "seed_20" / "layer_vectors"
    seed_dir.mkdir(parents=True)
    np.save(seed_dir / "layer_0.npy", np.array([1.0, 0.0], dtype=np.float32))

    loaded = steering_loader.load_emotion_vectors(root)

    assert set(loaded.keys()) == {0}
    assert np.allclose(np.linalg.norm(loaded[0]), 1.0)
