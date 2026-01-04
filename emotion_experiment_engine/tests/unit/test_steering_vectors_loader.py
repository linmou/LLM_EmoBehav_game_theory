#!/usr/bin/env python3
"""
Responsible file: emotion_experiment_engine/steering_vectors.py
Purpose: Validate loading `layer_*.npy` steering vectors from disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_load_layer_vectors_dir_reads_layer_files(tmp_path: Path) -> None:
    from emotion_experiment_engine.steering_vectors import load_layer_vectors_dir

    vec_dir = tmp_path / "layer_vectors"
    vec_dir.mkdir()
    np.save(vec_dir / "layer_2.npy", np.ones(4, dtype=np.float32))
    np.save(vec_dir / "layer_10.npy", np.zeros(4, dtype=np.float32))

    out = load_layer_vectors_dir(vec_dir)
    assert sorted(out.keys()) == [2, 10]
    assert out[2].shape == (4,)


def test_load_layer_vectors_dir_requires_files(tmp_path: Path) -> None:
    from emotion_experiment_engine.steering_vectors import load_layer_vectors_dir

    vec_dir = tmp_path / "layer_vectors"
    vec_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_layer_vectors_dir(vec_dir)

