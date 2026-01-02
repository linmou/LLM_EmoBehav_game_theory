"""
Responsible: auto_experiments/task_similarity/emotion_pd_delta_similarity.py
Purpose: Ensure emotion RepReader loading is keyed by the requested emotion.
"""

import pickle
from pathlib import Path

import numpy as np
import pytest


class _DummyRepReader:
    def __init__(self, directions):
        self.directions = directions


def test_load_emotion_layer_vectors_uses_requested_key(tmp_path: Path):
    from auto_experiments.task_similarity.emotion_pd_delta_similarity import _load_emotion_layer_vectors

    num_layers = 4
    control_layers = [1, 2]
    # mapping: key = layer - num_layers => -3, -2
    sadness = _DummyRepReader(
        directions={
            -3: np.ones((1, 3), dtype=np.float32),
            -2: np.ones((1, 3), dtype=np.float32) * 2,
        }
    )
    anger = _DummyRepReader(
        directions={
            -3: np.ones((1, 3), dtype=np.float32) * 9,
            -2: np.ones((1, 3), dtype=np.float32) * 8,
        }
    )
    pkl = tmp_path / "rr.pkl"
    pkl.write_bytes(pickle.dumps({"sadness": sadness, "anger": anger}))

    out = _load_emotion_layer_vectors(pkl, num_layers=num_layers, control_layers=control_layers, emotion="sadness")
    assert np.allclose(out[1], np.array([1, 1, 1], dtype=np.float32))
    assert np.allclose(out[2], np.array([2, 2, 2], dtype=np.float32))


def test_load_emotion_layer_vectors_missing_emotion_key(tmp_path: Path):
    from auto_experiments.task_similarity.emotion_pd_delta_similarity import _load_emotion_layer_vectors

    pkl = tmp_path / "rr.pkl"
    pkl.write_bytes(pickle.dumps({"anger": _DummyRepReader(directions={})}))

    with pytest.raises(ValueError, match="must be dict with key"):
        _load_emotion_layer_vectors(pkl, num_layers=4, control_layers=[0], emotion="sadness")
