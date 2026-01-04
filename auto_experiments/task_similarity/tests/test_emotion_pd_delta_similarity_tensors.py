"""
Responsible: auto_experiments/task_similarity/emotion_pd_delta_similarity.py
Purpose: Validate optional tensor outputs (hidden states + deltas) are written with correct shapes.
"""

from pathlib import Path

import numpy as np


def test_write_tensor_outputs(tmp_path: Path):
    from auto_experiments.task_similarity.emotion_pd_delta_similarity import write_tensor_outputs

    n_int = 2
    n_samples = 3
    n_layers = 4
    hidden = 5

    base_hidden = np.zeros((n_samples, n_layers, hidden), dtype=np.float32)
    emotion_hidden = np.ones((n_int, n_samples, n_layers, hidden), dtype=np.float32) * 2.0
    pd_hidden = np.ones((n_int, n_samples, n_layers, hidden), dtype=np.float32) * 3.0
    pd_cooperate_hidden = np.ones((n_int, n_samples, n_layers, hidden), dtype=np.float32) * 4.0
    delta_emotion = emotion_hidden - base_hidden[None, :, :, :]
    delta_pd = pd_hidden - base_hidden[None, :, :, :]
    delta_pd_cooperate = pd_cooperate_hidden - base_hidden[None, :, :, :]

    write_tensor_outputs(
        out_dir=tmp_path,
        base_hidden=base_hidden,
        emotion_hidden=emotion_hidden,
        pd_hidden=pd_hidden,
        pd_cooperate_hidden=pd_cooperate_hidden,
        delta_emotion=delta_emotion,
        delta_pd=delta_pd,
        delta_pd_cooperate=delta_pd_cooperate,
        dtype="float16",
    )

    paths = {
        "base": tmp_path / "hidden_base.npy",
        "emotion": tmp_path / "hidden_emotion.npy",
        "pd": tmp_path / "hidden_pd.npy",
        "pd_cooperate": tmp_path / "hidden_pd_cooperate.npy",
        "delta_emotion": tmp_path / "delta_emotion.npy",
        "delta_pd": tmp_path / "delta_pd.npy",
        "delta_pd_cooperate": tmp_path / "delta_pd_cooperate.npy",
    }
    for p in paths.values():
        assert p.exists()

    assert np.load(paths["base"]).shape == (n_samples, n_layers, hidden)
    assert np.load(paths["emotion"]).shape == (n_int, n_samples, n_layers, hidden)
    assert np.load(paths["pd"]).shape == (n_int, n_samples, n_layers, hidden)
    assert np.load(paths["pd_cooperate"]).shape == (n_int, n_samples, n_layers, hidden)
    assert np.load(paths["delta_emotion"]).shape == (n_int, n_samples, n_layers, hidden)
    assert np.load(paths["delta_pd"]).shape == (n_int, n_samples, n_layers, hidden)
    assert np.load(paths["delta_pd_cooperate"]).shape == (n_int, n_samples, n_layers, hidden)

    assert np.load(paths["base"]).dtype == np.float16
