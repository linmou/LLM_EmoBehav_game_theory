"""
tests/unit/neuro_manipulation/repe/test_rep_reader_viz.py

Purpose: Validate the utility that turns stored RepReader layer directions into
2D coordinates for visualization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class _FakeRepReader:
    directions: dict[int, np.ndarray]


def test_collect_direction_points_and_reduce_to_2d():
    # Responsible module: neuro_manipulation/rep_reader_viz.py
    from neuro_manipulation.rep_reader_viz import (
        collect_direction_points,
        reduce_vectors_to_2d,
    )

    emotion_rep_readers = {
        "layer_acc": {"anger": {0: 1.0}},
        "args": {"model_name_or_path": "fake"},
        "anger": _FakeRepReader(
            directions={
                0: np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
                1: np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
            }
        ),
        "happiness": _FakeRepReader(
            directions={
                0: np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            }
        ),
    }

    vectors, meta = collect_direction_points(emotion_rep_readers, model_id="fake-model")
    assert vectors.shape == (3, 3)
    assert len(meta) == 3
    assert {m["emotion"] for m in meta} == {"anger", "happiness"}
    assert all("layer" in m and "component" in m and "model_id" in m for m in meta)

    coords = reduce_vectors_to_2d(vectors, method="pca")
    assert coords.shape == (3, 2)
    assert np.isfinite(coords).all()


def test_emotion_reader_cache_path_matches_hash(tmp_path):
    # Responsible module: neuro_manipulation/rep_reader_viz.py
    from neuro_manipulation.rep_reader_viz import emotion_reader_cache_path

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "anger.json").write_text('["a", "b", "c"]', encoding="utf-8")
    (data_dir / "happiness.json").write_text('["x", "y", "z"]', encoding="utf-8")

    cfg = {
        "emotions": ["anger", "happiness"],
        "data_dir": str(data_dir),
        "model_name_or_path": "/fake/model",
        "rep_token": -1,
        "n_difference": 1,
        "direction_method": "pca",
        "multimodal_intent": False,
        "emotion_data_seed": 0,
    }
    hidden_layers = [-1, -2, -3]
    p = emotion_reader_cache_path(cfg, hidden_layers=hidden_layers)
    assert str(p).startswith("neuro_manipulation/representation_storage/emotion_rep_reader_")
    assert str(p).endswith(".pkl")


def test_infer_repe_config_uses_defaults_when_missing():
    # Responsible module: neuro_manipulation/rep_reader_viz.py
    from neuro_manipulation.rep_reader_viz import infer_repe_config_for_model

    series_cfg = {"emotions": ["anger"], "intensities": [1.0]}  # unrelated keys
    repe_cfg = infer_repe_config_for_model("/fake/model", series_cfg)
    assert repe_cfg["model_name_or_path"] == "/fake/model"
    assert repe_cfg["direction_method"] == "pca"
    assert repe_cfg["rep_token"] == -1
    assert "data_dir" in repe_cfg
