"""
Responsible: auto_experiments/task_similarity/emotion_pd_delta_similarity.py
Purpose: Unit tests for layer mapping and cosine logic used by anger-vs-PD delta similarity analysis.
"""

import numpy as np


def test_middle_third_layers_qwen_36():
    from auto_experiments.task_similarity.emotion_pd_delta_similarity import (
        middle_third_layers,
    )

    assert middle_third_layers(36) == list(range(12, 24))


def test_repreader_layer_key_mapping():
    from auto_experiments.task_similarity.emotion_pd_delta_similarity import (
        repreader_key_for_layer,
    )

    # Qwen2.5-3B has 36 layers: layer 0 -> -36, layer 35 -> -1
    assert repreader_key_for_layer(layer=0, num_layers=36) == -36
    assert repreader_key_for_layer(layer=12, num_layers=36) == -24
    assert repreader_key_for_layer(layer=35, num_layers=36) == -1


def test_cosine_similarity_nan_when_zero_norm():
    from auto_experiments.task_similarity.emotion_pd_delta_similarity import (
        cosine_per_layer,
    )

    # Two samples, two layers, dim=3
    a = np.zeros((2, 2, 3), dtype=np.float32)
    b = np.ones((2, 2, 3), dtype=np.float32)
    cos = cosine_per_layer(a, b, eps=1e-12)
    assert cos.shape == (2, 2)
    assert np.isnan(cos).all()


def test_cosine_similarity_matches_expected():
    from auto_experiments.task_similarity.emotion_pd_delta_similarity import (
        cosine_per_layer,
    )

    a = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=np.float32,
    )  # (1, 2 layers, 2 dim)
    b = np.array(
        [
            [[1.0, 0.0], [1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    cos = cosine_per_layer(a, b, eps=1e-12)
    assert cos.shape == (1, 2)
    assert np.allclose(cos[0, 0], 1.0)
    assert np.allclose(cos[0, 1], 0.0)


def test_load_split_indices_all_train_test(tmp_path):
    from auto_experiments.task_similarity.emotion_pd_delta_similarity import (
        _load_split_indices,
    )

    manifest = {
        "split_seed": 20,
        "train_indices": [0, 2, 4],
        "test_indices": [1, 3],
    }
    path = tmp_path / "split_manifest.json"
    path.write_text(__import__("json").dumps(manifest), encoding="utf-8")

    assert _load_split_indices(path, split="test") == [1, 3]
    assert _load_split_indices(path, split="train") == [0, 2, 4]
    assert _load_split_indices(path, split="all") == [0, 1, 2, 3, 4]
