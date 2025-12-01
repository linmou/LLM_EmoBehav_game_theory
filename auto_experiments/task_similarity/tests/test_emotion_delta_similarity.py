"""
Tests for auto_experiments/task_similarity/emotion_delta_similarity.py.

Focus: file loading, seed deduplication, and PCA-based global direction similarity.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

import numpy as np

from ..emotion_delta_similarity import (
    EmotionPCASummary,
    compute_pca_first_component,
    compute_pca_similarity,
    load_chat_seed_emotion_vectors,
    load_pd_seed_vectors,
)


def _write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _save_npz_vector(path: Path, vec) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, vector=np.asarray(vec, dtype=np.float32))


def test_compute_pca_first_component_orientation_and_variance() -> None:
    # Simple 2D case where the main variance is clearly along x-axis.
    # Points: (1, 0), (2, 0), (3, 0).
    mat = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    pc1, var_ratio = compute_pca_first_component(mat)

    # PC1 should be aligned with +x after orientation fix.
    assert pc1.shape == (2,)
    # Allow small numerical tolerance.
    assert pc1[0] > 0.0
    # All mass is on first component.
    assert var_ratio > 0.99


def test_compute_pca_first_component_degenerate_uses_mean_direction() -> None:
    # All rows identical: PCA should fall back to normalized mean direction with zero variance ratio.
    mat = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    pc1, var_ratio = compute_pca_first_component(mat)

    assert pc1.shape == (2,)
    assert np.allclose(pc1, np.array([1.0, 0.0], dtype=np.float32))
    assert var_ratio == 0.0


def test_compute_pca_similarity_ranks_aligned_emotion_higher() -> None:
    # PD deltas vary along x-axis.
    pd_vectors = {
        0: np.array([1.0, 0.0], dtype=np.float32),
        1: np.array([2.0, 0.0], dtype=np.float32),
        2: np.array([3.0, 0.0], dtype=np.float32),
    }

    # Emotion "anger" aligned with PD (x-axis), "happiness" orthogonal (y-axis).
    emo_vectors = {
        0: {"anger": np.array([1.0, 0.0], dtype=np.float32), "happiness": np.array([0.0, 1.0], dtype=np.float32)},
        1: {"anger": np.array([2.0, 0.0], dtype=np.float32), "happiness": np.array([0.0, 2.0], dtype=np.float32)},
        2: {"anger": np.array([3.0, 0.0], dtype=np.float32), "happiness": np.array([0.0, 3.0], dtype=np.float32)},
    }

    result = compute_pca_similarity(pd_vectors, emo_vectors, seeds=[0, 1, 2])

    assert set(result.keys()) == {"anger", "happiness"}

    anger: EmotionPCASummary = result["anger"]
    happiness: EmotionPCASummary = result["happiness"]

    # Anger PC1 should be much more aligned with PD PC1 than happiness.
    assert anger.pc1_cosine > 0.9
    assert happiness.pc1_cosine < 0.1


def test_load_pd_seed_vectors_single_key_delta(tmp_path) -> None:
    pd_root = Path(tmp_path) / "pd"
    model_prefix = "Qwen2.5-3B-Instruct_"

    # Create two PD runs for seeds 0 and 1.
    for seed in (0, 1):
        run_dir = pd_root / f"{model_prefix}20250101_0000{seed}"
        run_dir.mkdir(parents=True)
        meta = {
            "model_path": "/fake/model/path",
            "vector_path": "unused",
            "control_layers": [12, 13],
            "measurement_layer": 35,
            "intensity": 1.5,
            "seed": seed,
            "timestamp": f"20250101_0000{seed}",
            "prompt_hash": 123,
        }
        _write_json(run_dir / "metadata.json", meta)

        # delta.npz with a single key "35".
        np.savez(run_dir / "delta.npz", **{"35": np.array([seed + 1.0, 0.0], dtype=np.float32)})

    seed_to_vec = load_pd_seed_vectors(pd_root, model_prefix, seed_min=0, seed_max=1)

    assert set(seed_to_vec.keys()) == {0, 1}
    assert np.allclose(seed_to_vec[0], np.array([1.0, 0.0], dtype=np.float32))
    assert np.allclose(seed_to_vec[1], np.array([2.0, 0.0], dtype=np.float32))


def test_load_chat_seed_emotion_vectors_picks_latest_run(tmp_path) -> None:
    chat_root = Path(tmp_path) / "chat"
    chat_root.mkdir(parents=True)
    model_prefix = "Qwen2.5-3B-Instruct_"
    intensity = 1.5

    # Seed 20 has two runs, we expect latest (lexicographically largest) dirname to win.
    seeds = [20, 21]
    emotions = ["anger", "happiness"]
    intensities = [0.0, 1.5]

    def _make_metadata(seed: int) -> dict:
        return {
            "pipeline": "chat",
            "model_path": "/fake/model/path",
            "emotions": emotions,
            "intensities": intensities,
            "probe_hash": "abc",
            "timestamp": "20250101_000000",
            "chat_template": "unused",
            "prompt_config": {},
            "job_config": {
                "model_path": "/fake/model/path",
                "emotions": emotions,
                "intensities": intensities,
                "output_dir": "results/delta_activations",
                "loading_config": {"model_path": "/fake/model/path", "max_model_len": 4096, "seed": seed},
                "repe_eng_config": {},
            },
            "backend_metadata": {"backend": "hf", "control_layers": [12, 13], "max_length": 256},
        }

    # Older run for seed 20.
    run1 = chat_root / f"{model_prefix}20250125_000000"
    run1.mkdir()
    _write_json(run1 / "metadata.json", _make_metadata(20))
    for emo in emotions:
        _save_npz_vector(run1 / "deltas" / f"emotion={emo}_int={intensity}.npz", [1.0, 0.0])

    # Newer run for seed 20 with different vectors.
    run2 = chat_root / f"{model_prefix}20250127_010000"
    run2.mkdir()
    _write_json(run2 / "metadata.json", _make_metadata(20))
    for emo in emotions:
        _save_npz_vector(run2 / "deltas" / f"emotion={emo}_int={intensity}.npz", [2.0, 0.0])

    # Single run for seed 21.
    run3 = chat_root / f"{model_prefix}20250127_020000"
    run3.mkdir()
    _write_json(run3 / "metadata.json", _make_metadata(21))
    for emo in emotions:
        _save_npz_vector(run3 / "deltas" / f"emotion={emo}_int={intensity}.npz", [3.0, 0.0])

    seed_to_emo, emos = load_chat_seed_emotion_vectors(
        chat_root, model_prefix, intensity=intensity, seed_min=20, seed_max=21
    )

    # Emotions come back in the same set.
    assert set(emos) == set(emotions)

    # We should have both seeds 20 and 21.
    assert set(seed_to_emo.keys()) == {20, 21}

    # Seed 20 should use the newer vectors (value 2.0).
    for emo in emotions:
        assert np.allclose(seed_to_emo[20][emo], np.array([2.0, 0.0], dtype=np.float32))

    # Seed 21 should use the only available vectors (value 3.0).
    for emo in emotions:
        assert np.allclose(seed_to_emo[21][emo], np.array([3.0, 0.0], dtype=np.float32))
