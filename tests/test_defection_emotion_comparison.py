# tests/test_defection_emotion_comparison.py: verifies comparison of emotion and defection delta vectors.
import numpy as np
import pytest

from result_analysis.defection_emotion_comparison import (
    collect_defection_vectors,
    collect_emotion_vectors,
    compute_best_matrix,
    format_markdown_tables,
)


def test_builds_markdown_table_with_best_cosines(tmp_path) -> None:
    def_root = tmp_path / "defection"
    def_run = def_root / "best_layer" / "0.5B" / "ModelA_delta_20250101"
    def_run.mkdir(parents=True)
    # Layer 10 points along x-axis, layer 11 along y-axis.
    np.savez(def_run / "delta.npz", **{"10": np.array([1, 0, 0, 0], dtype=np.float32), "11": np.array([0, 1, 0, 0], dtype=np.float32)})

    emo_root = tmp_path / "emotion"
    emo_delta_dir = emo_root / "ModelA_20250101_000000" / "deltas"
    emo_delta_dir.mkdir(parents=True)
    np.savez(emo_delta_dir / "emotion=anger_int=1.0.npz", vector=np.array([0, 1, 0, 0], dtype=np.float32))
    np.savez(emo_delta_dir / "emotion=sadness_int=1.5.npz", vector=np.array([-1, -1, 0, 0], dtype=np.float32))

    def_vectors = collect_defection_vectors(def_root)
    emo_vectors = collect_emotion_vectors(emo_root)
    matrix = compute_best_matrix(def_vectors, emo_vectors)

    assert "ModelA" in matrix
    assert 1.0 in matrix["ModelA"]
    anger_cell = matrix["ModelA"][1.0]["anger"]
    assert anger_cell["cos"] == pytest.approx(1.0)
    sadness_cell = matrix["ModelA"][1.5]["sadness"]
    assert sadness_cell["def_layer"] == "11"
    assert sadness_cell["cos"] == pytest.approx(-0.70710678, rel=1e-6)

    markdown = format_markdown_tables(matrix)
    assert "| intensity\\emotion | anger | sadness |" in markdown
    assert "| 1 | 1.0000 (L11) | - |" in markdown
    assert "| 1.5 | - | -0.7071 (L11) |" in markdown
