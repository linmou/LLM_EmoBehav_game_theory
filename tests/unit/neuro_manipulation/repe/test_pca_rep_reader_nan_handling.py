"""
tests/unit/neuro_manipulation/repe/test_pca_rep_reader_nan_handling.py

Purpose: Ensure PCARepReader robustly handles non-finite hidden states (NaN/Inf)
instead of crashing inside sklearn PCA.
"""

import numpy as np
import pytest

from neuro_manipulation.repe.rep_readers import PCARepReader


def test_pca_rep_reader_drops_non_finite_rows():
    hidden_layers = [-1]
    hidden_states = {
        -1: np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [np.nan, 1.0, 2.0, 3.0],  # bad row
                [0.0, 1.0, np.inf, 3.0],  # bad row
                [0.0, 1.0, 2.0, 3.0],
                [0.5, 1.5, 2.5, 3.5],
                [-0.5, 0.5, 1.5, 2.5],
            ],
            dtype=np.float32,
        )
    }

    rep_reader = PCARepReader(n_components=1)
    directions = rep_reader.get_rep_directions(
        model=None, tokenizer=None, hidden_states=hidden_states, hidden_layers=hidden_layers
    )

    assert -1 in directions
    assert directions[-1].shape == (1, 4)
    assert np.isfinite(directions[-1]).all()


def test_pca_rep_reader_all_non_finite_raises():
    hidden_layers = [-1]
    hidden_states = {
        -1: np.array(
            [
                [np.nan, np.nan],
                [np.inf, -np.inf],
            ],
            dtype=np.float32,
        )
    }

    rep_reader = PCARepReader(n_components=1)
    with pytest.raises(ValueError, match="non-finite"):
        rep_reader.get_rep_directions(
            model=None, tokenizer=None, hidden_states=hidden_states, hidden_layers=hidden_layers
        )

