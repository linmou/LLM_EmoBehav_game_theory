# Tests for result_analysis/game_theory_impact_heatmaps.py
"""Verify heatmap supports symmetric-log normalization."""

from __future__ import annotations

from matplotlib.colors import SymLogNorm

from result_analysis.game_theory_impact_heatmaps import _build_heatmap_norm


def test_build_heatmap_norm_symlog() -> None:
    norm, vmin, vmax = _build_heatmap_norm([[0.001, -0.002], [0.5, -0.4]], mode="symlog", linthresh=0.01)
    assert isinstance(norm, SymLogNorm)
    assert vmin < 0
    assert vmax > 0

