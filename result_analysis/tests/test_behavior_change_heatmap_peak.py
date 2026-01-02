# Tests for result_analysis/game_theory_impact_heatmaps.py
"""Verify behavior-change heatmap uses peak-|delta| intensity per (model, emotion)."""

from __future__ import annotations

from pathlib import Path

from result_analysis.game_theory_impact_heatmaps import compute_peak_behavior_change_matrix


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(x) for x in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_compute_peak_behavior_change_matrix_picks_max_abs_delta(tmp_path: Path) -> None:
    behavior_csv = tmp_path / "summary_behavior_ratio.csv"
    _write_csv(
        behavior_csv,
        ["emotion", "intensity", "behavior", "ratio"],
        [
            ["neutral", 0.6, "defect", 0.2],
            ["neutral", 1.5, "defect", 0.2],
            ["anger", 0.6, "defect", 0.25],  # delta +0.05
            ["anger", 1.5, "defect", 0.90],  # delta +0.70 (peak)
        ],
    )

    models, emotions, values, chosen = compute_peak_behavior_change_matrix(
        task="TestGame",
        model_to_behavior_csv={"FooModel": behavior_csv},
        unknown_threshold=None,
    )
    assert models == ["FooModel"]
    assert emotions == ["anger"]
    assert values == [[0.70]]
    assert chosen[("FooModel", "anger")] == 1.5

