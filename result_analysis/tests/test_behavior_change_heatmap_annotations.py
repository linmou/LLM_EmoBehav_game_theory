# Tests for result_analysis/game_theory_impact_heatmaps.py
"""Verify behavior-change heatmap annotations include delta values and significance marks."""

from __future__ import annotations

from pathlib import Path

from result_analysis.game_theory_impact_heatmaps import compute_peak_behavior_change_annotation_matrix


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(x) for x in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_peak_behavior_heatmap_annotations_include_delta_and_sig(tmp_path: Path) -> None:
    run_dir = tmp_path / "FooModel_game_theory_TestGame_20250102_010101"
    behavior_csv = run_dir / "summary_behavior_ratio.csv"

    _write_csv(
        behavior_csv,
        ["emotion", "intensity", "behavior", "ratio"],
        [
            ["neutral", 0.0, "cooperate", 0.8],
            ["neutral", 0.0, "defect", 0.2],
            ["neutral", 1.5, "cooperate", 0.8],
            ["neutral", 1.5, "defect", 0.2],
            ["anger", 0.6, "cooperate", 0.75],
            ["anger", 0.6, "defect", 0.25],  # delta +0.05
            ["anger", 1.5, "cooperate", 0.10],
            ["anger", 1.5, "defect", 0.90],  # delta +0.70 (peak)
        ],
    )

    # Provide item-level paired choices for significance on target behavior (defect) at intensity=1.5.
    detailed = run_dir / "detailed_results.csv"
    lines = [
        "emotion,intensity,item_id,repeat_id,chosen_behavior",
    ]
    for item_id in range(10):
        lines.append(f"neutral,0.0,{item_id},0,cooperate")
        lines.append(f"anger,1.5,{item_id},0,defect")
    detailed.write_text("\n".join(lines) + "\n", encoding="utf-8")

    models, emotions, values, annotations = compute_peak_behavior_change_annotation_matrix(
        task="TestGame",
        model_to_behavior_csv={"FooModel": behavior_csv},
        unknown_threshold=None,
    )
    assert models == ["FooModel"]
    assert emotions == ["anger"]
    assert values == [[0.70]]
    assert annotations == [["+0.70**"]]
