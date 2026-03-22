# Tests for result_analysis/generate_game_theory_impact_report.py
"""Verify option- and behavior-level impact report generation for game theory results."""

from __future__ import annotations

from pathlib import Path

import pytest

from result_analysis.generate_game_theory_impact_report import generate_game_theory_impact_report


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(x) for x in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_generate_shuffle_choice_impact_report_selects_latest_and_renames_outputs(tmp_path: Path) -> None:
    root = tmp_path / "results" / "new_game_theory_decision" / "shuffle_choices"

    older_dir = root / "FooModel_game_theory_decision_Prisoners_Dilemma_20250101_010101"
    newer_dir = root / "FooModel_game_theory_decision_Prisoners_Dilemma_20250102_010101"

    # Older run (ignored): extreme deltas
    _write_csv(
        older_dir / "summary_behavior_ratio.csv",
        ["emotion", "intensity", "behavior_label", "ratio"],
        [
            ["neutral", 0.1, "cooperate", 0.0],
            ["neutral", 0.1, "defect", 1.0],
            ["anger", 0.1, "cooperate", 1.0],
            ["anger", 0.1, "defect", 0.0],
        ],
    )
    _write_csv(
        older_dir / "summary_choice_ratio.csv",
        ["emotion", "intensity", "option_id", "ratio"],
        [
            ["neutral", 0.1, 1, 0.0],
            ["neutral", 0.1, 2, 1.0],
            ["anger", 0.1, 1, 1.0],
            ["anger", 0.1, 2, 0.0],
        ],
    )

    # Newer run: intensity-collapsed neutral(cooperate)=0.4, anger(cooperate)=0.2 => delta=-0.2
    _write_csv(
        newer_dir / "summary_behavior_ratio.csv",
        ["emotion", "intensity", "behavior_label", "ratio"],
        [
            ["neutral", 0.1, "cooperate", 0.6],
            ["neutral", 0.1, "defect", 0.4],
            ["neutral", 1.5, "cooperate", 0.2],
            ["neutral", 1.5, "defect", 0.8],
            ["anger", 0.1, "cooperate", 0.2],
            ["anger", 0.1, "defect", 0.8],
            ["anger", 1.5, "cooperate", 0.2],
            ["anger", 1.5, "defect", 0.8],
        ],
    )

    # Newer run: option1 neutral mean=(0.6+0.2)/2=0.4, anger mean=(0.2+0.2)/2=0.2 => delta=-0.2
    _write_csv(
        newer_dir / "summary_choice_ratio.csv",
        ["emotion", "intensity", "option_id", "ratio"],
        [
            ["neutral", 0.1, 1, 0.6],
            ["neutral", 0.1, 2, 0.4],
            ["neutral", 1.5, 1, 0.2],
            ["neutral", 1.5, 2, 0.8],
            ["anger", 0.1, 1, 0.2],
            ["anger", 0.1, 2, 0.8],
            ["anger", 1.5, 1, 0.2],
            ["anger", 1.5, 2, 0.8],
        ],
    )

    out = generate_game_theory_impact_report(root=root)
    assert out.option_csv_path.name == "option_impacted_by_emo_vs_neutral_latest.csv"
    assert out.behavior_csv_path.name == "behavior_impacted_emo_vs_neutral_latest.csv"
    assert out.report_path.name == "game_theory_impact_report.md"

    option_text = out.option_csv_path.read_text(encoding="utf-8")
    behavior_text = out.behavior_csv_path.read_text(encoding="utf-8")
    md = out.report_path.read_text(encoding="utf-8")

    assert "FooModel" in option_text
    assert "FooModel" in behavior_text
    assert "summary_choice_ratio.csv" in md
    assert "summary_behavior_ratio.csv" in md
    assert "FooModel_game_theory_decision_Prisoners_Dilemma_20250102_010101" in md

    # Parse behavior CSV and verify anger delta for cooperate is -0.2
    lines = behavior_text.strip().splitlines()
    header = lines[0].split(",")
    idx_task = header.index("task")
    idx_model = header.index("model")
    idx_behavior = header.index("behavior_label")
    idx_best = header.index("best_emotion")
    idx_best_delta = header.index("best_delta_vs_neutral")

    rows = [l.split(",") for l in lines[1:]]
    cooperate = [
        r
        for r in rows
        if r[idx_task] == "Prisoners_Dilemma"
        and r[idx_model] == "FooModel"
        and r[idx_behavior] == "cooperate"
    ]
    assert len(cooperate) == 1
    assert cooperate[0][idx_best] == "anger"
    assert float(cooperate[0][idx_best_delta]) == pytest.approx(-0.2, abs=1e-6)


def test_report_per_game_tables_not_truncated_by_default(tmp_path: Path) -> None:
    """Ensure per-game sections include all models (not just top N)."""
    root = tmp_path / "results" / "new_game_theory_decision" / "shuffle_choices"

    # Create 12 models for the same task; old behavior would truncate to 10.
    for idx in range(12):
        model = f"Model{idx:02d}"
        run_dir = root / f"{model}_game_theory_decision_Prisoners_Dilemma_20250102_010101"
        _write_csv(
            run_dir / "summary_behavior_ratio.csv",
            ["emotion", "intensity", "behavior_label", "ratio"],
            [
                ["neutral", 0.1, "cooperate", 0.5],
                ["neutral", 0.1, "defect", 0.5],
                ["anger", 0.1, "cooperate", 0.5],
                ["anger", 0.1, "defect", 0.5],
            ],
        )
        _write_csv(
            run_dir / "summary_choice_ratio.csv",
            ["emotion", "intensity", "option_id", "ratio"],
            [
                ["neutral", 0.1, 1, 0.5],
                ["neutral", 0.1, 2, 0.5],
                ["anger", 0.1, 1, 0.5],
                ["anger", 0.1, 2, 0.5],
            ],
        )

    out = generate_game_theory_impact_report(root=root)
    md = out.report_path.read_text(encoding="utf-8")

    # Per-game table should include the last model.
    assert "### Prisoners_Dilemma" in md
    assert "| Model11 |" in md


def test_generate_report_without_behavior_csvs(tmp_path: Path) -> None:
    """`results/new_game_theory/` often lacks behavior ratios; still generate option report."""
    root = tmp_path / "results" / "new_game_theory"
    run_dir = root / "FooModel_game_theory_Prisoners_Dilemma_20250102_010101"
    _write_csv(
        run_dir / "summary_choice_ratio.csv",
        ["emotion", "intensity", "option_id", "ratio"],
        [
            ["neutral", 0.1, 1, 0.7],
            ["neutral", 0.1, 2, 0.3],
            ["anger", 0.1, 1, 0.6],
            ["anger", 0.1, 2, 0.4],
        ],
    )

    out = generate_game_theory_impact_report(root=root)
    assert out.option_csv_path.exists()
    assert out.report_path.exists()
    assert out.behavior_csv_path is None

    md = out.report_path.read_text(encoding="utf-8")
    assert "summary_choice_ratio.csv" in md
    assert "summary_behavior_ratio.csv" in md
    assert "No behavior ratio inputs found" in md


def test_missing_neutral_in_run_is_skipped(tmp_path: Path) -> None:
    root = tmp_path / "results" / "new_game_theory_decision" / "shuffle_choices"
    run_dir = root / "FooModel_game_theory_decision_Prisoners_Dilemma_20250102_010101"
    _write_csv(
        run_dir / "summary_choice_ratio.csv",
        ["emotion", "intensity", "option_id", "ratio"],
        [
            ["anger", 0.1, 1, 0.5],
            ["anger", 0.1, 2, 0.5],
        ],
    )

    valid_dir = root / "BarModel_game_theory_decision_Prisoners_Dilemma_20250102_010102"
    _write_csv(
        valid_dir / "summary_choice_ratio.csv",
        ["emotion", "intensity", "option_id", "ratio"],
        [
            ["neutral", 0.1, 1, 0.7],
            ["neutral", 0.1, 2, 0.3],
            ["anger", 0.1, 1, 0.6],
            ["anger", 0.1, 2, 0.4],
        ],
    )

    out = generate_game_theory_impact_report(root=root)
    md = out.report_path.read_text(encoding="utf-8")
    assert "Skipped runs (missing neutral)" in md


def test_generate_heatmaps_from_evaluated_behavior_and_choice_ratios(tmp_path: Path) -> None:
    """Tests for result_analysis/generate_game_theory_impact_report.py: emit heatmap PNGs from evaluated ratios."""
    root = tmp_path / "results" / "vlm_mm_game_theory_decision" / "sample300"

    run_dir = root / "FooModel_game_theory_decision_Prisoners_Dilemma_20250102_010101"
    _write_csv(
        run_dir / "summary_choice_ratio.csv",
        ["emotion", "intensity", "option_id", "ratio"],
        [
            ["neutral", 0.0, 1, 0.7],
            ["neutral", 0.0, 2, 0.3],
            ["anger", 0.8, 1, 0.4],
            ["anger", 0.8, 2, 0.6],
            ["anger", 1.5, 1, 0.2],
            ["anger", 1.5, 2, 0.8],
            ["happiness", 0.8, 1, 0.8],
            ["happiness", 0.8, 2, 0.2],
            ["happiness", 1.5, 1, 0.9],
            ["happiness", 1.5, 2, 0.1],
        ],
    )
    _write_csv(
        run_dir / "summary_behavior_ratio.csv",
        ["emotion", "intensity", "behavior_label", "ratio"],
        [
            ["neutral", 0.0, "cooperate", 0.7],
            ["neutral", 0.0, "defect", 0.3],
            ["anger", 0.8, "cooperate", 0.4],
            ["anger", 0.8, "defect", 0.6],
            ["anger", 1.5, "cooperate", 0.2],
            ["anger", 1.5, "defect", 0.8],
            ["happiness", 0.8, "cooperate", 0.8],
            ["happiness", 0.8, "defect", 0.2],
            ["happiness", 1.5, "cooperate", 0.9],
            ["happiness", 1.5, "defect", 0.1],
        ],
    )

    out = generate_game_theory_impact_report(root=root)

    assert out.option_csv_path.exists()
    assert out.behavior_csv_path is not None
    assert out.report_path.exists()

    behavior_heatmap = root / "behavior_delta_heatmap_vs_neutral_latest.png"
    option_heatmap = root / "option_delta_heatmap_vs_neutral_latest.png"

    assert behavior_heatmap.exists()
    assert option_heatmap.exists()
    assert behavior_heatmap.stat().st_size > 0
    assert option_heatmap.stat().st_size > 0


def test_generate_report_accepts_behavior_column_from_evaluated_outputs(tmp_path: Path) -> None:
    """Tests for result_analysis/generate_game_theory_impact_report.py: accept evaluated behavior CSV header `behavior`."""
    root = tmp_path / "results" / "vlm_mm_game_theory_decision" / "sample300"
    run_dir = root / "FooModel_game_theory_decision_Prisoners_Dilemma_20250102_010101"

    _write_csv(
        run_dir / "summary_choice_ratio.csv",
        ["emotion", "intensity", "option_id", "ratio"],
        [
            ["neutral", 0.0, 1, 0.7],
            ["neutral", 0.0, 2, 0.3],
            ["anger", 1.5, 1, 0.4],
            ["anger", 1.5, 2, 0.6],
        ],
    )
    _write_csv(
        run_dir / "summary_behavior_ratio.csv",
        ["emotion", "intensity", "behavior", "count", "ratio"],
        [
            ["neutral", 0.0, "cooperate", 70, 0.7],
            ["neutral", 0.0, "defect", 30, 0.3],
            ["anger", 1.5, "cooperate", 40, 0.4],
            ["anger", 1.5, "defect", 60, 0.6],
        ],
    )

    out = generate_game_theory_impact_report(root=root)

    assert out.behavior_csv_path is not None
    behavior_text = out.behavior_csv_path.read_text(encoding="utf-8")
    assert "cooperate" in behavior_text
    assert "defect" in behavior_text
