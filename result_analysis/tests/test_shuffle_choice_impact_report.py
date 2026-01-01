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
    assert out.option_intensity_csv_path.name == "option_intensity_impacted_by_emo_vs_neutral_latest.csv"
    assert out.behavior_intensity_csv_path.name == "behavior_intensity_impacted_emo_vs_neutral_latest.csv"
    assert out.report_path.name == "game_theory_impact_report.md"

    option_text = out.option_csv_path.read_text(encoding="utf-8")
    behavior_text = out.behavior_csv_path.read_text(encoding="utf-8")
    option_intensity_text = out.option_intensity_csv_path.read_text(encoding="utf-8")
    behavior_intensity_text = out.behavior_intensity_csv_path.read_text(encoding="utf-8")
    md = out.report_path.read_text(encoding="utf-8")

    assert "FooModel" in option_text
    assert "FooModel" in behavior_text
    assert "FooModel" in option_intensity_text
    assert "FooModel" in behavior_intensity_text
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


def test_generate_shuffle_choice_impact_report_includes_intensity_deltas(tmp_path: Path) -> None:
    """Ensure intensity-aware output preserves per-intensity deltas (vs neutral mean)."""
    root = tmp_path / "results" / "new_game_theory_decision" / "shuffle_choices"
    run_dir = root / "FooModel_game_theory_decision_Prisoners_Dilemma_20250102_010101"

    _write_csv(
        run_dir / "summary_choice_ratio.csv",
        ["emotion", "intensity", "option_id", "ratio"],
        [
            ["neutral", 0.1, 1, 0.4],
            ["neutral", 1.5, 1, 0.4],
            ["anger", 0.1, 1, 0.0],  # delta -0.4
            ["anger", 1.5, 1, 0.8],  # delta +0.4
        ],
    )

    out = generate_game_theory_impact_report(root=root)
    text = out.option_intensity_csv_path.read_text(encoding="utf-8").strip().splitlines()
    header = text[0].split(",")
    idx_task = header.index("task")
    idx_model = header.index("model")
    idx_option = header.index("option_id")
    idx_emotion = header.index("emotion")
    idx_delta_range = header.index("delta_range_across_intensity")
    idx_deltas = header.index("deltas_by_intensity")

    rows = [l.split(",") for l in text[1:]]
    matches = [
        r
        for r in rows
        if r[idx_task] == "Prisoners_Dilemma"
        and r[idx_model] == "FooModel"
        and r[idx_option] == "1"
        and r[idx_emotion] == "anger"
    ]
    assert len(matches) == 1
    assert float(matches[0][idx_delta_range]) == pytest.approx(0.8, abs=1e-6)
    # stable ordering by intensity asc
    assert matches[0][idx_deltas] == "0.1:-0.400000;1.5:+0.400000"


def test_generate_report_falls_back_when_significance_map_empty(tmp_path: Path) -> None:
    """result_analysis/generate_game_theory_impact_report.py: avoid blank per-game delta cells when sig extraction fails."""
    root = tmp_path / "results" / "new_game_theory" / "shuffle_choices"
    run_dir = root / "FooModel_game_theory_TestGame_20250102_010101"

    # Summary ratios are present and should drive non-empty deltas.
    _write_csv(
        run_dir / "summary_behavior_ratio.csv",
        ["emotion", "intensity", "behavior", "ratio"],
        [
            ["neutral", 1.0, "offer_high", 0.8],
            ["anger", 1.0, "offer_high", 0.2],
        ],
    )
    _write_csv(
        run_dir / "summary_choice_ratio.csv",
        ["emotion", "intensity", "option_id", "ratio"],
        [
            ["neutral", 1.0, 1, 0.8],
            ["anger", 1.0, 1, 0.2],
        ],
    )

    # raw_results.json exists but is intentionally unparsable for *neutral*,
    # causing an empty significance map (before the fallback fix).
    raw = [
        {
            "emotion": "neutral",
            "item_id": 1,
            "repeat_id": 1,
            "metadata": {
                "item_metadata": {
                    "options": [
                        {"id": 1, "text": "A", "behavior": "offer_high"},
                    ]
                }
            },
            "response": "NOT_AN_OPTION",
        },
        {
            "emotion": "anger",
            "item_id": 1,
            "repeat_id": 1,
            "metadata": {
                "item_metadata": {
                    "options": [
                        {"id": 1, "text": "A", "behavior": "offer_high"},
                    ]
                }
            },
            "response": "A",
        },
    ]
    import json

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "raw_results.json").write_text(json.dumps(raw), encoding="utf-8")

    out = generate_game_theory_impact_report(root=root)
    behavior_csv = out.behavior_csv_path.read_text(encoding="utf-8")
    assert "offer_high" in behavior_csv
    # Ensure the "all_emotion_deltas_vs_neutral" cell is not blank.
    assert "anger:" in behavior_csv
    md = out.report_path.read_text(encoding="utf-8")
    assert "| FooModel | offer_high |" in md
    assert "anger:" in md


def test_generate_report_writes_heatmaps_when_enabled(tmp_path: Path) -> None:
    """result_analysis/generate_game_theory_impact_report.py: write heatmap PNGs when requested."""
    root = tmp_path / "results" / "new_game_theory" / "shuffle_choices"
    run_dir = root / "FooModel_game_theory_TestGame_20250102_010101"

    _write_csv(
        run_dir / "summary_choice_ratio.csv",
        ["emotion", "intensity", "option_id", "ratio"],
        [
            ["neutral", 1.0, 1, 0.8],
            ["neutral", 1.0, 2, 0.2],
            ["anger", 1.0, 1, 0.2],
            ["anger", 1.0, 2, 0.8],
        ],
    )
    _write_csv(
        run_dir / "summary_behavior_ratio.csv",
        ["emotion", "intensity", "behavior", "ratio"],
        [
            ["neutral", 1.0, "offer_high", 0.8],
            ["neutral", 1.0, "offer_low", 0.2],
            ["anger", 1.0, "offer_high", 0.2],
            ["anger", 1.0, "offer_low", 0.8],
        ],
    )

    out_dir = tmp_path / "out"
    out = generate_game_theory_impact_report(root=root, out_dir=out_dir, write_heatmaps=True)
    assert out.heatmaps_dir is not None
    assert out.heatmaps_dir.exists()

    # Expect at least one option + one behavior heatmap.
    heatmaps = sorted(p.name for p in out.heatmaps_dir.glob("*.png"))
    assert any("option" in name for name in heatmaps)
    assert any("behavior" in name for name in heatmaps)


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
    run_dir.mkdir(parents=True, exist_ok=True)
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


def test_unknown_emotion_is_hidden_when_ratio_is_tiny(tmp_path: Path) -> None:
    """Ensure `unknown` doesn't show up when its occupancy ratio is <1%."""
    root = tmp_path / "results" / "new_game_theory_decision" / "shuffle_choices"
    run_dir = root / "FooModel_game_theory_decision_Prisoners_Dilemma_20250102_010101"

    _write_csv(
        run_dir / "summary_choice_ratio.csv",
        ["emotion", "intensity", "option_id", "ratio"],
        [
            ["neutral", 0.1, 1, 0.5],
            ["neutral", 0.1, 2, 0.5],
            ["anger", 0.1, 1, 0.6],
            ["anger", 0.1, 2, 0.4],
            ["unknown", 0.1, 1, 0.005],
            ["unknown", 0.1, 2, 0.005],
        ],
    )
    _write_csv(
        run_dir / "summary_behavior_ratio.csv",
        ["emotion", "intensity", "behavior_label", "ratio"],
        [
            ["neutral", 0.1, "cooperate", 0.5],
            ["neutral", 0.1, "defect", 0.5],
            ["anger", 0.1, "cooperate", 0.6],
            ["anger", 0.1, "defect", 0.4],
            ["unknown", 0.1, "cooperate", 0.005],
            ["unknown", 0.1, "defect", 0.005],
        ],
    )

    out = generate_game_theory_impact_report(root=root)
    option_csv = out.option_csv_path.read_text(encoding="utf-8")
    behavior_csv = out.behavior_csv_path.read_text(encoding="utf-8")
    md = out.report_path.read_text(encoding="utf-8")

    assert "unknown:" not in option_csv
    assert "unknown:" not in behavior_csv
    assert "unknown:" not in md


def test_significance_annotation_is_in_per_game_tables_when_raw_results_exist(tmp_path: Path) -> None:
    """Ensure `all emotion deltas` includes stars+CI (and delta-desc ranking) when raw_results.json exists."""
    root = tmp_path / "results" / "new_game_theory_decision" / "shuffle_choices"
    run_dir = root / "FooModel_game_theory_decision_Prisoners_Dilemma_20250102_010101"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build 10 paired items: neutral intensity=0.0 (common in real runs), anger=1.5.
    # Neutral always cooperate (option_id=1), anger always defect (option_id=2).
    raw = []
    for item_id in range(10):
        options = [
            {"id": 1, "text": "Cooperate", "behavior": "cooperate"},
            {"id": 2, "text": "Defect", "behavior": "defect"},
        ]
        raw.append(
            {
                "emotion": "neutral",
                "intensity": 0.0,
                "item_id": item_id,
                "repeat_id": 0,
                "task_name": "Prisoners_Dilemma",
                "response": '{"decision":"Cooperate"}',
                "metadata": {"item_metadata": {"options": options}, "repeat_id": 0, "benchmark": "game_theory_decision"},
                "score": 1.0,
                "error": "",
                "prompt": "",
                "ground_truth": "",
            }
        )
        raw.append(
            {
                "emotion": "anger",
                "intensity": 1.5,
                "item_id": item_id,
                "repeat_id": 0,
                "task_name": "Prisoners_Dilemma",
                "response": '{"decision":"Defect"}',
                "metadata": {"item_metadata": {"options": options}, "repeat_id": 0, "benchmark": "game_theory_decision"},
                "score": 1.0,
                "error": "",
                "prompt": "",
                "ground_truth": "",
            }
        )

    (run_dir / "raw_results.json").write_text(__import__("json").dumps(raw), encoding="utf-8")

    # Also include sadness to confirm delta-desc sorting inside the cell.
    for item_id in range(10):
        options = [
            {"id": 1, "text": "Cooperate", "behavior": "cooperate"},
            {"id": 2, "text": "Defect", "behavior": "defect"},
        ]
        raw.append(
            {
                "emotion": "sadness",
                "intensity": 1.5,
                "item_id": item_id,
                "repeat_id": 0,
                "task_name": "Prisoners_Dilemma",
                "response": '{"decision":"Cooperate"}',
                "metadata": {"item_metadata": {"options": options}, "repeat_id": 0, "benchmark": "game_theory_decision"},
                "score": 1.0,
                "error": "",
                "prompt": "",
                "ground_truth": "",
            }
        )
    (run_dir / "raw_results.json").write_text(__import__("json").dumps(raw), encoding="utf-8")

    _write_csv(
        run_dir / "summary_choice_ratio.csv",
        ["emotion", "intensity", "option_id", "ratio"],
        [
            ["neutral", 1.5, 1, 1.0],
            ["neutral", 1.5, 2, 0.0],
            ["anger", 1.5, 1, 0.0],
            ["anger", 1.5, 2, 1.0],
            ["sadness", 1.5, 1, 1.0],
            ["sadness", 1.5, 2, 0.0],
        ],
    )
    _write_csv(
        run_dir / "summary_behavior_ratio.csv",
        ["emotion", "intensity", "behavior_label", "ratio"],
        [
            ["neutral", 1.5, "cooperate", 1.0],
            ["neutral", 1.5, "defect", 0.0],
            ["anger", 1.5, "cooperate", 0.0],
            ["anger", 1.5, "defect", 1.0],
            ["sadness", 1.5, "cooperate", 1.0],
            ["sadness", 1.5, "defect", 0.0],
        ],
    )

    out = generate_game_theory_impact_report(root=root)
    md = out.report_path.read_text(encoding="utf-8")

    # Expect CI brackets and at least one star for the strong anger shift.
    assert "anger:+1.000" in md
    assert "[" in md and "]" in md
    assert "anger:+1.000" in md
    assert "anger:+1.000!" in md or "anger:+1.000!!" in md or "anger:+1.000!!!" in md

    # Delta-desc order: anger (positive) appears before sadness (0.0) in the same cell.
    line = next(l for l in md.splitlines() if "anger:+1.000" in l)
    assert line.find("anger:+1.000") < line.find("sadness:+0.000")


def test_item_change_block_is_rendered_in_behavior_sections(tmp_path: Path) -> None:
    """Ensure the per-game Behavior sections include the item-change block when detailed_results.csv exists."""
    root = tmp_path / "results" / "new_game_theory_decision" / "shuffle_choices"
    run_dir = root / "FooModel_game_theory_decision_Trust_Game_Trustee_20250102_010101"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Minimal summary_behavior_ratio.csv just to create behavior rows.
    _write_csv(
        run_dir / "summary_behavior_ratio.csv",
        ["emotion", "intensity", "behavior_label", "ratio"],
        [
            ["neutral", 1.5, "return_high", 1.0],
            ["anger", 1.5, "return_high", 0.0],
        ],
    )
    _write_csv(
        run_dir / "summary_choice_ratio.csv",
        ["emotion", "intensity", "option_id", "ratio"],
        [
            ["neutral", 1.5, 1, 1.0],
            ["anger", 1.5, 1, 0.0],
        ],
    )

    # detailed_results.csv provides chosen_behavior for pairing.
    (run_dir / "detailed_results.csv").write_text(
        "emotion,intensity,item_id,task_name,response,ground_truth,score,benchmark,repeat_id,error,chosen_behavior\n"
        "neutral,0.0,0,Trust_Game_Trustee,,,1.0,game_theory_decision,0,,return_high\n"
        "anger,1.5,0,Trust_Game_Trustee,,,1.0,game_theory_decision,0,,return_medium\n",
        encoding="utf-8",
    )

    out = generate_game_theory_impact_report(root=root)
    md = out.report_path.read_text(encoding="utf-8")
    assert "### Trust_Game_Trustee" in md
    assert "#### Item Change vs Neutral" in md
    assert "anger:100.0%" in md
