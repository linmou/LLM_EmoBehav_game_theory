#!/usr/bin/env python3
# Responsible file: result_analysis/repeat_stability_analysis.py
# Purpose: verify repeat-stability analysis selects latest runs, computes per-repeat deltas vs neutral, and flags sign flips for target behaviors.

from __future__ import annotations

from pathlib import Path

import json
import pandas as pd
import pytest

from result_analysis.repeat_stability_analysis import (
    PROSOCIAL_BEHAVIOR_BY_TASK,
    analyze_repeat_stability,
)


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(value) for value in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_raw_results(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_analyze_repeat_stability_flags_flip_against_neutral(tmp_path: Path) -> None:
    root = tmp_path / "results" / "new_game_theory_decision" / "sampling_qwen25_3b_full_300"
    run_dir = root / "FooModel_game_theory_decision_Prisoners_Dilemma_20250102_010101"

    _write_csv(
        run_dir / "summary_behavior_ratio_by_repeat.csv",
        ["emotion", "intensity", "repeat_id", "behavior", "count", "ratio"],
        [
            ["neutral", 0.0, 0, "cooperate", 150, 0.50],
            ["neutral", 0.0, 1, "cooperate", 150, 0.50],
            ["neutral", 0.0, 2, "cooperate", 150, 0.50],
            ["happiness", 1.0, 0, "cooperate", 180, 0.60],
            ["happiness", 1.0, 1, "cooperate", 135, 0.45],
            ["happiness", 1.0, 2, "cooperate", 165, 0.55],
        ],
    )

    out = analyze_repeat_stability(root=root)
    df = pd.read_csv(out.csv_path)

    row = df.iloc[0]
    assert row["task"] == "Prisoners_Dilemma"
    assert row["target_behavior"] == PROSOCIAL_BEHAVIOR_BY_TASK["Prisoners_Dilemma"]
    assert row["repeat_sign_pattern"] == "+-+"
    assert bool(row["flip_across_repeats"])
    assert row["repeat_deltas"] == "[+0.100000,-0.050000,+0.050000]"
    assert row["mean_delta"] == pytest.approx(0.03333333333333333, abs=1e-12)


def test_analyze_repeat_stability_selects_latest_run_dir(tmp_path: Path) -> None:
    root = tmp_path / "results" / "new_game_theory_decision" / "sampling_qwen25_3b_full_300"
    older_dir = root / "FooModel_game_theory_decision_Stag_Hunt_20250101_010101"
    newer_dir = root / "FooModel_game_theory_decision_Stag_Hunt_20250102_010101"

    _write_csv(
        older_dir / "summary_behavior_ratio_by_repeat.csv",
        ["emotion", "intensity", "repeat_id", "behavior", "count", "ratio"],
        [
            ["neutral", 0.0, 0, "cooperate", 150, 0.50],
            ["anger", 1.0, 0, "cooperate", 210, 0.70],
        ],
    )
    _write_csv(
        newer_dir / "summary_behavior_ratio_by_repeat.csv",
        ["emotion", "intensity", "repeat_id", "behavior", "count", "ratio"],
        [
            ["neutral", 0.0, 0, "cooperate", 150, 0.50],
            ["anger", 1.0, 0, "cooperate", 120, 0.40],
        ],
    )

    out = analyze_repeat_stability(root=root)
    df = pd.read_csv(out.csv_path)

    row = df.iloc[0]
    assert row["task"] == "Stag_Hunt"
    assert row["source_run_dir"].endswith("20250102_010101")
    assert row["repeat_deltas"] == "[-0.100000]"


def test_analyze_repeat_stability_writes_summary_report(tmp_path: Path) -> None:
    root = tmp_path / "results" / "new_game_theory_decision" / "sampling_qwen25_15b_full_300"
    run_dir = root / "FooModel_game_theory_decision_Trust_Game_Trustor_20250102_010101"

    _write_csv(
        run_dir / "summary_behavior_ratio_by_repeat.csv",
        ["emotion", "intensity", "repeat_id", "behavior", "count", "ratio"],
        [
            ["neutral", 0.0, 0, "trust_high", 120, 0.40],
            ["neutral", 0.0, 1, "trust_high", 120, 0.40],
            ["fear", 1.2, 0, "trust_high", 90, 0.30],
            ["fear", 1.2, 1, "trust_high", 60, 0.20],
        ],
    )

    out = analyze_repeat_stability(root=root)
    report = out.report_path.read_text(encoding="utf-8")

    assert "Repeat Stability Report" in report
    assert "flip_across_repeats" in report
    assert "Trust_Game_Trustor" in report
    assert "fear" in report


def test_analyze_repeat_stability_includes_significance_from_raw_results(tmp_path: Path) -> None:
    root = tmp_path / "results" / "new_game_theory_decision" / "sampling_qwen25_05b_full_300"
    run_dir = root / "FooModel_game_theory_decision_Prisoners_Dilemma_20250102_010101"

    _write_csv(
        run_dir / "summary_behavior_ratio_by_repeat.csv",
        ["emotion", "intensity", "repeat_id", "behavior", "count", "ratio"],
        [
            ["neutral", 0.0, 0, "cooperate", 0, 0.0],
            ["anger", 1.0, 0, "cooperate", 10, 1.0],
        ],
    )

    options = [
        {"id": 1, "text": "Cooperate", "behavior": "cooperate"},
        {"id": 2, "text": "Defect", "behavior": "defect"},
    ]
    raw_rows: list[dict[str, object]] = []
    for item_id in range(10):
        raw_rows.append(
            {
                "emotion": "neutral",
                "intensity": 0.0,
                "item_id": item_id,
                "repeat_id": 0,
                "metadata": {"item_metadata": {"options": options}},
                "response": json.dumps({"decision": "Defect"}),
            }
        )
        raw_rows.append(
            {
                "emotion": "anger",
                "intensity": 1.0,
                "item_id": item_id,
                "repeat_id": 0,
                "metadata": {"item_metadata": {"options": options}},
                "response": json.dumps({"decision": "Cooperate"}),
            }
        )
    _write_raw_results(run_dir / "raw_results.json", raw_rows)

    out = analyze_repeat_stability(root=root)
    df = pd.read_csv(out.csv_path)
    row = df.iloc[0]

    assert row["n_pairs"] == 10
    assert row["pooled_delta"] == pytest.approx(1.0, abs=1e-12)
    assert row["ci_low"] > 0.0
    assert row["p_value"] < 0.01
    assert row["q_value"] < 0.01
    assert bool(row["significant"])
