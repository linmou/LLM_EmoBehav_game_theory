"""Tests for `result_analysis/generate_game_theory_impact_report.py`.

This repository's local version does not compute expected-score deltas from `raw_results.json` in the
game-theory impact report. Raw JSON should not crash the report, but it is ignored.
"""

from __future__ import annotations

import json
from pathlib import Path

from result_analysis.generate_game_theory_impact_report import generate_game_theory_impact_report


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(x) for x in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_raw_results_trustor(path: Path) -> None:
    options = [
        {"id": 1, "behavior": "trust_none"},
        {"id": 2, "behavior": "trust_low"},
        {"id": 3, "behavior": "trust_high"},
        {"id": 4, "behavior": "unknown"},
    ]

    # Two items: delta should be +1 at intensity=1.0 and +2 at intensity=2.0.
    rows = [
        {
            "item_id": 1,
            "emotion": "neutral",
            "intensity": 0.0,
            "score": 1,  # trust_none -> 0
            "metadata": {"item_metadata": {"options": options}},
            "error": None,
        },
        {
            "item_id": 2,
            "emotion": "neutral",
            "intensity": 0.0,
            "score": 2,  # trust_low -> 1
            "metadata": {"item_metadata": {"options": options}},
            "error": None,
        },
        # anger @ 1.0: item1 trust_low (1), item2 trust_high (2) => deltas (1-0)=1, (2-1)=1 => mean=1.
        {
            "item_id": 1,
            "emotion": "anger",
            "intensity": 1.0,
            "score": 2,
            "metadata": {"item_metadata": {"options": options}},
            "error": None,
        },
        {
            "item_id": 2,
            "emotion": "anger",
            "intensity": 1.0,
            "score": 3,
            "metadata": {"item_metadata": {"options": options}},
            "error": None,
        },
        # anger @ 2.0: item1 trust_high (2), item2 trust_high (2) => deltas (2-0)=2, (2-1)=1 => mean=1.5.
        {
            "item_id": 1,
            "emotion": "anger",
            "intensity": 2.0,
            "score": 3,
            "metadata": {"item_metadata": {"options": options}},
            "error": None,
        },
        {
            "item_id": 2,
            "emotion": "anger",
            "intensity": 2.0,
            "score": 3,
            "metadata": {"item_metadata": {"options": options}},
            "error": None,
        },
        # Unknown option chosen: must be dropped, not crash.
        {
            "item_id": 1,
            "emotion": "anger",
            "intensity": 2.0,
            "score": 4,
            "metadata": {"item_metadata": {"options": options}},
            "error": None,
        },
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_raw_results_json_is_ignored_by_report(tmp_path: Path) -> None:
    root = tmp_path / "results" / "new_game_theory_decision" / "shuffle_choices"
    run_dir = root / "FooModel_game_theory_decision_Trust_Game_Trustor_20250102_010101"

    _write_csv(
        run_dir / "summary_choice_ratio.csv",
        ["emotion", "intensity", "option_id", "ratio"],
        [
            ["neutral", 0.0, 1, 1.0],
            ["anger", 1.0, 1, 1.0],
        ],
    )
    _write_csv(
        run_dir / "summary_behavior_ratio.csv",
        ["emotion", "intensity", "behavior_label", "ratio"],
        [
            ["neutral", 0.0, "trust_none", 1.0],
            ["anger", 1.0, "trust_low", 1.0],
        ],
    )
    _write_raw_results_trustor(run_dir / "raw_results.json")

    out = generate_game_theory_impact_report(root=root)

    md = out.report_path.read_text(encoding="utf-8")
    assert "Expected-Score Effects" not in md
    assert not (root / "expected_score_delta_vs_neutral_by_emotion_intensity_latest.csv").exists()
