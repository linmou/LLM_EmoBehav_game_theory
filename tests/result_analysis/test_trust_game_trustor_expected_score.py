"""
tests/result_analysis/test_trust_game_trustor_expected_score.py
Purpose: TDD for report-driven Trust Game (Trustor) expected-score deltas vs neutral.
Targets: result_analysis/trust_game_expected_score.py (Trustor)
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def test_generates_expected_score_outputs_from_report(tmp_path: Path) -> None:
    _ensure_repo_on_path()
    from result_analysis.trust_game_expected_score import TRUSTOR_SPEC, run_from_report

    base_out = tmp_path / "results" / "series"
    run_dir = base_out / "Qwen2.5-0.5B-Instruct_game_theory_decision_Trust_Game_Trustor_20250101_000000"
    run_dir.mkdir(parents=True)

    # Minimal raw_results.json with per-item option->behavior mapping.
    raw = [
        {
            "emotion": "neutral",
            "intensity": 0.0,
            "item_id": "itemA:trustor",
            "task_name": "Trust_Game_Trustor",
            "response": {"decision": "Option 1"},
            "score": 1.0,
            "repeat_id": 0,
            "metadata": {
                "item_metadata": {
                    "options": [
                        {"id": 1, "text": "x", "behavior": "trust_low"},
                        {"id": 2, "text": "y", "behavior": "trust_high"},
                        {"id": 3, "text": "z", "behavior": "trust_none"},
                    ]
                }
            },
            "error": None,
        },
        # Two non-neutral conditions with identical delta (+1), to ensure tie-breaking is deterministic.
        {
            "emotion": "happiness",
            "intensity": 1.0,
            "item_id": "itemA:trustor",
            "task_name": "Trust_Game_Trustor",
            "response": {"decision": "Option 2"},
            "score": 2.0,
            "repeat_id": 0,
            "metadata": {
                "item_metadata": {
                    "options": [
                        {"id": 1, "text": "x", "behavior": "trust_low"},
                        {"id": 2, "text": "y", "behavior": "trust_high"},
                        {"id": 3, "text": "z", "behavior": "trust_none"},
                    ]
                }
            },
            "error": None,
        },
        {
            "emotion": "anger",
            "intensity": 1.5,
            "item_id": "itemA:trustor",
            "task_name": "Trust_Game_Trustor",
            "response": {"decision": "Option 2"},
            "score": 2.0,
            "repeat_id": 0,
            "metadata": {
                "item_metadata": {
                    "options": [
                        {"id": 1, "text": "x", "behavior": "trust_low"},
                        {"id": 2, "text": "y", "behavior": "trust_high"},
                        {"id": 3, "text": "z", "behavior": "trust_none"},
                    ]
                }
            },
            "error": None,
        },
    ]
    (run_dir / "raw_results.json").write_text(json.dumps(raw))

    report = {
        "series_name": "dummy",
        "experiments": {
            "trustor_run": {
                "benchmark_name": "game_theory_decision_Trust_Game_Trustor",
                "output_dir": str(run_dir),
                "model_name": "/models/Qwen2.5-0.5B-Instruct",
                "status": "completed",
                "error": None,
            },
            "trustee_run": {
                "benchmark_name": "game_theory_decision_Trust_Game_Trustee",
                "output_dir": str(base_out / "other"),
                "model_name": "/models/Qwen2.5-0.5B-Instruct",
                "status": "completed",
                "error": None,
            },
        },
    }
    report_path = base_out / "memory_experiment_series_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report))

    out_dir = base_out
    outputs = run_from_report(report_path=report_path, out_dir=out_dir, spec=TRUSTOR_SPEC)

    # Core CSV should exist and have the expected delta: neutral score=1, anger score=2 => +1.
    df = outputs.item_expected_score_deltas
    row = df[(df["item_id"] == "itemA:trustor") & (df["emotion"] == "anger") & (df["intensity"] == 1.5)].iloc[0]
    assert row["neutral_decision_score"] == 1.0
    assert row["decision_score_mean"] == 2.0
    assert row["delta_decision_score"] == 1.0

    # Tie-break for max increase: prefer smaller intensity, then emotion.
    summ = outputs.item_max_delta_summary
    srow = summ[summ["item_id"] == "itemA:trustor"].iloc[0]
    assert srow["max_inc_intensity"] == 1.0
    assert srow["max_inc_emotion"] == "happiness"
    assert srow["neutral_p_trust_low"] == 1.0
    assert srow["neutral_p_trust_high"] == 0.0
    assert srow["neutral_p_trust_none"] == 0.0

    for path in [
        outputs.item_expected_score_deltas_path,
        outputs.item_max_delta_summary_path,
        outputs.aggregate_by_emotion_intensity_path,
        outputs.aggregate_by_emotion_path,
        outputs.report_md_path,
    ]:
        assert Path(path).exists()
