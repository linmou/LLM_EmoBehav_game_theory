"""
tests/result_analysis/test_trust_game_trustee_expected_score.py
Purpose: TDD for report-driven Trust Game (Trustee) expected-score deltas vs neutral.
Targets: result_analysis/trust_game_expected_score.py (Trustee)
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
    from result_analysis.trust_game_expected_score import TRUSTEE_SPEC, run_from_report

    base_out = tmp_path / "results" / "series"
    run_dir = base_out / "Qwen2.5-0.5B-Instruct_game_theory_decision_Trust_Game_Trustee_20250101_000000"
    run_dir.mkdir(parents=True)

    # return_none=0, return_medium=1, return_high=2
    raw = [
        {
            "emotion": "neutral",
            "intensity": 0.0,
            "item_id": "itemB:trustee",
            "task_name": "Trust_Game_Trustee",
            "response": {"decision": "Option 1"},
            "score": 1.0,
            "repeat_id": 0,
            "metadata": {
                "item_metadata": {
                    "options": [
                        {"id": 1, "text": "x", "behavior": "return_medium"},
                        {"id": 2, "text": "y", "behavior": "return_high"},
                        {"id": 3, "text": "z", "behavior": "return_none"},
                    ]
                }
            },
            "error": None,
        },
        {
            "emotion": "sadness",
            "intensity": 1.5,
            "item_id": "itemB:trustee",
            "task_name": "Trust_Game_Trustee",
            "response": {"decision": "Option 2"},
            "score": 2.0,
            "repeat_id": 0,
            "metadata": {
                "item_metadata": {
                    "options": [
                        {"id": 1, "text": "x", "behavior": "return_medium"},
                        {"id": 2, "text": "y", "behavior": "return_high"},
                        {"id": 3, "text": "z", "behavior": "return_none"},
                    ]
                }
            },
            "error": None,
        },
        # Tie with same delta to force deterministic selection.
        {
            "emotion": "happiness",
            "intensity": 1.0,
            "item_id": "itemB:trustee",
            "task_name": "Trust_Game_Trustee",
            "response": {"decision": "Option 2"},
            "score": 2.0,
            "repeat_id": 0,
            "metadata": {
                "item_metadata": {
                    "options": [
                        {"id": 1, "text": "x", "behavior": "return_medium"},
                        {"id": 2, "text": "y", "behavior": "return_high"},
                        {"id": 3, "text": "z", "behavior": "return_none"},
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
            "trustee_run": {
                "benchmark_name": "game_theory_decision_Trust_Game_Trustee",
                "output_dir": str(run_dir),
                "model_name": "/models/Qwen2.5-0.5B-Instruct",
                "status": "completed",
                "error": None,
            }
        },
    }
    report_path = base_out / "memory_experiment_series_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report))

    outputs = run_from_report(report_path=report_path, out_dir=base_out, spec=TRUSTEE_SPEC)
    df = outputs.item_expected_score_deltas
    row = df[(df["item_id"] == "itemB:trustee") & (df["emotion"] == "sadness") & (df["intensity"] == 1.5)].iloc[0]
    assert row["neutral_decision_score"] == 1.0
    assert row["decision_score_mean"] == 2.0
    assert row["delta_decision_score"] == 1.0

    summ = outputs.item_max_delta_summary
    srow = summ[summ["item_id"] == "itemB:trustee"].iloc[0]
    assert srow["max_inc_intensity"] == 1.0
    assert srow["max_inc_emotion"] == "happiness"
    assert srow["neutral_p_return_medium"] == 1.0
    assert srow["neutral_p_return_high"] == 0.0
    assert srow["neutral_p_return_none"] == 0.0

    for path in [
        outputs.item_expected_score_deltas_path,
        outputs.item_max_delta_summary_path,
        outputs.aggregate_by_emotion_intensity_path,
        outputs.aggregate_by_emotion_path,
        outputs.report_md_path,
    ]:
        assert Path(path).exists()
