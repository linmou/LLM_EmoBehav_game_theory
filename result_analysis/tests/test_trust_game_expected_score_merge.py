# Tests `result_analysis/trust_game_expected_score.py`: shared logic for trustor + trustee.

import json
from pathlib import Path

import pandas as pd


def _write_raw_results(path: Path, role: str) -> None:
    if role == "trustor":
        options = [
            {"id": 1, "behavior": "trust_none"},
            {"id": 2, "behavior": "trust_low"},
            {"id": 3, "behavior": "trust_high"},
        ]
        neutral_score = 1
        emo_score = 3
    else:
        options = [
            {"id": 1, "behavior": "return_none"},
            {"id": 2, "behavior": "return_medium"},
            {"id": 3, "behavior": "return_high"},
        ]
        neutral_score = 1
        emo_score = 3

    rows = [
        {
            "item_id": 10,
            "emotion": "neutral",
            "intensity": 0.0,
            "score": neutral_score,
            "metadata": {"item_metadata": {"options": options}},
        },
        {
            "item_id": 10,
            "emotion": "anger",
            "intensity": 1.0,
            "score": emo_score,
            "metadata": {"item_metadata": {"options": options}},
        },
    ]
    path.write_text(json.dumps(rows))


def test_trust_game_expected_score_filters_by_benchmark_name(tmp_path: Path) -> None:
    from result_analysis.trust_game_expected_score import TRUSTEE_SPEC, TRUSTOR_SPEC, run_from_report

    trustor_run = tmp_path / "trustor_run"
    trustee_run = tmp_path / "trustee_run"
    pending_run = tmp_path / "pending_run"
    trustor_run.mkdir()
    trustee_run.mkdir()
    pending_run.mkdir()
    _write_raw_results(trustor_run / "raw_results.json", role="trustor")
    _write_raw_results(trustee_run / "raw_results.json", role="trustee")

    report = {
        "experiments": {
            "t1": {
                "benchmark_name": "X_Trust_Game_Trustor_Y",
                "output_dir": str(trustor_run),
                "model_name": "/models/m",
                "status": "completed",
            },
            "t2": {
                "benchmark_name": "X_Trust_Game_Trustee_Y",
                "output_dir": str(trustee_run),
                "model_name": "/models/m",
                "status": "completed",
            },
            # Unfinished experiments must be ignored (may not have raw_results.json yet).
            "t3": {
                "benchmark_name": "X_Trust_Game_Trustor_Y",
                "output_dir": str(pending_run),
                "model_name": "/models/m",
                "status": "pending",
            },
        }
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))

    out_trustor = tmp_path / "out_trustor"
    out_trustee = tmp_path / "out_trustee"

    trustor_outputs = run_from_report(report_path=report_path, out_dir=out_trustor, spec=TRUSTOR_SPEC)
    trustee_outputs = run_from_report(report_path=report_path, out_dir=out_trustee, spec=TRUSTEE_SPEC)

    assert Path(trustor_outputs.item_expected_score_deltas_path).name.startswith("trustor_")
    assert Path(trustee_outputs.item_expected_score_deltas_path).name.startswith("trustee_")

    df_trustor = pd.read_csv(trustor_outputs.item_expected_score_deltas_path)
    df_trustee = pd.read_csv(trustee_outputs.item_expected_score_deltas_path)
    assert df_trustor["emotion"].tolist() == ["anger"]
    assert df_trustee["emotion"].tolist() == ["anger"]
