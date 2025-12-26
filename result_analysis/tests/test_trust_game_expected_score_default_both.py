# Tests `result_analysis/trust_game_expected_score.py`: default CLI runs both roles.

import json
from pathlib import Path


def _write_raw_results(path: Path, role: str) -> None:
    if role == "trustor":
        options = [
            {"id": 1, "behavior": "trust_none"},
            {"id": 2, "behavior": "trust_low"},
            {"id": 3, "behavior": "trust_high"},
        ]
    else:
        options = [
            {"id": 1, "behavior": "return_none"},
            {"id": 2, "behavior": "return_medium"},
            {"id": 3, "behavior": "return_high"},
        ]

    rows = [
        {
            "item_id": 10,
            "emotion": "neutral",
            "intensity": 0.0,
            "score": 1,
            "metadata": {"item_metadata": {"options": options}},
        },
        {
            "item_id": 10,
            "emotion": "anger",
            "intensity": 1.0,
            "score": 3,
            "metadata": {"item_metadata": {"options": options}},
        },
    ]
    path.write_text(json.dumps(rows))


def test_main_default_runs_both_roles(tmp_path: Path) -> None:
    from result_analysis.trust_game_expected_score import main

    trustor_run = tmp_path / "trustor_run"
    trustee_run = tmp_path / "trustee_run"
    trustor_run.mkdir()
    trustee_run.mkdir()
    _write_raw_results(trustor_run / "raw_results.json", role="trustor")
    _write_raw_results(trustee_run / "raw_results.json", role="trustee")

    report = {
        "experiments": {
            "t1": {"benchmark_name": "Trust_Game_Trustor", "output_dir": str(trustor_run), "model_name": "/m"},
            "t2": {"benchmark_name": "Trust_Game_Trustee", "output_dir": str(trustee_run), "model_name": "/m"},
        }
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))

    out_dir = tmp_path / "out"
    rc = main(["--report", str(report_path), "--out_dir", str(out_dir)])
    assert rc == 0

    assert (out_dir / "trustor_item_expected_score_delta_vs_neutral.csv").exists()
    assert (out_dir / "trustee_item_expected_score_delta_vs_neutral.csv").exists()

