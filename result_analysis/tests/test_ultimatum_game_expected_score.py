# Tests `result_analysis/ultimatum_game_expected_score.py`: Ultimatum expected-score deltas vs neutral (proposer+responder).

import json
from pathlib import Path

import pandas as pd


def _write_raw_results(path: Path, role: str) -> None:
    if role == "proposer":
        options = [
            {"id": 1, "behavior": "offer_low"},
            {"id": 2, "behavior": "offer_medium"},
            {"id": 3, "behavior": "offer_high"},
        ]
        neutral_score = 1
        emo_score = 3
    else:
        options = [
            {"id": 1, "behavior": "reject"},
            {"id": 2, "behavior": "accept"},
        ]
        neutral_score = 1
        emo_score = 2

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


def test_ultimatum_expected_score_filters_by_benchmark_and_status(tmp_path: Path) -> None:
    from result_analysis.ultimatum_game_expected_score import (
        PROPOSER_SPEC,
        RESPONDER_SPEC,
        run_from_report,
    )

    proposer_run = tmp_path / "proposer_run"
    responder_run = tmp_path / "responder_run"
    pending_run = tmp_path / "pending_run"
    proposer_run.mkdir()
    responder_run.mkdir()
    pending_run.mkdir()
    _write_raw_results(proposer_run / "raw_results.json", role="proposer")
    _write_raw_results(responder_run / "raw_results.json", role="responder")

    report = {
        "experiments": {
            "p1": {
                "benchmark_name": "X_Ultimatum_Game_Proposer_Y",
                "output_dir": str(proposer_run),
                "model_name": "/models/m",
                "status": "completed",
            },
            "r1": {
                "benchmark_name": "X_Ultimatum_Game_Responder_Y",
                "output_dir": str(responder_run),
                "model_name": "/models/m",
                "status": "completed",
            },
            # Must be ignored (unfinished, may not have raw_results.json).
            "p2": {
                "benchmark_name": "X_Ultimatum_Game_Proposer_Y",
                "output_dir": str(pending_run),
                "model_name": "/models/m",
                "status": "pending",
            },
        }
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))

    out_p = tmp_path / "out_p"
    out_r = tmp_path / "out_r"
    proposer_outputs = run_from_report(report_path=report_path, out_dir=out_p, spec=PROPOSER_SPEC)
    responder_outputs = run_from_report(report_path=report_path, out_dir=out_r, spec=RESPONDER_SPEC)

    df_p = pd.read_csv(proposer_outputs.item_expected_score_deltas_path)
    df_r = pd.read_csv(responder_outputs.item_expected_score_deltas_path)
    assert df_p["emotion"].tolist() == ["anger"]
    assert df_r["emotion"].tolist() == ["anger"]


def test_main_default_runs_both_roles(tmp_path: Path) -> None:
    from result_analysis.ultimatum_game_expected_score import main

    proposer_run = tmp_path / "proposer_run"
    responder_run = tmp_path / "responder_run"
    proposer_run.mkdir()
    responder_run.mkdir()
    _write_raw_results(proposer_run / "raw_results.json", role="proposer")
    _write_raw_results(responder_run / "raw_results.json", role="responder")

    report = {
        "experiments": {
            "p": {
                "benchmark_name": "Ultimatum_Game_Proposer",
                "output_dir": str(proposer_run),
                "model_name": "/m",
                "status": "completed",
            },
            "r": {
                "benchmark_name": "Ultimatum_Game_Responder",
                "output_dir": str(responder_run),
                "model_name": "/m",
                "status": "completed",
            },
        }
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))

    out_dir = tmp_path / "out"
    rc = main(["--report", str(report_path), "--out_dir", str(out_dir)])
    assert rc == 0

    assert (out_dir / "ultimatum_proposer_item_expected_score_delta_vs_neutral.csv").exists()
    assert (out_dir / "ultimatum_responder_item_expected_score_delta_vs_neutral.csv").exists()

