# Tests for result_analysis/postprocess_prob_argmax_from_existing_csv.py
# Purpose: generate behavior match + distributions from an existing scored CSV.

from __future__ import annotations

import csv
import json
from pathlib import Path

from result_analysis.postprocess_prob_argmax_from_existing_csv import postprocess_run_dir


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_postprocess_creates_renamed_enriched_csv_and_summaries(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    raw = [
        {
            "emotion": "anger",
            "intensity": 1.0,
            "item_id": 1,
            "prompt": "p",
            "metadata": {
                "item_metadata": {
                    "options": [
                        {"id": 1, "text": "A", "behavior": "cooperate"},
                        {"id": 2, "text": "B", "behavior": "defect"},
                    ]
                }
            },
            "score": 2,
        }
    ]
    (run_dir / "raw_results.json").write_text(json.dumps(raw), encoding="utf-8")

    _write_csv(
        run_dir / "option_prob_argmax_matches_score.csv",
        [
            {
                "item_id": 1,
                "emotion": "anger",
                "intensity": 1.0,
                "chosen_option_id": 2,
                "predicted_option_id": 1,
                "is_match": False,
                "p_option_1": 0.7,
                "p_option_2": 0.3,
            }
        ],
    )

    out = postprocess_run_dir(run_dir=run_dir)

    assert (run_dir / "prob_argmax_matches_score.csv").exists()
    assert (run_dir / "behavior_prob_argmax_matches_score.csv").exists()
    assert (run_dir / "summary_predicted_option_argmax_ratio.csv").exists()
    assert (run_dir / "summary_predicted_behavior_argmax_ratio.csv").exists()
    assert out == run_dir / "prob_argmax_matches_score.csv"

    # Ensure behavior columns are present in the renamed main CSV.
    with (run_dir / "prob_argmax_matches_score.csv").open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert rows[0]["predicted_behavior"] == "cooperate"
    assert rows[0]["chosen_behavior"] == "defect"

