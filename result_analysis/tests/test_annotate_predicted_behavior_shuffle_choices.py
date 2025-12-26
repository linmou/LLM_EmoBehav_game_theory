# result_analysis/tests/test_annotate_predicted_behavior_shuffle_choices.py
# Purpose: catch mislabeling when `shuffle_choices` reuses item_id across emotions/intensities.

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from result_analysis.annotate_predicted_behavior import annotate_csv


def test_annotate_csv_keys_by_emotion_intensity_item_id(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    raw_results_path = run_dir / "raw_results.json"
    scored_csv_path = run_dir / "option_prob_argmax_matches_score.csv"
    out_csv_path = run_dir / "option_prob_argmax_matches_score_annotated.csv"

    # Same item_id, different emotion => options swapped under shuffle_choices.
    raw = [
        {
            "emotion": "anger",
            "intensity": 1.0,
            "item_id": 0,
            "metadata": {
                "item_metadata": {
                    "options": [
                        {"id": 1, "text": "A-coop", "behavior": "cooperate"},
                        {"id": 2, "text": "A-def", "behavior": "defect"},
                    ]
                }
            },
        },
        {
            "emotion": "sadness",
            "intensity": 1.0,
            "item_id": 0,
            "metadata": {
                "item_metadata": {
                    "options": [
                        {"id": 1, "text": "S-def", "behavior": "defect"},
                        {"id": 2, "text": "S-coop", "behavior": "cooperate"},
                    ]
                }
            },
        },
    ]
    raw_results_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    with scored_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "item_id",
                "emotion",
                "intensity",
                "chosen_option_id",
                "predicted_option_id",
                "is_match",
                "p_option_1",
                "p_option_2",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "item_id": 0,
                "emotion": "anger",
                "intensity": 1.0,
                "chosen_option_id": 1,
                "predicted_option_id": 1,
                "is_match": True,
                "p_option_1": 0.9,
                "p_option_2": 0.1,
            }
        )
        writer.writerow(
            {
                "item_id": 0,
                "emotion": "sadness",
                "intensity": 1.0,
                "chosen_option_id": 1,
                "predicted_option_id": 1,
                "is_match": True,
                "p_option_1": 0.9,
                "p_option_2": 0.1,
            }
        )

    annotate_csv(scored_csv_path=scored_csv_path, raw_results_path=raw_results_path, out_csv_path=out_csv_path)

    out_rows = list(csv.DictReader(out_csv_path.open("r", newline="", encoding="utf-8")))
    assert len(out_rows) == 2

    anger = out_rows[0]
    sadness = out_rows[1]

    assert anger["emotion"] == "anger"
    assert anger["predicted_option_text"] == "A-coop"
    assert anger["predicted_behavior"] == "cooperate"
    assert anger["chosen_option_text"] == "A-coop"
    assert anger["chosen_behavior"] == "cooperate"

    assert sadness["emotion"] == "sadness"
    assert sadness["predicted_option_text"] == "S-def"
    assert sadness["predicted_behavior"] == "defect"
    assert sadness["chosen_option_text"] == "S-def"
    assert sadness["chosen_behavior"] == "defect"

