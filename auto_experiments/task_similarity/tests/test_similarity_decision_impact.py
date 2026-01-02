"""
Responsible: auto_experiments/task_similarity/analyze_similarity_decision_impact.py
Purpose: Tests for parsing and join logic between similarity outputs and decision logs.
"""

import json
from pathlib import Path

import numpy as np


def test_join_similarity_with_decisions(tmp_path: Path):
    from auto_experiments.task_similarity.analyze_similarity_decision_impact import (
        load_similarity_run,
        load_pd_decisions_from_detailed_results,
        load_prompts_from_raw_results,
        join_similarity_with_decisions,
    )

    # Fake similarity run
    run_dir = tmp_path / "sim"
    run_dir.mkdir()
    meta = {
        "intensities": [0.6, 0.8],
        "controlled_layers": [1],
        "measurement_layers": [0, 1, 2],
        "item_ids": [10, 11, 12],
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    cos = np.random.RandomState(0).randn(2, 3, 3).astype(np.float32)
    np.save(run_dir / "cosines.npy", cos)

    # Fake decisions csv
    decision_csv = tmp_path / "detailed_results.csv"
    decision_csv.write_text(
        "\n".join(
            [
                "emotion,intensity,item_id,chosen_behavior",
                "anger,0.6,10,defect",
                "anger,0.6,11,cooperate",
                "anger,0.8,10,cooperate",
                "sadness,0.6,10,defect",
            ]
        ),
        encoding="utf-8",
    )

    sim = load_similarity_run(run_dir)
    decisions = load_pd_decisions_from_detailed_results(decision_csv, emotion="anger")

    raw = tmp_path / "raw_results.json"
    raw.write_text(
        json.dumps(
            [
                {
                    "emotion": "anger",
                    "intensity": 0.6,
                    "item_id": 10,
                    "prompt": "PROMPT_10_0.6",
                    "response": "RESP",
                },
                {
                    "emotion": "sadness",
                    "intensity": 0.6,
                    "item_id": 10,
                    "prompt": "PROMPT_SAD",
                    "response": "RESP",
                },
            ]
        ),
        encoding="utf-8",
    )
    prompts = load_prompts_from_raw_results(raw, emotion="anger")
    assert prompts[(10, 0.6)]["prompt"] == "PROMPT_10_0.6"

    joined = join_similarity_with_decisions(sim, decisions)

    # Only rows where (item_id,intensity) exist in both.
    assert (10, 0.6) in joined
    assert (11, 0.6) in joined
    assert (10, 0.8) in joined
    assert (12, 0.6) not in joined

    # Ensure defect label is correct
    assert joined[(10, 0.6)]["defect"] == 1
    assert joined[(11, 0.6)]["defect"] == 0
