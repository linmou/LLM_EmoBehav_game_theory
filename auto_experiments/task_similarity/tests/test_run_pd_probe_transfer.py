"""
Responsible: auto_experiments/task_similarity/run_pd_probe_transfer.py
Purpose: Integration test for PD-probe transfer runner (filesystem + CSV outputs).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def test_run_pd_probe_transfer_writes_csv_and_high_auc(tmp_path: Path) -> None:
    from auto_experiments.task_similarity.run_pd_probe_transfer import run_pd_probe_transfer

    run_dir = tmp_path / "run"
    model = "Qwen-Fake"
    seed_dir = run_dir / model / "anger" / "seed_20"
    seed_dir.mkdir(parents=True)
    (run_dir / model / "happiness" / "seed_20").mkdir(parents=True)

    split_manifest = tmp_path / "split_manifest.json"
    split_manifest.write_text(json.dumps({"split_seed": 20, "train_indices": [0, 1, 2, 3], "test_indices": [4, 5]}))

    result_dir = tmp_path / "results"
    result_dir.mkdir()

    # detailed_results.csv labels correspond to item_id, emotion, intensity.
    df = pd.DataFrame(
        [
            {"emotion": "anger", "intensity": 1.0, "item_id": 4, "chosen_behavior": "defect"},
            {"emotion": "anger", "intensity": 1.0, "item_id": 5, "chosen_behavior": "cooperate"},
            {"emotion": "happiness", "intensity": 1.0, "item_id": 4, "chosen_behavior": "cooperate"},
            {"emotion": "happiness", "intensity": 1.0, "item_id": 5, "chosen_behavior": "cooperate"},
        ]
    )
    df.to_csv(result_dir / "detailed_results.csv", index=False)

    meta = {
        "model_path": "/fake/model",
        "emotion": "anger",
        "num_layers": 2,
        "controlled_layers": [0],
        "measurement_layers": [0, 1],
        "intensities": [1.0],
        "dataset_split": "all",
        "split_manifest": str(split_manifest),
        "raw_results_path": str(result_dir),
    }
    (seed_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    # Copy metadata to other emotion dir (runner should not depend on it being different).
    (run_dir / model / "happiness" / "seed_20" / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    # Build deltas: n_int=1, n_items=6, n_layers=2, d=3.
    n_int, n_items, n_layers, d = 1, 6, 2, 3
    delta_pd = np.zeros((n_int, n_items, n_layers, d), dtype=np.float32)
    delta_pd_coop = np.zeros((n_int, n_items, n_layers, d), dtype=np.float32)
    delta_pd[:, :, :, 0] = 1.0
    delta_pd_coop[:, :, :, 0] = -1.0

    delta_anger = np.zeros((n_int, n_items, n_layers, d), dtype=np.float32)
    delta_happy = np.zeros((n_int, n_items, n_layers, d), dtype=np.float32)
    # For anger, item 4 defect-like (+), item 5 cooperate-like (-) at layer 0.
    delta_anger[0, 4, 0, 0] = 1.0
    delta_anger[0, 5, 0, 0] = -1.0
    # Happiness pushes toward cooperate for both items.
    delta_happy[0, 4, 0, 0] = -1.0
    delta_happy[0, 5, 0, 0] = -1.0

    np.save(seed_dir / "delta_pd.npy", delta_pd)
    np.save(seed_dir / "delta_pd_cooperate.npy", delta_pd_coop)
    np.save(seed_dir / "delta_emotion.npy", delta_anger)

    np.save(run_dir / model / "happiness" / "seed_20" / "delta_pd.npy", delta_pd)
    np.save(run_dir / model / "happiness" / "seed_20" / "delta_pd_cooperate.npy", delta_pd_coop)
    np.save(run_dir / model / "happiness" / "seed_20" / "delta_emotion.npy", delta_happy)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    run_pd_probe_transfer(run_dir=run_dir, out_dir=out_dir, n_boot=50, random_seed=0)

    by_layer = pd.read_csv(out_dir / "pd_probe_transfer_auc_by_layer.csv")
    assert set(by_layer.columns) >= {
        "emotion",
        "intensity",
        "layer",
        "auc",
        "ci_low",
        "ci_high",
        "n_test_items",
    }
    # Anger at layer 0 should be perfectly predicted on the 2 test items.
    row = by_layer[(by_layer["emotion"] == "anger") & (by_layer["intensity"] == 1.0) & (by_layer["layer"] == 0)].iloc[
        0
    ]
    assert float(row["auc"]) > 0.99

