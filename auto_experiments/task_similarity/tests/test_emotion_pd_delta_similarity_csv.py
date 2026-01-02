"""
Responsible: auto_experiments/task_similarity/emotion_pd_delta_similarity.py
Purpose: Validate CSV outputs for emotion-vs-PD delta similarity analysis.
"""

import csv
from pathlib import Path

import numpy as np


def test_write_csv_outputs(tmp_path: Path):
    from auto_experiments.task_similarity.emotion_pd_delta_similarity import write_csv_outputs

    item_ids = [10, 11]
    prompts = ["p0", "p1"]
    intensities = [0.4, 1.0]
    controlled_layers = [1]
    num_layers = 3

    cosines = np.array(
        [
            # intensity 0.4
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            # intensity 1.0
            [[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6]],
        ],
        dtype=np.float32,
    )  # (2 intensities, 2 samples, 3 layers)
    norms_a = np.ones_like(cosines, dtype=np.float32) * 2.0
    norms_b = np.ones_like(cosines, dtype=np.float32) * 3.0

    write_csv_outputs(
        out_dir=tmp_path,
        item_ids=item_ids,
        prompts=prompts,
        intensities=intensities,
        controlled_layers=controlled_layers,
        cosines=cosines,
        delta_norms_anger=norms_a,
        delta_norms_pd=norms_b,
    )

    samples_csv = tmp_path / "samples.csv"
    cos_csv = tmp_path / "cosines.csv"
    assert samples_csv.exists()
    assert cos_csv.exists()

    with samples_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["item_id"] == "10"
    assert rows[0]["prompt"] == "p0"

    with cos_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # 2 samples * 2 intensities * 3 layers
    assert len(rows) == 12
    # Check controlled annotation and one value
    r0 = rows[0]
    assert r0["item_id"] == "10"
    assert r0["intensity"] == "0.4"
    assert r0["layer"] == "0"
    assert r0["controlled"] == "0"
    assert abs(float(r0["cosine"]) - 0.1) < 1e-6

    # Find layer 1 row: controlled
    row_l1 = next(r for r in rows if r["item_id"] == "10" and r["intensity"] == "0.4" and r["layer"] == "1")
    assert row_l1["controlled"] == "1"
