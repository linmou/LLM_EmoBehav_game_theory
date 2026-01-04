"""
Responsible: auto_experiments/task_similarity/analyze_similarity_decision_impact.py
Purpose: Ensure optional PD-cooperate similarity tensors are loaded and joined for item-level correlation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def test_load_similarity_run_optional_pd_cooperate_and_pref(tmp_path: Path) -> None:
    from auto_experiments.task_similarity.analyze_similarity_decision_impact import (
        join_similarity_with_decisions,
        load_similarity_run,
    )

    run_dir = tmp_path / "sim"
    run_dir.mkdir()
    meta = {
        "intensities": [0.6],
        "controlled_layers": [1],
        "measurement_layers": [0, 1],
        "item_ids": [10, 11],
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    cos_def = np.array([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32)
    cos_coop = np.array([[[-0.1, -0.2], [-0.3, -0.4]]], dtype=np.float32)
    np.save(run_dir / "cosines.npy", cos_def)
    np.save(run_dir / "cosines_pd_cooperate.npy", cos_coop)
    np.save(run_dir / "pref_cosines.npy", cos_def - cos_coop)

    sim = load_similarity_run(run_dir)
    assert sim.cosines.shape == (1, 2, 2)
    assert sim.cosines_pd_cooperate is not None
    assert sim.pref_cosines is not None

    decisions = {(10, 0.6): "defect", (11, 0.6): "cooperate"}
    joined = join_similarity_with_decisions(sim, decisions)
    assert "cosine_pd_cooperate" in joined[(10, 0.6)]
    assert "pref_cosine" in joined[(10, 0.6)]

