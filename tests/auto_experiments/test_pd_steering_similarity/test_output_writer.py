#!/usr/bin/env python3
# Tests for output_writer: ensure similarity records are persisted.

from pathlib import Path
import json

from auto_experiments.layer_vector_sim.pd_steering_similarity import layer_similarity
from auto_experiments.layer_vector_sim.pd_steering_similarity import output_writer


def test_write_similarity_records(tmp_path: Path) -> None:
    records = [
        layer_similarity.LayerSimilarityRecord(
            sample_id="s1",
            steering_condition_id="anger_1.0",
            layer_index=0,
            similarity_baseline=0.5,
            similarity_steered=0.6,
            similarity_delta=0.1,
        )
    ]

    out_path = output_writer.write_similarity_records(records, tmp_path)

    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert len(data) == 1
    assert data[0]["sample_id"] == "s1"
