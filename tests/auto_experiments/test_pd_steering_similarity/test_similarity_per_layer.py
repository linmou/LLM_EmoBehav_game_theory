#!/usr/bin/env python3
# Tests for layer_similarity: per-layer similarity delta computation.

import numpy as np

from auto_experiments.layer_vector_sim.pd_steering_similarity import layer_similarity


def test_compute_layer_similarity_records_baseline_and_delta() -> None:
    hidden_baseline = {0: np.array([1.0, 0.0], dtype=np.float32)}
    hidden_steered = {0: np.array([0.0, 1.0], dtype=np.float32)}
    pd_vectors = {0: np.array([1.0, 0.0], dtype=np.float32)}

    records = layer_similarity.compute_similarity_records(
        sample_id="s1",
        steering_condition_id="anger_1.0",
        hidden_baseline=hidden_baseline,
        hidden_steered=hidden_steered,
        pd_defection_vectors=pd_vectors,
    )

    assert len(records) == 1
    rec = records[0]
    assert rec.layer_index == 0
    assert rec.sample_id == "s1"
    assert rec.steering_condition_id == "anger_1.0"
    assert rec.similarity_baseline == 1.0
    assert rec.similarity_steered == 0.0
    assert rec.similarity_delta == -1.0
