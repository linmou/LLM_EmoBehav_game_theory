#!/usr/bin/env python3
# Smoke test for PD steering similarity pipeline on fixtures.

from pathlib import Path

import numpy as np

from auto_experiments.layer_vector_sim.pd_steering_similarity import (
    benchmark_io,
    layer_similarity,
    pd_defection_loader,
    sample_grouping,
    steering_loader,
)


def test_pipeline_smoke(tmp_path: Path) -> None:
    fixtures = Path(__file__).parent / "fixtures"

    # Prepare PD defection vectors and steering vectors
    pd_vec_dir = tmp_path / "pd_vectors"
    pd_vec_dir.mkdir(parents=True)
    np.save(pd_vec_dir / "layer_0.npy", np.array([1.0, 0.0], dtype=np.float32))

    steering_vec_dir = tmp_path / "run_root" / "layer_vectors"
    steering_vec_dir.mkdir(parents=True)
    np.save(steering_vec_dir / "layer_0.npy", np.array([1.0, 0.0], dtype=np.float32))

    # Load raw results and group samples
    raw_results = benchmark_io.load_raw_results(fixtures / "raw_results_switchers.json")
    samples = sample_grouping.load_samples(fixtures / "raw_results_switchers.json")
    switchers = sample_grouping.filter_switchers(samples)

    assert raw_results and switchers

    # Fake hidden states: align baseline with pd vector, steered orthogonal
    hidden_baseline = {0: np.array([1.0, 0.0], dtype=np.float32)}
    hidden_steered = {0: np.array([0.0, 1.0], dtype=np.float32)}

    pd_vectors = pd_defection_loader.load_pd_defection_vectors(pd_vec_dir)
    steering_vectors = steering_loader.load_emotion_vectors(steering_vec_dir)
    assert steering_vectors

    records = layer_similarity.compute_similarity_records(
        sample_id=switchers[0].sample_id,
        steering_condition_id="anger_1.0",
        hidden_baseline=hidden_baseline,
        hidden_steered=hidden_steered,
        pd_defection_vectors=pd_vectors,
    )

    assert len(records) == 1
    assert records[0].similarity_delta < 0.0
