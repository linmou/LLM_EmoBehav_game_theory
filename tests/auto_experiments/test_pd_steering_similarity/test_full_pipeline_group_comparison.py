#!/usr/bin/env python3
# Integration test for group comparison summaries.

from pathlib import Path
import numpy as np

from auto_experiments.layer_vector_sim.pd_steering_similarity import (
    group_aggregation,
    layer_similarity,
    pd_defection_loader,
    sample_grouping,
    output_writer,
)


def test_group_comparison_pipeline(tmp_path: Path) -> None:
    fixtures = Path(__file__).parent / "fixtures"
    samples = sample_grouping.load_samples(fixtures / "raw_results_groups.json")
    switchers = sample_grouping.filter_switchers(samples)
    non_switchers = sample_grouping.filter_non_switchers(samples)
    group_labels = {s.sample_id: "switcher" for s in switchers}
    group_labels.update({s.sample_id: "non-switcher" for s in non_switchers})

    pd_vec_dir = tmp_path / "pd_vectors"
    pd_vec_dir.mkdir(parents=True)
    np.save(pd_vec_dir / "layer_0.npy", np.array([1.0, 0.0], dtype=np.float32))
    pd_vectors = pd_defection_loader.load_pd_defection_vectors(pd_vec_dir)

    hidden_baseline = {0: np.array([1.0, 0.0], dtype=np.float32)}
    hidden_steered = {0: np.array([0.0, 1.0], dtype=np.float32)}

    records = []
    for sample in samples:
        records.extend(
            layer_similarity.compute_similarity_records(
                sample_id=sample.sample_id,
                steering_condition_id="anger_1.0",
                hidden_baseline=hidden_baseline,
                hidden_steered=hidden_steered,
                pd_defection_vectors=pd_vectors,
            )
        )

    summaries = group_aggregation.aggregate_by_group(records, group_labels)
    assert summaries
    out_path = output_writer.write_group_summaries(summaries, tmp_path)
    assert out_path.exists()
    summary_map = {(s.layer_index, s.group_label): s for s in summaries}
    assert summary_map[(0, "switcher")].n_samples == len(switchers)
