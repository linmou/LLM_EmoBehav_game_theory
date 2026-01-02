#!/usr/bin/env python3
# Tests for group aggregation of similarity records.

import pytest

from auto_experiments.layer_vector_sim.pd_steering_similarity import group_aggregation, layer_similarity


def test_aggregate_group_summaries() -> None:
    records = [
        layer_similarity.LayerSimilarityRecord(
            sample_id="s1",
            steering_condition_id="anger_1.0",
            layer_index=0,
            similarity_baseline=0.2,
            similarity_steered=0.6,
            similarity_delta=0.4,
        ),
        layer_similarity.LayerSimilarityRecord(
            sample_id="s2",
            steering_condition_id="anger_1.0",
            layer_index=0,
            similarity_baseline=0.1,
            similarity_steered=0.2,
            similarity_delta=0.1,
        ),
    ]

    summaries = group_aggregation.aggregate_by_group(
        records,
        group_labels={"s1": "switcher", "s2": "non-switcher"},
    )

    assert len(summaries) == 2
    summary_map = {(s.layer_index, s.group_label): s for s in summaries}
    assert summary_map[(0, "switcher")].mean_similarity_delta == pytest.approx(0.4)
    assert summary_map[(0, "non-switcher")].mean_similarity_delta == pytest.approx(0.1)
