#!/usr/bin/env python3
# Tests for emotion-level aggregation and ranking.

from auto_experiments.layer_vector_sim.pd_steering_similarity import emotion_aggregation


def test_rank_emotions_from_group_summaries() -> None:
    summaries = [
        emotion_aggregation.GroupSummaryInput(
            steering_condition_id="anger_1.0",
            mean_similarity_delta=0.5,
        ),
        emotion_aggregation.GroupSummaryInput(
            steering_condition_id="fear_1.0",
            mean_similarity_delta=0.2,
        ),
    ]

    rankings = emotion_aggregation.rank_emotions(summaries)
    assert rankings[0].steering_condition_id == "anger_1.0"
    assert rankings[0].mean_similarity_delta == 0.5
