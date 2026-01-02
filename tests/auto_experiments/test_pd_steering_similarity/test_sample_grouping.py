#!/usr/bin/env python3
# Tests for sample_grouping: identify switchers vs non-switchers.

from pathlib import Path

from auto_experiments.layer_vector_sim.pd_steering_similarity import sample_grouping


def test_group_samples_identifies_switchers(tmp_path: Path) -> None:
    raw_path = (
        Path(__file__).parent / "fixtures" / "raw_results_switchers.json"
    )
    samples = sample_grouping.load_samples(raw_path)

    switchers = [s.sample_id for s in sample_grouping.filter_switchers(samples)]
    non_switchers = [s.sample_id for s in sample_grouping.filter_non_switchers(samples)]

    assert switchers == ["s1"]
    assert set(non_switchers) == {"s2", "s3"}


def test_group_samples_handles_missing_fields() -> None:
    samples = sample_grouping.load_samples(Path(__file__).parent / "fixtures" / "raw_results_switchers.json")
    # Manually add malformed entry
    samples.append(sample_grouping.PDSample(sample_id="bad", baseline_choice="defect", steered_choice="defect"))
    switchers = sample_grouping.filter_switchers(samples)
    assert any(s.sample_id == "bad" for s in switchers) is False
