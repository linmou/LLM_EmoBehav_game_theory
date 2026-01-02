#!/usr/bin/env python3
# Tests for auto_experiments.layer_vector_sim.pd_steering_similarity.benchmark_io.

import json
from pathlib import Path

import pytest

from auto_experiments.layer_vector_sim.pd_steering_similarity import benchmark_io


def test_load_raw_results_returns_data(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw_results.json"
    data = [{"id": 1, "choice": "cooperate"}, {"id": 2, "choice": "defect"}]
    raw_path.write_text(json.dumps(data))

    loaded = benchmark_io.load_raw_results(raw_path)

    assert loaded == data


def test_load_raw_results_raises_on_missing(tmp_path: Path) -> None:
    raw_path = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        benchmark_io.load_raw_results(raw_path)
