#!/usr/bin/env python3
# Edge case tests for PD steering similarity.

from pathlib import Path
import pytest

from auto_experiments.layer_vector_sim.pd_steering_similarity import steering_loader, pd_defection_loader, benchmark_io


def test_steering_loader_raises_on_missing_vectors(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        steering_loader.load_emotion_vectors(tmp_path)


def test_pd_defection_loader_raises_on_missing(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        pd_defection_loader.load_pd_defection_vectors(tmp_path)


def test_benchmark_io_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        benchmark_io.load_raw_results(tmp_path / "missing.json")
