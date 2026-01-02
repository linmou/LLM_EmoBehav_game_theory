#!/usr/bin/env python3
# Tests for auto_experiments.layer_vector_sim.pd_steering_similarity.similarity_utils.

import numpy as np
import pytest

from auto_experiments.layer_vector_sim.pd_steering_similarity import similarity_utils


def test_cosine_similarity_handles_basic_case() -> None:
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert similarity_utils.cosine_similarity(a, b) == pytest.approx(1.0)


def test_cosine_similarity_handles_zero_vector() -> None:
    a = np.array([0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0], dtype=np.float32)
    assert similarity_utils.cosine_similarity(a, b) == pytest.approx(0.0)


def test_similarity_delta_computes_diff() -> None:
    assert similarity_utils.similarity_delta(0.1, 0.4) == pytest.approx(0.3)


def test_cosine_similarity_handles_negative_alignment() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([-1.0, 0.0], dtype=np.float32)
    assert similarity_utils.cosine_similarity(a, b) == pytest.approx(-1.0)
