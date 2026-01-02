"""
Responsible: auto_experiments/task_similarity/summarize_similarity_decision_impact.py
Purpose: Unit tests for permutation p-values + BH-FDR helpers used in summary/significance outputs.
"""

import numpy as np


def test_bh_fdr_monotone_and_bounds():
    from auto_experiments.task_similarity.summarize_similarity_decision_impact import bh_fdr

    p = [0.001, 0.01, 0.2, 0.05]
    q = bh_fdr(p)
    assert len(q) == len(p)
    assert all(0.0 <= float(v) <= 1.0 for v in q)
    # smallest p should have smallest q (not necessarily strictly, but should be <=)
    i_min = int(np.argmin(p))
    assert float(q[i_min]) == min(float(v) for v in q)


def test_perm_p_value_detects_strong_signal():
    from auto_experiments.task_similarity.summarize_similarity_decision_impact import perm_p_value_pearson_abs

    rng = np.random.default_rng(0)
    # Construct x that separates y strongly.
    y = np.array([0] * 20 + [1] * 20, dtype=np.float32)
    x = y + 0.01 * rng.normal(size=y.shape[0]).astype(np.float32)

    p = perm_p_value_pearson_abs(x, y, B=2000, seed=0)
    assert 0.0 <= float(p) <= 0.05


def test_perm_p_value_near_uniform_under_null():
    from auto_experiments.task_similarity.summarize_similarity_decision_impact import perm_p_value_pearson_abs

    rng = np.random.default_rng(0)
    x = rng.normal(size=100).astype(np.float32)
    y = rng.integers(0, 2, size=100).astype(np.float32)

    p = perm_p_value_pearson_abs(x, y, B=2000, seed=0)
    assert 0.0 <= float(p) <= 1.0
