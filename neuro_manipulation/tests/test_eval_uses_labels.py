"""
Tests evaluation logic uses labels, not fixed first-element assumption.

Focus: neuro_manipulation/utils.py::test_direction
Purpose: Ensure evaluation picks the labeled-true index within each test pair,
not always index 0, and handles sign scalarization correctly.
"""

from types import SimpleNamespace

import neuro_manipulation.utils as U


class StubPipeline:
    def __call__(self, data, rep_token, hidden_layers, rep_reader, batch_size):
        # Deterministic per-sample projection values for a single layer [-1]
        # Four samples → two pairs: [0.1, 0.2] and [0.3, 0.25]
        vals = [0.1, 0.2, 0.3, 0.25]
        return [{-1: v} for v in vals]


def test_eval_uses_pair_labels_instead_of_first_element():
    pipeline = StubPipeline()
    # Force positive sign so eval_func is max
    rep_reader = SimpleNamespace(direction_signs={-1: 1.0})

    # Two pairs; first pair label = [0,1] (second is true), second pair label = [1,0] (first is true)
    test_data = {
        "data": [object()] * 4,
        "labels": [[0, 1], [1, 0]],
    }

    results, _ = U.test_direction(hidden_layers=[-1], rep_reading_pipeline=pipeline, rep_reader=rep_reader, test_data=test_data)

    # With max as eval_func, pair1 max is 0.2 at index 1 (label=1), pair2 max is 0.3 at index 0 (label=1)
    # Expect perfect correctness when labels are used
    assert results[-1] == 1.0
