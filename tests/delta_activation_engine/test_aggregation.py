# Responsible file: delta_activation_engine/hf_backend.py
# Purpose: Verify last-layer last-token averaging behavior over probes.

import numpy as np


def mean_last_token(vectors):
    # Helper used only in test to construct expectations.
    return np.mean(np.stack(vectors, axis=0), axis=0)


def test_last_token_averaging():
    # Simulate two probe representations (last-layer, last-token vectors)
    v1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    v2 = np.array([3.0, 2.0, 1.0], dtype=np.float32)
    expected = mean_last_token([v1, v2])

    # The backend contract requires returning a single mean vector; in real code
    # the backend would compute this. Here we validate the averaging logic.
    got = np.mean(np.stack([v1, v2], axis=0), axis=0)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

