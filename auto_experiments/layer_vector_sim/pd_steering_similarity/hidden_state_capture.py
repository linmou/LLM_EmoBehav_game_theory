"""
Hidden state capture interface placeholder.
"""

from typing import Dict, Tuple

import numpy as np


def get_hidden_states_for_sample(sample_id: str, steering_condition_id: str) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Placeholder for real hidden-state capture. To be implemented with vLLM hooks.
    Raises NotImplementedError in production path.
    """
    raise NotImplementedError("Hidden state capture not implemented")


def get_hidden_states_from_fixture(
    baseline: Dict[int, np.ndarray],
    steered: Dict[int, np.ndarray],
):
    """
    Return a callable suitable for run_analysis injection to bypass real capture in tests.
    """

    def _fn(sample_id: str, steering_condition_id: str):
        return baseline, steered

    return _fn
