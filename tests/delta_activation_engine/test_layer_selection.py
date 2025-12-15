# Responsible file: delta_activation_engine/hf_backend.py (or layer_utils.py)
# Purpose: Select the middle third of layer indices given total layer count.

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from delta_activation_engine.backends import select_middle_third_layers


@pytest.mark.parametrize(
    "total, expected",
    [
        (12, list(range(4, 8))),
        (24, list(range(8, 16))),
        (1, []),
        (3, [1]),
        (4, [1]),  # parity with experiment.py style: [len//3 : 2*len//3)
    ],
)
def test_select_middle_third_layers(total, expected):
    assert select_middle_third_layers(total) == expected
