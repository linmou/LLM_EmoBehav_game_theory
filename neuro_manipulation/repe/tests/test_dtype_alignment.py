"""
Tests for dtype/device alignment in projection.

Focus file: neuro_manipulation/repe/rep_readers.py
Purpose: Ensure project_onto_direction aligns direction to hidden state tensor
device and dtype to avoid runtime dtype mismatch errors (Half vs Float).
"""

import torch
from neuro_manipulation.repe.rep_readers import project_onto_direction


def test_project_onto_direction_matches_tensor_dtype():
    H = torch.randn(4, 8, dtype=torch.bfloat16)
    direction = torch.randn(8, dtype=torch.float32)

    out = project_onto_direction(H, direction)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == H.dtype
    assert out.shape == (4,)

