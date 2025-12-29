"""Tests for WrappedBlock attribute passthrough.

Responsible files:
- neuro_manipulation/repe/rep_control_reading_vec.py

Purpose:
- Ensure wrapper blocks preserve model-layer attributes required by some architectures
  (e.g., Qwen2.5-VL expects `attention_type` on decoder layers).
"""

from __future__ import annotations

import torch


def test_wrapped_block_delegates_attention_type() -> None:
    """I am starting with a failing test. This is the Red phase."""
    from neuro_manipulation.repe.rep_control_reading_vec import WrappedBlock

    class _Block(torch.nn.Module):
        attention_type = "full"

        def forward(self, x):
            return x

    wrapped = WrappedBlock(_Block())
    assert wrapped.attention_type == "full"

