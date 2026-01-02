"""
Responsible: auto_experiments/task_similarity/run_pd_defection_experiment.py
Purpose: Regression test for `_decision_rate` to ensure it uses the last non-pad
         token logits (attention_mask) rather than `logits[:, -1, :]`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch

from .. import run_pd_defection_experiment as mod


@dataclass
class _Meta:
    description: str
    opt_a: str
    opt_b: str
    defect_label: str


@dataclass
class _Pair:
    meta: _Meta


class _DummyTokenizer:
    def __call__(
        self,
        texts: Sequence[str],
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 256,
        add_special_tokens: bool = False,
    ) -> Dict[str, torch.Tensor]:
        del return_tensors, padding, truncation, max_length, add_special_tokens

        # Two sequences, padded to length 3.
        # seq0: [1, 2, 0] with mask [1, 1, 0]  -> last non-pad index = 1
        # seq1: [1, 2, 3] with mask [1, 1, 1]  -> last non-pad index = 2
        input_ids = torch.tensor([[1, 2, 0], [1, 2, 3]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **_: Any):
        del input_ids, attention_mask
        # vocab size 5; we care about tokens 0(A) and 1(B) by label_to_token below.
        # For seq0, the pad-position (-1) should strongly prefer A (wrong),
        # while the last non-pad position should strongly prefer B (correct).
        # For seq1, last position prefers A (correct for defect_label="A").
        logits = torch.zeros((2, 3, 5), dtype=torch.float32)

        # seq0: position 1 prefers B
        logits[0, 1, 1] = 10.0
        logits[0, 1, 0] = -10.0
        # seq0: position 2 (PAD) prefers A
        logits[0, 2, 0] = 10.0
        logits[0, 2, 1] = -10.0

        # seq1: position 2 prefers A
        logits[1, 2, 0] = 10.0
        logits[1, 2, 1] = -10.0

        return type("Out", (), {"logits": logits})


def test_decision_rate_uses_last_nonpad_token() -> None:
    model = _DummyModel()
    tokenizer = _DummyTokenizer()

    pairs: List[_Pair] = [
        _Pair(meta=_Meta(description="d0", opt_a="a0", opt_b="b0", defect_label="B")),
        _Pair(meta=_Meta(description="d1", opt_a="a1", opt_b="b1", defect_label="A")),
    ]
    label_to_token = {"A": 0, "B": 1}

    # If `_decision_rate` incorrectly uses logits[:, -1, :], seq0 will be wrong
    # (it will look at PAD position and pick A). Correct behavior yields 2/2.
    rate = mod._decision_rate(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        label_to_token=label_to_token,
        batch_size=2,
        max_length=16,
    )
    assert rate == 1.0

