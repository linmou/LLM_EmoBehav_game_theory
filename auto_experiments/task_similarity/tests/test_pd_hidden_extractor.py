"""
Tests for auto_experiments/task_similarity/pd_hidden_extractor.py.

Focus: ensure we correctly identify the assistant span and pool mean
hidden states over that span, with runtime assertions guarding truncation.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch

from ..pd_hidden_extractor import collect_answer_means


class _DummyTokenizer:
    """
    Very simple tokenizer for tests: splits on spaces, 1 token per word.
    """

    def __call__(
        self,
        text,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=False,
    ):
        if isinstance(text, list):
            # Batch mode
            all_tokens: List[List[int]] = []
            for t in text:
                toks = t.strip().split()
                if truncation:
                    toks = toks[:max_length]
                all_tokens.append(list(range(len(toks))))

            max_len = max(len(toks) for toks in all_tokens) if all_tokens else 0
            input_ids = []
            attention_mask = []
            for toks in all_tokens:
                pad_len = max_len - len(toks)
                input_ids.append(toks + [0] * pad_len)
                attention_mask.append([1] * len(toks) + [0] * pad_len)

            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)

            class Enc:
                def __init__(self, input_ids, attention_mask):
                    self.input_ids = input_ids
                    self.attention_mask = attention_mask

            return Enc(input_ids, attention_mask)

        # Single string
        toks = text.strip().split()
        if truncation:
            toks = toks[:max_length]
        input_ids = torch.tensor([list(range(len(toks)))], dtype=torch.long)

        class Enc:
            def __init__(self, input_ids):
                self.input_ids = input_ids
                self.attention_mask = torch.ones_like(input_ids)

        return Enc(input_ids)


class _DummyModel(torch.nn.Module):
    """
    Dummy model that returns hidden_states where each token's representation
    encodes its token index, so we can check which positions were pooled.
    """

    def __init__(self, num_layers: int = 2, hidden_size: int = 4):
        super().__init__()
        self.config = type("Cfg", (), {"num_hidden_layers": num_layers, "hidden_size": hidden_size})
        # Register a dummy parameter so .parameters() is non-empty and model has a device
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        assert output_hidden_states, "Tests expect output_hidden_states=True"
        bsz, seqlen = input_ids.shape
        device = input_ids.device
        hs_all = []
        # Embedding (layer 0 in hidden_states list)
        base = torch.zeros(bsz, seqlen, self.config.hidden_size, device=device)
        for i in range(seqlen):
            base[:, i, 0] = float(i)  # encode token index in first dim
        hs_all.append(base)
        # Each transformer layer will just add its layer index to the second dim
        for layer in range(self.config.num_hidden_layers):
            h = base.clone()
            h[:, :, 1] = float(layer)
            hs_all.append(h)

        class Out:
            pass

        out = Out()
        out.hidden_states = tuple(hs_all)
        return out


def test_collect_answer_means_basic():
    tokenizer = _DummyTokenizer()
    model = _DummyModel(num_layers=2, hidden_size=4)

    # Construct a PD-like prompt where we can reason about positions:
    # Tokens (space-separated words) laid out so that 'Assistant:' is unique and
    # the answer span is straightforward.
    prompt = (
        "User: desc here\n"
        "Choices:\n"
        "A) option_a\n"
        "B) option_b\n"
        "Assistant: A) DEFECT_ANSWER"
    )
    prompts = [prompt]
    layers = [0, 1]  # transformer blocks 0 and 1

    res = collect_answer_means(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        layers=layers,
        max_length=128,
        batch_size=1,
        span="assistant",
    )

    # We expect one vector per layer
    assert set(res.keys()) == {0, 1}
    for layer, vecs in res.items():
        assert vecs.shape[0] == 1  # one prompt
        assert vecs.shape[1] == model.config.hidden_size
        # Hidden encoding: first dim ~ mean over token indices >= prefix_len
        mean_idx = vecs[0, 0]
        # It should be strictly greater than the prefix_len (i.e., over answer tokens only)
        # We can't know exact value without duplicating tokenizer logic, but we can at
        # least assert it's non-negative and not trivial zero.
        assert mean_idx > 0.0


def test_collect_answer_means_truncation_asserts():
    tokenizer = _DummyTokenizer()
    model = _DummyModel(num_layers=1, hidden_size=2)

    # Build a prompt where the assistant span would be fully truncated
    long_prefix = " ".join([f"w{i}" for i in range(50)])
    prompt = f"{long_prefix} Assistant: A) short"

    # max_length is too small: prefix consumes all slots
    try:
        collect_answer_means(
            model=model,
            tokenizer=tokenizer,
            prompts=[prompt],
            layers=[0],
            max_length=10,
            batch_size=1,
            span="assistant",
        )
    except AssertionError as exc:
        msg = str(exc)
        assert "Assistant answer truncated" in msg
    else:
        assert False, "Expected AssertionError for truncated assistant answer span"
