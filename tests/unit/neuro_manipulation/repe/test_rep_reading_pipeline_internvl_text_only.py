# Responsible file: `tests/unit/neuro_manipulation/repe/test_rep_reading_pipeline_internvl_text_only.py`
# Purpose: Text-only rep-reading for InternVL models must bypass the multimodal wrapper and call `language_model` directly because the wrapper unconditionally dereferences image inputs.

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from neuro_manipulation.repe.rep_reading_pipeline import RepReadingPipeline


class _Output:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states

    def __getitem__(self, key):
        return getattr(self, key)


class _InternVLTextOnlyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            model_type="internvl_chat",
            _name_or_path="OpenGVLab/InternVL3-1B",
        )
        self._probe = nn.Parameter(torch.zeros(1))
        self.device = torch.device("cpu")
        self.forward_called = False
        self.language_model = _InternVLLanguageModel()

    def forward(self, *args, **kwargs):
        self.forward_called = True
        raise AssertionError("InternVL multimodal forward should not be used for text-only rep-reading")


class _InternVLLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_input_ids = None
        self.seen_attention_mask = None
        self.seen_use_cache = "unset"

    def forward(
        self,
        input_ids,
        attention_mask=None,
        output_hidden_states=False,
        use_cache=None,
        **kwargs,
    ):
        self.seen_input_ids = input_ids
        self.seen_attention_mask = attention_mask
        self.seen_use_cache = use_cache
        hidden_states = [torch.randn(1, input_ids.shape[1], 8)]
        return _Output(hidden_states)


def test_forward_uses_language_model_for_internvl_text_only() -> None:
    tokenizer = SimpleNamespace(pad_token="[PAD]")
    model = _InternVLTextOnlyModel()
    pipeline = RepReadingPipeline(
        model=model,
        tokenizer=tokenizer,
        image_processor=None,
    )

    model_inputs = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
    }

    with patch("neuro_manipulation.repe.rep_reading_pipeline.torch.no_grad"):
        result = pipeline._forward(
            model_inputs=model_inputs,
            rep_token=-1,
            hidden_layers=[0],
            rep_reader=None,
        )

    assert 0 in result
    assert model.forward_called is False
    assert torch.equal(model.language_model.seen_input_ids, model_inputs["input_ids"])
    assert torch.equal(model.language_model.seen_attention_mask, model_inputs["attention_mask"])
    assert model.language_model.seen_use_cache is False
