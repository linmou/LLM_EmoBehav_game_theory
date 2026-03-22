# Responsible file: `tests/unit/neuro_manipulation/repe/test_multimodal_pipeline_batching.py`
# Purpose: Multimodal rep-reading must bypass transformers' generic batch collator for image inputs because it crashes on processor outputs.

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import torch
from PIL import Image
from transformers import PreTrainedModel

from neuro_manipulation.repe.rep_reading_pipeline import RepReadingPipeline


class _MockMultimodalModel(PreTrainedModel):
    config = SimpleNamespace(model_type="qwen2_vl")

    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.config = SimpleNamespace(model_type="qwen2_vl")

    def forward(self, **kwargs):
        hidden_states = [torch.randn(1, 10, 8) for _ in range(2)]

        class _Output:
            def __init__(self, states):
                self.hidden_states = states

            def __getitem__(self, key):
                return getattr(self, key)

        return _Output(hidden_states)

    def parameters(self, recurse: bool = True):
        probe = Mock()
        probe.device = torch.device("cpu")
        yield probe


def test_multimodal_pipeline_call_bypasses_transformers_batch_collation() -> None:
    tokenizer = Mock()
    tokenizer.name_or_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    image_processor = Mock(
        return_value={
            "input_ids": torch.ones(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
            "pixel_values": torch.ones(1, 3, 16, 16),
            "image_grid_thw": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }
    )

    pipeline = RepReadingPipeline(
        model=_MockMultimodalModel(),
        tokenizer=tokenizer,
        image_processor=image_processor,
    )

    sample_image = Image.new("RGB", (16, 16), color="red")
    multimodal_inputs = [
        {"images": [sample_image], "text": "emotion test 1"},
        {"images": [sample_image], "text": "emotion test 2"},
    ]

    results = pipeline(
        multimodal_inputs,
        rep_token=-1,
        hidden_layers=[0, 1],
        rep_reader=None,
        batch_size=2,
    )

    assert isinstance(results, list)
    assert len(results) == 2
    assert image_processor.call_count >= 2
