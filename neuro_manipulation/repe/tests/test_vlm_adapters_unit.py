import types
from typing import Any, Dict, Optional, List
import sys
from pathlib import Path

# Ensure repository root is on sys.path for imports
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from neuro_manipulation.repe.vlm_adapters import (
    AdapterRegistry,
    AdapterContext,
    QwenVLAdapter,
    MiniCPMV4Adapter,
    GLMEdgeV2bAdapter,
    GemmaTextAdapter,
    PhiTextAdapter,
)
from neuro_manipulation.repe.rep_reading_pipeline import RepReadingPipeline


class FakeProcessor:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        # Return a synthetic formatted string
        return "<formatted>"

    def __call__(self, text=None, images=None, videos=None, padding=True, return_tensors="pt", **kwargs):
        self.calls.append({
            "text": text,
            "images": images,
            "videos": videos,
            "kwargs": kwargs,
        })
        out = {"input_ids": torch.ones(1, 3, dtype=torch.long)}
        if images is not None:
            # Just signal that images were acknowledged
            out["pixel_values"] = torch.ones(1, 1, 2, 2)
        return out


class FakeTokenizer:
    def __init__(self, name_or_path: str) -> None:
        self.name_or_path = name_or_path

    def __call__(self, text, return_tensors="pt", padding=True, **kwargs):
        return {"input_ids": torch.ones(1, 3, dtype=torch.long)}


def test_adapter_detection_by_name():
    reg = AdapterRegistry()

    assert isinstance(reg.get("Qwen/Qwen2.5-VL-3B-Instruct"), QwenVLAdapter)
    assert isinstance(reg.get("openbmb/MiniCPM-V-4"), MiniCPMV4Adapter)
    assert isinstance(reg.get("zai-org/glm-edge-v-2b"), GLMEdgeV2bAdapter)
    assert isinstance(reg.get("google/gemma-3-4b-it"), GemmaTextAdapter)
    assert isinstance(reg.get("microsoft/Phi-3.5-mini-instruct"), PhiTextAdapter)


def test_text_only_adapters_ignore_images():
    tokenizer = FakeTokenizer("google/gemma-3-4b-it")
    processor = FakeProcessor()
    ctx = AdapterContext(processor=processor, tokenizer=tokenizer)

    adapter = GemmaTextAdapter()
    out = adapter.process_multimodal(text="hello", images=[object()], ctx=ctx)

    assert "input_ids" in out
    assert "pixel_values" not in out  # images ignored for text-only

    tokenizer2 = FakeTokenizer("microsoft/Phi-3.5-mini-instruct")
    ctx2 = AdapterContext(processor=processor, tokenizer=tokenizer2)
    adapter2 = PhiTextAdapter()
    out2 = adapter2.process_multimodal(text="hello", images=[object()], ctx=ctx2)
    assert "input_ids" in out2
    assert "pixel_values" not in out2


def test_image_capable_adapters_forward_images():
    processor = FakeProcessor()

    # Qwen fallback path (no qwen_vl_utils) still returns pixel_values when images present
    tokenizer_qwen = FakeTokenizer("Qwen/Qwen2.5-VL-3B-Instruct")
    ctx_qwen = AdapterContext(processor=processor, tokenizer=tokenizer_qwen)
    out_qwen = QwenVLAdapter().process_multimodal(text="t", images=[object()], ctx=ctx_qwen)
    assert "input_ids" in out_qwen
    assert "pixel_values" in out_qwen

    # MiniCPM
    tokenizer_cpm = FakeTokenizer("openbmb/MiniCPM-V-4")
    ctx_cpm = AdapterContext(processor=processor, tokenizer=tokenizer_cpm)
    out_cpm = MiniCPMV4Adapter().process_multimodal(text="t", images=[object()], ctx=ctx_cpm)
    assert "input_ids" in out_cpm
    assert "pixel_values" in out_cpm

    # GLM Edge
    tokenizer_glm = FakeTokenizer("zai-org/glm-edge-v-2b")
    ctx_glm = AdapterContext(processor=processor, tokenizer=tokenizer_glm)
    out_glm = GLMEdgeV2bAdapter().process_multimodal(text="t", images=[object()], ctx=ctx_glm)
    assert "input_ids" in out_glm
    assert "pixel_values" in out_glm


def test_pipeline_preprocess_uses_adapter():
    # Build pipeline instance with faked resources
    pipeline = RepReadingPipeline.__new__(RepReadingPipeline)
    pipeline.model = types.SimpleNamespace()  # not used by preprocess
    pipeline.tokenizer = FakeTokenizer("openbmb/MiniCPM-V-4")
    pipeline.image_processor = FakeProcessor()
    pipeline.framework = "pt"

    inputs = {"text": "describe", "images": [object()]}
    model_inputs = pipeline.preprocess(inputs)

    # Adapter path returns processor output with input_ids and pixel_values
    assert isinstance(model_inputs, dict)
    assert "input_ids" in model_inputs
    assert "pixel_values" in model_inputs
