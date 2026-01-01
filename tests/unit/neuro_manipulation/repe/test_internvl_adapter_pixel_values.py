# tests/unit/neuro_manipulation/repe/test_internvl_adapter_pixel_values.py
# Purpose: Ensure InternVL multimodal adapter supplies `pixel_values` even when images are file paths.

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

import importlib.util
import sys


def _load_vlm_adapters_module():
    repo_root = Path(__file__).resolve().parents[4]
    vlm_adapters_path = repo_root / "neuro_manipulation" / "repe" / "vlm_adapters.py"
    spec = importlib.util.spec_from_file_location(
        "vlm_adapters_module_internvl", vlm_adapters_path
    )
    assert spec and spec.loader
    vlm_adapters = importlib.util.module_from_spec(spec)
    sys.modules["vlm_adapters_module_internvl"] = vlm_adapters
    spec.loader.exec_module(vlm_adapters)
    return vlm_adapters


class _FakeTokenizer:
    def __init__(self, name_or_path: str) -> None:
        self.name_or_path = name_or_path

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<IMG_CONTEXT>"
        return 42

    def __call__(self, text, return_tensors="pt", padding=True, **kwargs):
        # Encode IMG_CONTEXT occurrences into id=42 so we can assert adapter inserted tokens
        ids = [1, 2]
        if "<IMG_CONTEXT>" in (text or ""):
            ids.extend([42] * 3)
        ids.append(3)
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class _FakeImageProcessor:
    def __call__(self, images=None, return_tensors="pt", **kwargs):
        assert isinstance(images, list) and len(images) == 1
        assert isinstance(images[0], Image.Image)
        return {"pixel_values": torch.ones(1, 3, 8, 8)}


def test_internvl_adapter_decodes_image_paths(monkeypatch, tmp_path: Path):
    vlm_adapters = _load_vlm_adapters_module()
    AdapterContext = vlm_adapters.AdapterContext
    InternVLAdapter = vlm_adapters.InternVLAdapter

    img_path = tmp_path / "img.png"
    Image.new("RGB", (16, 16), "red").save(img_path)

    monkeypatch.setattr(
        vlm_adapters, "_internvl_image_processor", lambda _: _FakeImageProcessor()
    )

    adapter = InternVLAdapter()
    ctx = AdapterContext(
        processor=None,
        tokenizer=_FakeTokenizer("OpenGVLab/InternVL3-1B"),
        model=type("M", (), {"num_image_token": 3, "img_context_token_id": None, "parameters": lambda self: iter([torch.zeros(1, dtype=torch.float16)])})(),
    )

    out = adapter.process_multimodal(text="hi", images=[str(img_path)], ctx=ctx)

    assert "input_ids" in out
    assert "pixel_values" in out
    assert "image_flags" in out
    assert tuple(out["image_flags"].shape) == (1, 1)
    assert out["image_flags"].dtype == torch.long
    assert out["pixel_values"].dtype == torch.float16
    assert ctx.model.img_context_token_id == 42
    assert bool((out["input_ids"] == 42).any())
