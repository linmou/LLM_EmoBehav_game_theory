# Responsible file: `tests/unit/neuro_manipulation/repe/test_phi_vlm_adapter.py`
# Purpose: Phi vision processors require explicit image placeholder tokens in text; adapter must inject them and bypass chat templates.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_vlm_adapters_module():
    repo_root = Path(__file__).resolve().parents[4]
    vlm_adapters_path = repo_root / "neuro_manipulation" / "repe" / "vlm_adapters.py"
    spec = importlib.util.spec_from_file_location("vlm_adapters_module", vlm_adapters_path)
    assert spec and spec.loader
    vlm_adapters = importlib.util.module_from_spec(spec)
    sys.modules["vlm_adapters_module"] = vlm_adapters
    spec.loader.exec_module(vlm_adapters)
    return vlm_adapters


def test_phi_vision_adapter_bypasses_chat_template() -> None:
    vlm_adapters = _load_vlm_adapters_module()

    AdapterContext = vlm_adapters.AdapterContext
    AdapterRegistry = vlm_adapters.AdapterRegistry

    class _Proc:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, *args, **kwargs):
            raise AssertionError("Should not call apply_chat_template for Phi vision processors")

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return {"ok": True}

    processor = _Proc()
    ctx = AdapterContext(processor=processor, tokenizer=SimpleNamespace(name_or_path="microsoft/Phi-3.5-vision-instruct"))

    adapter = AdapterRegistry().get("microsoft/Phi-3.5-vision-instruct")
    assert adapter is not None

    out = adapter.process_multimodal(text="hello", images=["img"], ctx=ctx)
    assert out == {"ok": True}
    assert processor.calls
    assert "<|image_1|>" in processor.calls[0]["text"]
    assert "hello" in processor.calls[0]["text"]
    assert processor.calls[0]["images"] == ["img"]
