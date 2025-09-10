"""
Tests for load_model_tokenizer dtype decisions.

Focus file: neuro_manipulation/utils.py
Purpose: Ensure Gemma-3 loads in bfloat16 to avoid NaNs in hidden states,
while other models (e.g., Qwen-VL) may continue using float16.
"""

from types import SimpleNamespace
import torch
import builtins
import types

import pytest


@pytest.fixture
def patch_transformers(monkeypatch):
    """Patch AutoConfig/AutoModel(ForCausalLM) to capture torch_dtype/device_map without network calls."""
    captured = {}

    class DummyModel:
        def __init__(self):
            self.config = SimpleNamespace(architectures=["ForCausalLM"])

        def eval(self):
            return self

    def fake_auto_config_from_pretrained(name, *args, **kwargs):
        # Return a config that triggers AutoModelForCausalLM path
        return SimpleNamespace(architectures=["ForCausalLM"])  # minimal

    def fake_from_pretrained(model_name_or_path, torch_dtype=None, device_map=None, **kwargs):
        captured["torch_dtype"] = torch_dtype
        captured["device_map"] = device_map
        return DummyModel()

    # Patch
    import transformers as tf
    monkeypatch.setattr(tf, "AutoConfig", SimpleNamespace(from_pretrained=fake_auto_config_from_pretrained))
    monkeypatch.setattr(tf, "AutoModelForCausalLM", SimpleNamespace(from_pretrained=fake_from_pretrained))
    monkeypatch.setattr(tf, "AutoModel", SimpleNamespace(from_pretrained=fake_from_pretrained))
    monkeypatch.setattr(tf, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(pad_token_id=0)))

    return captured


def test_gemma3_uses_bfloat16(patch_transformers, monkeypatch):
    from neuro_manipulation.utils import load_model_tokenizer

    captured = patch_transformers

    model, tokenizer, processor = load_model_tokenizer("google/gemma-3-4b-it", auto_load_multimodal=False)

    assert captured["torch_dtype"] == torch.bfloat16


def test_qwen_vl_remains_float16(patch_transformers, monkeypatch):
    from neuro_manipulation.utils import load_model_tokenizer

    captured = patch_transformers

    model, tokenizer, processor = load_model_tokenizer("Qwen/Qwen2.5-VL-3B-Instruct", auto_load_multimodal=False)

    assert captured["torch_dtype"] == torch.float16

