"""
Tests for vLLM dtype policy in load_model_tokenizer.

Focus file: neuro_manipulation/utils.py
Purpose: When from_vllm=True, ensure Gemma-3 sets dtype='bfloat16' and
Qwen-VL remains dtype='float16' in the vLLM kwargs.
"""

import sys
import types

import pytest


@pytest.fixture
def stub_vllm_and_transformers(monkeypatch):
    captured = {"kwargs": None}

    # Stub vllm.LLM to capture kwargs
    class FakeLLM:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    vllm_mod = types.ModuleType("vllm")
    vllm_mod.LLM = FakeLLM
    sys.modules["vllm"] = vllm_mod

    # Minimal transformers stub for tokenizer to avoid network
    tf = types.ModuleType("transformers")
    # Minimal API surface used by utils
    class Tok:
        def __init__(self):
            self.pad_token_id = 0
    def fake_tok_from_pretrained(*a, **k):
        return Tok()
    tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=fake_tok_from_pretrained)
    tf.AutoModel = object
    tf.AutoModelForCausalLM = object
    tf.MistralForCausalLM = object
    tf.AutoConfig = types.SimpleNamespace(from_pretrained=lambda *a, **k: types.SimpleNamespace(architectures=["ForCausalLM"]))
    tf.pipeline = lambda *a, **k: None
    sys.modules["transformers"] = tf

    return captured


def test_vllm_uses_bfloat16_for_gemma3(stub_vllm_and_transformers):
    captured = stub_vllm_and_transformers

    # Re-import utils after stubbing vllm/transformers
    sys.modules.pop("neuro_manipulation.utils", None)
    from neuro_manipulation.utils import load_model_tokenizer

    load_model_tokenizer("google/gemma-3-4b-it", from_vllm=True)
    assert captured["kwargs"]["dtype"] == "bfloat16"


def test_vllm_keeps_float16_for_qwen_vl(stub_vllm_and_transformers):
    captured = stub_vllm_and_transformers

    sys.modules.pop("neuro_manipulation.utils", None)
    from neuro_manipulation.utils import load_model_tokenizer

    load_model_tokenizer("Qwen/Qwen2.5-VL-3B-Instruct", from_vllm=True)
    assert captured["kwargs"]["dtype"] == "float16"
