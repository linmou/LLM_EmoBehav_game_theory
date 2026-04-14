"""Tests for vLLM loader behavior.

Responsible files:
- neuro_manipulation/utils.py (load_model_only)

Purpose:
- When `from_vllm=True`, do not swallow vLLM errors and do not fall back to HF.
"""

from __future__ import annotations

import pytest


def test_load_model_only_from_vllm_raises_and_does_not_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """I am starting with a failing test. This is the Red phase."""
    from neuro_manipulation import utils as nm_utils

    def _boom(**_kwargs):
        raise RuntimeError("vllm init failed")

    # Force vLLM init to fail by patching the symbol used by the function.
    monkeypatch.setitem(nm_utils.load_model_only.__globals__, "LLM", _boom)

    # Ensure HF fallback isn't attempted.
    def _should_not_be_called(*_args, **_kwargs):
        raise AssertionError("HF fallback must not be used when from_vllm=True")

    monkeypatch.setitem(
        nm_utils.load_model_only.__globals__,
        "AutoModel",
        type("X", (), {"from_pretrained": staticmethod(_should_not_be_called)}),
    )

    with pytest.raises(RuntimeError, match="vLLM loading failed"):
        nm_utils.load_model_only("dummy-model", from_vllm=True, loading_config=None)


def test_load_model_only_from_vllm_allows_attention_backend_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """I am starting with a failing test. This is the Red phase."""
    import os

    from neuro_manipulation import utils as nm_utils

    seen = {}

    def _fake_llm(**kwargs):
        seen["kwargs"] = kwargs
        return object()

    monkeypatch.setitem(nm_utils.load_model_only.__globals__, "LLM", _fake_llm)
    monkeypatch.delenv("VLLM_ATTENTION_BACKEND", raising=False)

    class _Cfg:
        def to_vllm_kwargs(self):
            return {
                "model": "dummy-model",
                "tensor_parallel_size": 1,
                "max_model_len": 8,
                "trust_remote_code": True,
                "enforce_eager": True,
                "gpu_memory_utilization": 0.9,
                "dtype": "float16",
                "seed": 1,
                "disable_custom_all_reduce": False,
                "attention_backend": "TRITON_ATTN",
                "mm_encoder_attn_backend": "TORCH_SDPA",
            }

    nm_utils.load_model_only("dummy-model", from_vllm=True, loading_config=_Cfg())
    assert os.environ.get("VLLM_ATTENTION_BACKEND") == "TRITON_ATTN"
    assert "attention_backend" not in seen["kwargs"]
    assert seen["kwargs"]["mm_encoder_attn_backend"] == "TORCH_SDPA"


def test_load_model_only_rejects_torch_sdpa_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """I am starting with a failing test. This is the Red phase."""
    from neuro_manipulation import utils as nm_utils

    def _fake_llm(**_kwargs):
        return object()

    monkeypatch.setitem(nm_utils.load_model_only.__globals__, "LLM", _fake_llm)

    class _Cfg:
        def to_vllm_kwargs(self):
            return {
                "model": "dummy-model",
                "tensor_parallel_size": 1,
                "max_model_len": 8,
                "trust_remote_code": True,
                "enforce_eager": True,
                "gpu_memory_utilization": 0.9,
                "dtype": "float16",
                "seed": 1,
                "disable_custom_all_reduce": False,
                "attention_backend": "TORCH_SDPA",
            }

    with pytest.raises(ValueError, match="TORCH_SDPA"):
        nm_utils.load_model_only("dummy-model", from_vllm=True, loading_config=_Cfg())
