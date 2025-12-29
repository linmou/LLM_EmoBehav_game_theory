"""Tests for vLLM attention backend env override.

Responsible files:
- emotion_experiment_engine/emotion_experiment_series_runner.py

Purpose:
- Ensure we set `VLLM_ATTENTION_BACKEND` before importing vLLM (timing matters).
"""

from __future__ import annotations

import os

import pytest

from emotion_experiment_engine.data_models import VLLMLoadingConfig


def test_apply_vllm_env_overrides_sets_attention_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """I am starting with a failing test. This is the Red phase."""
    from emotion_experiment_engine.emotion_experiment_series_runner import (
        _apply_vllm_env_overrides,
    )

    monkeypatch.delenv("VLLM_ATTENTION_BACKEND", raising=False)
    monkeypatch.delenv("VLLM_DISABLE_FLASH_ATTN", raising=False)

    cfg = VLLMLoadingConfig(
        model_path="dummy",
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        max_model_len=8,
        enforce_eager=True,
        quantization=None,
        trust_remote_code=True,
        dtype="float16",
        seed=1,
        disable_custom_all_reduce=False,
        additional_vllm_kwargs={"attention_backend": "TRITON_ATTN"},
    )

    _apply_vllm_env_overrides(cfg)
    assert os.environ.get("VLLM_ATTENTION_BACKEND") == "TRITON_ATTN"
    assert os.environ.get("VLLM_DISABLE_FLASH_ATTN") is None


def test_apply_vllm_env_overrides_rejects_torch_sdpa() -> None:
    """I am starting with a failing test. This is the Red phase."""
    from emotion_experiment_engine.emotion_experiment_series_runner import (
        _apply_vllm_env_overrides,
    )

    cfg = VLLMLoadingConfig(
        model_path="dummy",
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        max_model_len=8,
        enforce_eager=True,
        quantization=None,
        trust_remote_code=True,
        dtype="float16",
        seed=1,
        disable_custom_all_reduce=False,
        additional_vllm_kwargs={"attention_backend": "TORCH_SDPA"},
    )

    with pytest.raises(ValueError, match="TORCH_SDPA"):
        _apply_vllm_env_overrides(cfg)


def test_apply_vllm_env_overrides_sets_disable_flash_attn_and_pythonpath(monkeypatch: pytest.MonkeyPatch) -> None:
    """I am starting with a failing test. This is the Red phase."""
    from emotion_experiment_engine.emotion_experiment_series_runner import (
        _apply_vllm_env_overrides,
    )

    monkeypatch.delenv("VLLM_DISABLE_FLASH_ATTN", raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)

    cfg = VLLMLoadingConfig(
        model_path="dummy",
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        max_model_len=8,
        enforce_eager=True,
        quantization=None,
        trust_remote_code=True,
        dtype="float16",
        seed=1,
        disable_custom_all_reduce=False,
        additional_vllm_kwargs={
            "attention_backend": "TRITON_ATTN",
            "mm_encoder_attn_backend": "TORCH_SDPA",
        },
    )

    _apply_vllm_env_overrides(cfg)
    assert os.environ.get("VLLM_DISABLE_FLASH_ATTN") == "1"
    assert os.environ.get("PYTHONPATH")
