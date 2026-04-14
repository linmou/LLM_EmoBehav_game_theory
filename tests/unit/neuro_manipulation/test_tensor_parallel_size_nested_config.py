# Responsible file: `tests/unit/neuro_manipulation/test_tensor_parallel_size_nested_config.py`
# Purpose: Newer multimodal configs may store attention-head counts under nested text/llm configs; tensor-parallel sizing must still detect a usable multi-GPU split.

from __future__ import annotations

import torch


def test_get_optimal_tensor_parallel_size_reads_qwen3_vl_text_config(monkeypatch) -> None:
    import neuro_manipulation.utils as nm_utils

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        nm_utils,
        "get_model_config",
        lambda _path: ({"text_config": {"num_attention_heads": 32}}, "ignored"),
    )

    assert nm_utils.get_optimal_tensor_parallel_size("Qwen/Qwen3-VL-8B-Instruct-FP8") == 2


def test_get_optimal_tensor_parallel_size_reads_internvl_llm_config(monkeypatch) -> None:
    import neuro_manipulation.utils as nm_utils

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        nm_utils,
        "get_model_config",
        lambda _path: ({"llm_config": {"num_attention_heads": 28}}, "ignored"),
    )

    assert nm_utils.get_optimal_tensor_parallel_size("OpenGVLab/InternVL3-8B-AWQ") == 2
