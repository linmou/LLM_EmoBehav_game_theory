# Responsible file: `tests/unit/neuro_manipulation/test_qwen3_vl_tokenizer_prefers_fast.py`
# Purpose: Qwen3-VL tokenizers should be loaded with `use_fast=True` because some local checkpoints do not ship a slow-tokenizer merges file.

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch


def test_load_tokenizer_only_uses_fast_for_qwen3_vl_models() -> None:
    import neuro_manipulation.utils as nm_utils

    dummy_tok = SimpleNamespace(pad_token_id=0, name_or_path="Qwen/Qwen3-VL-8B-Instruct-FP8")

    with patch.object(nm_utils.utils_module, "AutoTokenizer") as auto_tok:
        auto_tok.from_pretrained.return_value = dummy_tok
        _tok, _proc = nm_utils.load_tokenizer_only(
            "Qwen/Qwen3-VL-8B-Instruct-FP8", auto_load_multimodal=False
        )

        _args, kwargs = auto_tok.from_pretrained.call_args
        assert kwargs.get("use_fast") is True
