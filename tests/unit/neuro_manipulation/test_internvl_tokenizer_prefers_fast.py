# Responsible file: `tests/unit/neuro_manipulation/test_internvl_tokenizer_prefers_fast.py`
# Purpose: InternVL tokenizers should be loaded with `use_fast=True` (Qwen2TokenizerFast) to avoid missing merges_file issues.

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch


def test_load_tokenizer_only_uses_fast_for_internvl_models() -> None:
    import neuro_manipulation.utils as nm_utils

    dummy_tok = SimpleNamespace(pad_token_id=0, name_or_path="OpenGVLab/InternVL3-8B")

    with patch.object(nm_utils.utils_module, "AutoTokenizer") as auto_tok:
        auto_tok.from_pretrained.return_value = dummy_tok
        _tok, _proc = nm_utils.load_tokenizer_only(
            "OpenGVLab/InternVL3-8B", auto_load_multimodal=False
        )

        _args, kwargs = auto_tok.from_pretrained.call_args
        assert kwargs.get("use_fast") is True

