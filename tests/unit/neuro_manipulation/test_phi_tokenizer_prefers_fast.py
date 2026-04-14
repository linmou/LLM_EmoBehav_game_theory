# Responsible file: `tests/unit/neuro_manipulation/test_phi_tokenizer_prefers_fast.py`
# Purpose: Phi VLM tokenizers should be loaded with `use_fast=True` to avoid missing vocab file issues.

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch


def test_load_tokenizer_only_uses_fast_for_phi_models() -> None:
    import neuro_manipulation.utils as nm_utils

    dummy_tok = SimpleNamespace(pad_token_id=0, name_or_path="microsoft/Phi-3.5-vision-instruct")

    with patch.object(nm_utils.utils_module, "AutoTokenizer") as auto_tok:
        auto_tok.from_pretrained.return_value = dummy_tok
        _tok, _proc = nm_utils.load_tokenizer_only(
            "microsoft/Phi-3.5-vision-instruct", auto_load_multimodal=False
        )

        _args, kwargs = auto_tok.from_pretrained.call_args
        assert kwargs.get("use_fast") is True

