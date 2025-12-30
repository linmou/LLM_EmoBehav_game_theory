"""
Tests for `neuro_manipulation/utils.py::load_tokenizer_only`.

Purpose: Ensure tokenizer loading prefers fast tokenizers to avoid failures for
models that ship `tokenizer.json` but not a SentencePiece `tokenizer.model`.
"""

import unittest
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch


class TestTokenizerLoading(unittest.TestCase):
    """Unit tests for tokenizer-only loading utilities."""

    def test_load_tokenizer_only_prefers_fast_tokenizer(self):
        """`load_tokenizer_only` should prefer fast tokenizers (use_fast=True)."""
        from neuro_manipulation.utils import load_tokenizer_only

        fake_tokenizer = MagicMock()
        fake_tokenizer.pad_token_id = None

        transformers_stub = ModuleType("transformers")
        auto_tokenizer_stub = MagicMock()
        auto_tokenizer_stub.from_pretrained.return_value = fake_tokenizer
        transformers_stub.AutoTokenizer = auto_tokenizer_stub

        with patch.dict("sys.modules", {"transformers": transformers_stub}):
            load_tokenizer_only(
                model_name_or_path="dummy-model",
                expand_vocab=False,
                auto_load_multimodal=False,
            )

        _, kwargs = auto_tokenizer_stub.from_pretrained.call_args
        self.assertIs(kwargs.get("use_fast"), True)

    def test_load_tokenizer_only_falls_back_to_slow(self):
        """If fast tokenizer loading fails, retry with `use_fast=False`."""
        from neuro_manipulation.utils import load_tokenizer_only

        fake_tokenizer = MagicMock()
        fake_tokenizer.pad_token_id = None

        transformers_stub = ModuleType("transformers")
        auto_tokenizer_stub = MagicMock()
        auto_tokenizer_stub.from_pretrained.side_effect = [
            Exception("fast tokenizer not available"),
            fake_tokenizer,
        ]
        transformers_stub.AutoTokenizer = auto_tokenizer_stub

        with patch.dict("sys.modules", {"transformers": transformers_stub}):
            load_tokenizer_only(
                model_name_or_path="dummy-model",
                expand_vocab=False,
                auto_load_multimodal=False,
            )

        self.assertEqual(auto_tokenizer_stub.from_pretrained.call_count, 2)
        first_kwargs = auto_tokenizer_stub.from_pretrained.call_args_list[0].kwargs
        second_kwargs = auto_tokenizer_stub.from_pretrained.call_args_list[1].kwargs
        self.assertIs(first_kwargs.get("use_fast"), True)
        self.assertIs(second_kwargs.get("use_fast"), False)

    def test_load_tokenizer_only_uses_sibling_tokenizer_if_missing_files(self):
        """
        If a local model dir has no tokenizer artifacts (e.g. mamba2 weights-only),
        `load_tokenizer_only` should fall back to a sibling Mamba tokenizer directory.
        """
        from neuro_manipulation.utils import load_tokenizer_only

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            mamba2_dir = base / "mamba2-780m"
            mamba_dir = base / "mamba-790m-hf"
            mamba2_dir.mkdir(parents=True)
            mamba_dir.mkdir(parents=True)

            # Minimal configs: mamba2 has weights-only, sibling has tokenizer.json.
            (mamba2_dir / "config.json").write_text('{"ssm_cfg":{"layer":"Mamba2"}}')
            (mamba_dir / "tokenizer.json").write_text("{}")

            fake_tokenizer = MagicMock()
            fake_tokenizer.pad_token_id = None

            transformers_stub = ModuleType("transformers")
            auto_tokenizer_stub = MagicMock()
            auto_tokenizer_stub.from_pretrained.return_value = fake_tokenizer
            transformers_stub.AutoTokenizer = auto_tokenizer_stub

            with patch.dict("sys.modules", {"transformers": transformers_stub}):
                tokenizer, _ = load_tokenizer_only(
                    model_name_or_path=str(mamba2_dir),
                    expand_vocab=False,
                    auto_load_multimodal=False,
                )

            self.assertIs(tokenizer, fake_tokenizer)
            self.assertGreaterEqual(auto_tokenizer_stub.from_pretrained.call_count, 1)
            for call in auto_tokenizer_stub.from_pretrained.call_args_list:
                self.assertEqual(Path(call.args[0]), mamba_dir)
