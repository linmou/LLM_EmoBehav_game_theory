"""
File: neuro_manipulation/prompt_formats.py
Purpose: Ensure PromptFormat gracefully handles tokenizers without `chat_template`
         (e.g. state-spaces/mamba-790m-hf) by skipping `apply_chat_template`.
"""

import unittest
from unittest.mock import MagicMock

from neuro_manipulation.prompt_formats import PromptFormat


class _NoChatTemplateTokenizer:
    def __init__(self, name_or_path: str):
        self.name_or_path = name_or_path
        self.chat_template = None
        self.apply_chat_template = MagicMock(
            side_effect=ValueError(
                "Cannot use chat template functions because tokenizer.chat_template is not set "
                "and no template argument was passed!"
            )
        )


class TestPromptFormatNoChatTemplate(unittest.TestCase):
    def test_skips_apply_chat_template_when_chat_template_missing(self):
        tokenizer = _NoChatTemplateTokenizer(
            "/data/home/jjl7137/huggingface_models/state-spaces/mamba-790m-hf"
        )
        fmt = PromptFormat(tokenizer)  # type: ignore[arg-type]

        prompt = fmt.build(
            system_prompt="SYS",
            user_messages=["hello"],
            assistant_messages=[],
        )

        tokenizer.apply_chat_template.assert_not_called()
        self.assertIn("SYS", prompt)
        self.assertIn("hello", prompt)

