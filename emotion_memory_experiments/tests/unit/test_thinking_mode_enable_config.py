"""
Test: Thinking mode enable_thinking integration and prompt injection
Responsible files:
- neuro_manipulation/prompt_formats.py (PromptFormat, Qwen3InstFormat)
- emotion_memory_experiments/memory_prompt_wrapper.py (MemoryPromptWrapper)

Purpose:
- Verify that when `enable_thinking=True`, the Qwen3 prompt format injects '/think' into the first user message
- Verify wrappers pass `enable_thinking` through to the PromptFormat and resulting prompt reflects thinking mode
"""

import unittest
import sys
import types

# Stub external deps so we can import PromptFormat without installed packages
if 'transformers' not in sys.modules:
    transformers_stub = types.ModuleType('transformers')
    class _AutoTokenizerStub:  # minimal placeholder
        pass
    transformers_stub.AutoTokenizer = _AutoTokenizerStub
    sys.modules['transformers'] = transformers_stub

if 'jinja2' not in sys.modules:
    jinja2_stub = types.ModuleType('jinja2')
    exceptions_stub = types.ModuleType('jinja2.exceptions')
    class _TemplateErrorStub(Exception):
        pass
    exceptions_stub.TemplateError = _TemplateErrorStub
    sys.modules['jinja2'] = jinja2_stub
    sys.modules['jinja2.exceptions'] = exceptions_stub

from neuro_manipulation.prompt_formats import PromptFormat


class FakeQwen3Tokenizer:
    """Minimal tokenizer stub exposing only what's needed for PromptFormat.
    We avoid calling apply_chat_template by ensuring Qwen3 + enable_thinking path is taken.
    """

    def __init__(self, name_or_path: str = "Qwen3-1.7B") -> None:
        self.name_or_path = name_or_path


class TestThinkingModeEnableConfig(unittest.TestCase):
    def setUp(self) -> None:
        # Use a fake Qwen3 tokenizer so PromptFormat detects Qwen3 model
        self.tokenizer = FakeQwen3Tokenizer()
        self.prompt_format = PromptFormat(self.tokenizer)  # real PromptFormat logic

    def test_prompt_format_injects_think_for_qwen3_when_enabled(self):
        system = "You are Alice."
        user_msgs = ["State your choice succinctly."]

        prompt = self.prompt_format.build(
            system_prompt=system,
            user_messages=user_msgs,
            assistant_messages=[],
            enable_thinking=True,
        )

        # Should include Qwen3 chat tags and the '/think' directive at the end of the first user content
        self.assertIn("<|im_start|>system", prompt)
        self.assertIn("<|im_start|>user", prompt)
        self.assertIn("<|im_start|>assistant", prompt)
        self.assertIn(user_msgs[0] + "\n/think", prompt)

    def test_prompt_format_does_not_inject_think_when_disabled(self):
        system = "You are Alice."
        user_msgs = ["State your choice succinctly."]

        prompt = self.prompt_format.build(
            system_prompt=system,
            user_messages=user_msgs,
            assistant_messages=[],
            enable_thinking=False,
        )

        self.assertNotIn("/think", prompt)
        # Since we manually inject directives for Qwen3, expect /no_think to appear
        self.assertIn("/no_think", prompt)


if __name__ == "__main__":
    unittest.main()
