"""
Responsible: delta_activation_engine/prompts/wrappers.py
Purpose: Prompt wrapper for probe-style inputs (chat-template aware).
"""

from __future__ import annotations

from typing import Optional


class DeltaProbesPromptWrapper:
    def __init__(
        self,
        prompt_format,
        *,
        user_messages: str = "Please provide your answer.",
        enable_thinking: bool = False,
        system_prompt: str = "",
    ) -> None:
        self.prompt_format = prompt_format
        self.default_user = user_messages
        self.enable_thinking = bool(enable_thinking)
        self.system_prompt = system_prompt

    def __call__(
        self,
        *,
        context: Optional[str] = None,
        question: str,
        answer: Optional[str] = None,
        options: Optional[object] = None,
    ) -> str:
        if context:
            user_text = f"{context}\n{question}" if question else context
        else:
            user_text = question

        return self.prompt_format.build(
            self.system_prompt,
            [user_text],
            [],
            images=None,
            enable_thinking=self.enable_thinking,
        )

