"""
Test file responsible for: emotion_experiment_engine/memory_prompt_wrapper.py
Purpose: verify EmotionCheckPromptWrapper supports runtime system prompt override for
auto experiments without code edits between iterations.
"""

import os
from unittest.mock import patch

from emotion_experiment_engine.memory_prompt_wrapper import EmotionCheckPromptWrapper


class DummyPromptFormat:
    def __init__(self) -> None:
        self.last_system_prompt = None
        self.last_user_messages = None

    def build(
        self,
        system_prompt,
        user_messages,
        assistant_messages=None,
        enable_thinking=False,
    ):
        self.last_system_prompt = system_prompt
        self.last_user_messages = user_messages
        return "ok"


def test_emotion_check_uses_env_system_prompt_override():
    prompt_format = DummyPromptFormat()
    wrapper = EmotionCheckPromptWrapper(
        prompt_format=prompt_format,
        task_type="psyset_emotion_eval",
    )
    override = (
        "You are a respondent writing concise first-person statements. "
        "Be concrete, specific, and natural."
    )

    with patch.dict(os.environ, {"EMOTION_CHECK_SYSTEM_PROMPT_OVERRIDE": override}):
        wrapper(
            context=None,
            question="Describe this scene in one short sentence.",
            user_messages="Please provide your answer.",
            emotion="anger",
            options=None,
        )

    assert prompt_format.last_system_prompt == override

