# Tests for emotion_experiment_engine/game_prompt_wrapper.py
"""Unit tests for the game benchmark prompt wrapper adapter."""

from typing import Any, List

import pytest

from emotion_experiment_engine.game_prompt_wrapper import GameBenchmarkPromptWrapper


class _DummyPromptFormat:
    """Minimal prompt format stub."""

    def __init__(self) -> None:
        self.records: List[Any] = []

    def build(self, system_prompt: str, user_messages: List[str], enable_thinking: bool = False):
        self.records.append((system_prompt, user_messages, enable_thinking))
        return f"PROMPT::{enable_thinking}"


@pytest.fixture(autouse=True)
def patch_game_config(monkeypatch):
    class _FakeDecision:
        @staticmethod
        def example() -> str:
            return "{\"decision\": \"Option 1\"}"

    monkeypatch.setattr(
        "emotion_experiment_engine.game_prompt_wrapper.get_game_config",
        lambda task_type: {"decision_class": _FakeDecision},
    )


@pytest.fixture()
def prompt_wrapper(monkeypatch) -> GameBenchmarkPromptWrapper:
    captured = {}

    class _FakeGameReactWrapper:
        def __init__(self, prompt_format, response_format):
            captured["init"] = (prompt_format, response_format)

        def __call__(self, event, options, user_messages, enable_thinking=False):
            captured["call"] = {
                "event": event,
                "options": options,
                "user_messages": user_messages,
                "enable_thinking": enable_thinking,
            }
            return "react-output"

    monkeypatch.setattr(
        "emotion_experiment_engine.game_prompt_wrapper.GameReactPromptWrapper",
        _FakeGameReactWrapper,
    )

    wrapper = GameBenchmarkPromptWrapper(_DummyPromptFormat(), "Prisoners_Dilemma")
    wrapper._captured = captured  # type: ignore[attr-defined]
    return wrapper


def test_wrapper_builds_prompt(prompt_wrapper: GameBenchmarkPromptWrapper):
    prompt = prompt_wrapper(
        context="irrelevant",
        question="Describe the payoff matrix",
        user_messages=["Choose wisely."],
        enable_thinking=True,
        augmentation_config=None,
        answer=None,
        emotion="anger",
        options=[
            {"id": 1, "text": "Cooperate"},
            {"id": 2, "text": "Defect"},
        ],
    )

    assert prompt == "react-output"

    captured = prompt_wrapper._captured  # type: ignore[attr-defined]
    assert captured["call"]["event"] == "Describe the payoff matrix"

    option_texts = captured["call"]["options"]
    assert option_texts == ["Cooperate", "Defect"]
    assert captured["call"]["user_messages"] == ["Choose wisely."]
    assert captured["call"]["enable_thinking"] is True
