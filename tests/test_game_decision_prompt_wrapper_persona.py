# Tests for emotion_experiment_engine/game_prompt_wrapper.py
"""Ensure GameDecisionPromptWrapper selects persona for Diplomacy scenarios."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_game_prompt_wrapper_module():
    path = Path(__file__).resolve().parents[1] / "emotion_experiment_engine" / "game_prompt_wrapper.py"
    spec = importlib.util.spec_from_file_location("ee_game_prompt_wrapper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_game_decision_prompt_wrapper_defaults_to_alice() -> None:
    mod = _load_game_prompt_wrapper_module()
    wrapper = mod.GameDecisionPromptWrapper(prompt_format=None, task_type="Prisoners_Dilemma")
    prompt = wrapper(
        context=None,
        question="Scenario: Prisoners_Dilemma_01",
        user_messages=None,
        enable_thinking=False,
        augmentation_config=None,
        answer=None,
        emotion=None,
        options=["Cooperate", "Defect"],
    )
    assert "You are Alice." in prompt


def test_game_decision_prompt_wrapper_uses_commander_for_diplomacy() -> None:
    mod = _load_game_prompt_wrapper_module()
    wrapper = mod.GameDecisionPromptWrapper(prompt_format=None, task_type="Prisoners_Dilemma")
    prompt = wrapper(
        context=None,
        question="Scenario: Diplomacy_Test_01",
        user_messages=None,
        enable_thinking=False,
        augmentation_config=None,
        answer=None,
        emotion=None,
        options=["Attack", "Hold"],
    )
    assert "You are a commander who needs to make the decision." in prompt

