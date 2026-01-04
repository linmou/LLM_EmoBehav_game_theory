################################################################################
# File: data_creation/tests/test_diplomacy_behavior_choices_verification.py
# Purpose: TDD for behavior verification:
#   - Prompt-level criteria for emotion-neutral, objective behaviors
#   - LLM wiring via verify_behavior_choices helper
#   - Diplomacy-specific key matching against GameClass.behavior_choices.example()
################################################################################

from __future__ import annotations

import importlib
import runpy
from typing import Any, Dict, List

import pytest


def _load_verifier():
    return runpy.run_path(
        "data_creation/scenario_creation/langgraph_creation/behavior_choices_verifier.py"
    )


def _minimal_state(behavior_choices: Dict[str, str]) -> Dict[str, object]:
    return {
        "game_name": "Escalation_Game",
        "participants": ["ENGLAND", "FRANCE"],
        "scenario_draft": {
            "description": "Fleet movements near a contested coast.",
            "behavior_choices": behavior_choices,
        },
    }


def test_verify_behavior_choices_flags_emotion_words():
    """
    The prompt should explicitly discourage emotional wording.
    """
    mod = _load_verifier()
    build_prompt = mod["build_behavior_verification_prompt"]
    state = _minimal_state(
        {
            "withdraw": "pull back to hold the line",
            "escalate": "advance aggressively into the sea lane",
        }
    )

    system_prompt, human_prompt = build_prompt(state)
    combined = (system_prompt + human_prompt).lower()

    assert "emotional adjectives or adverbs" in combined
    assert "aggressive" in combined
    assert "happily" in combined


def test_verify_behavior_choices_accepts_neutral_phrasing():
    """
    Neutral, concrete behaviors should be representable in the prompt payload
    without being rewritten.
    """
    mod = _load_verifier()
    build_prompt = mod["build_behavior_verification_prompt"]
    state = _minimal_state(
        {
            "withdraw": "hold position near the coast",
            "escalate": "move fleet toward the channel",
        }
    )

    system_prompt, human_prompt = build_prompt(state)
    payload = pytest.approx  # type: ignore[attr-defined]
    # Basic sanity: the behaviors appear in the JSON payload unchanged.
    assert "hold position near the coast" in human_prompt
    assert "move fleet toward the channel" in human_prompt


def test_verify_behavior_choices_small_lexicon_catches_multiple_hits():
    """
    The prompt should explicitly treat coordination verbs like 'coordinate'
    and 'cooperate' as non-objective wording.
    """
    mod = _load_verifier()
    build_prompt = mod["build_behavior_verification_prompt"]
    state = _minimal_state(
        {
            "option1": "cheerfully coordinate with allies",
            "option2": "calmly patrol neutral waters",
            "option3": "angrily blockade the strait",
        }
    )

    system_prompt, human_prompt = build_prompt(state)
    combined = (system_prompt + human_prompt).lower()
    assert "coordinate" in combined
    assert "cooperate" in combined


def test_verify_behavior_choices_uses_llm_and_parses_response():
    """
    verify_behavior_choices should call the provided LLM and parse its JSON result.
    """
    mod = _load_verifier()
    verify_behavior_choices = mod["verify_behavior_choices"]

    calls: List[List[Dict[str, Any]]] = []

    class FakeLLM:
        def invoke(self, messages, response_format=None):  # type: ignore[override]
            calls.append(messages)

            class Resp:
                content = '{"feedback": ["option1 is too emotional"], "converged": false}'

            return Resp()

    state = _minimal_state(
        {
            "option1": "cheerfully coordinate with allies",
            "option2": "move fleet toward the channel",
        }
    )

    result = verify_behavior_choices(state, FakeLLM())

    assert result["behavior_converged"] is False
    assert result["behavior_feedback"] == ["option1 is too emotional"]
    # Ensure we actually passed both system and user messages to the LLM.
    assert len(calls) == 1
    assert len(calls[0]) == 2


def test_diplomacy_verify_behavior_enforces_template_keys():
    """
    Diplomacy verify_behavior should enforce that behavior_choices keys
    match GameClass.behavior_choices.example() (e.g., {'escalate','withdraw'}
    for Escalation_Game).
    """
    import runpy

    mod = runpy.run_path(
        "data_creation/scenario_creation/langgraph_creation/diplomacy_scenario_creation_graph.py"
    )
    verify_behavior = mod["verify_behavior"]

    # Deliberately use a wrong key set: 'advance' instead of 'escalate'
    state = {
        "game_name": "Escalation_Game",
        "participants": ["ENGLAND", "FRANCE"],
        "scenario_draft": {
            "description": "Fleet movements near a contested coast.",
            "behavior_choices": {
                "advance": "Move fleet into the contested sea.",
                "withdraw": "Hold current fleet position.",
            },
        },
    }

    result = verify_behavior(state)

    assert result["behavior_converged"] is False
    feedback = " ".join(result["behavior_feedback"]).lower()
    # Should mention expected template keys and the unexpected one.
    assert "escalate" in feedback
    assert "withdraw" in feedback
    assert "advance" in feedback


def test_graph_includes_behavior_node_by_default():
    """
    The diplomacy graph should wire the behavior verifier into the verification stage.
    """
    pytest.importorskip("azure")
    pytest.importorskip("langgraph")
    mod = importlib.import_module(
        "data_creation.scenario_creation.langgraph_creation.diplomacy_scenario_creation_graph"
    )
    graph = mod.build_scenario_creation_graph(debug_mode=True)
    edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}
    assert ("propose_scenario", "verify_behavior") in edges
    assert ("verify_behavior", "aggregate_verification") in edges
