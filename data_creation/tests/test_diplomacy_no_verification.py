################################################################################
# File: data_creation/tests/test_diplomacy_no_verification.py
# Purpose: TDD for data_creation/scenario_creation/langgraph_creation/
#          diplomacy_scenario_creation_graph.py behavior when verification_nodes=[]
################################################################################

from __future__ import annotations

from typing import Any, Dict


def _make_minimal_state(scenario_draft: Dict[str, Any]) -> Dict[str, Any]:
    # Minimal ScenarioCreationState-like dict for finalize_scenario
    return {
        "game_name": "Escalation_Game",
        "participants": ["RUSSIA", "TURKEY"],
        "raw_record": {},
        "map_summary": "",
        "scenario_draft": scenario_draft,
        "gradient_options": None,
        "narrative_feedback": [],
        "preference_feedback": [],
        "payoff_feedback": [],
        "iteration_count": 1,
        "final_scenario": None,
        # When verification_nodes = [], aggregate_verification_dynamic is never run
        # today, so all_converged remains None in the last state.
        "narrative_converged": False,
        "preference_converged": True,
        "payoff_converged": False,
        "all_converged": None,
        "auto_save_path": None,
    }


def test_finalize_scenario_treats_draft_as_final_when_all_converged_is_none():
    """
    When running Diplomacy graph with verification_nodes = [],
    the first scenario_draft should be treated as the final scenario.

    This test encodes the desired behavior directly on finalize_scenario:
    if all_converged is None but a scenario_draft exists, the function
    should return a state whose final_scenario equals scenario_draft.
    """
    from data_creation.scenario_creation.langgraph_creation import (  # type: ignore
        diplomacy_scenario_creation_graph as mod,
    )

    draft = {
        "scenario": "Dummy_Diplomacy_Scenario",
        "participants": [{"name": "RUSSIA"}, {"name": "TURKEY"}],
    }
    state = _make_minimal_state(draft)

    new_state = mod.finalize_scenario(state)
    final = new_state["final_scenario"]

    # Final scenario should be based on the draft
    assert isinstance(final, dict)
    assert final["scenario"] == draft["scenario"]
    assert final["participants"] == draft["participants"]
    # Ensure we didn't drop the scenario_draft itself
    assert new_state["scenario_draft"] == draft


def test_finalize_scenario_exposes_gradient_options_field():
    """
    Even when gradient_options is None (no gradient stage configured),
    finalize_scenario should expose a 'gradient_options' field on the
    final_scenario so downstream code can rely on its presence.
    """
    from data_creation.scenario_creation.langgraph_creation import (  # type: ignore
        diplomacy_scenario_creation_graph as mod,
    )

    draft = {
        "scenario": "Dummy_Diplomacy_Scenario",
        "participants": [{"name": "RUSSIA"}, {"name": "TURKEY"}],
    }
    state = _make_minimal_state(draft)

    new_state = mod.finalize_scenario(state)
    final = new_state["final_scenario"]

    assert isinstance(final, dict)
    assert "gradient_options" in final
    assert isinstance(final["gradient_options"], list)


# Ensure the Diplomacy graph still routes through gradient generation even when all verification nodes are disabled.
def test_graph_routes_to_gradient_generation_without_verifiers():
    from data_creation.scenario_creation.langgraph_creation import (  # type: ignore
        diplomacy_scenario_creation_graph as mod,
    )

    graph = mod.build_scenario_creation_graph(debug_mode=True, verification_nodes=[])
    edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}
    assert ("aggregate_verification", "propose_gradient_options") in edges
    assert ("propose_gradient_options", "finalize_scenario") in edges


def test_finalize_scenario_injects_whose_option_label():
    """Finalize should copy the first participant label into the root whose_option."""
    from data_creation.scenario_creation.langgraph_creation import (  # type: ignore
        diplomacy_scenario_creation_graph as mod,
    )

    draft = {
        "scenario": "Dummy_Diplomacy_Scenario",
        "participants": [{"name": "RUSSIA"}, {"name": "TURKEY"}],
    }
    state = _make_minimal_state(draft)
    state["participants"] = ["RUSSIA", "TURKEY"]

    final = mod.finalize_scenario(state)["final_scenario"]
    assert final["whose_option"] == "RUSSIA"


def test_save_files_falls_back_to_draft(tmp_path):
    """If final_scenario is missing, save_files should persist the scenario_draft."""
    import asyncio
    import json
    import types
    from data_creation import create_scenario_langgraph as runner

    # Minimal fake history with a scenario_draft but no final_scenario
    draft = {
        "scenario": "Fallback_Scenario",
        "participants": [{"name": "FRANCE"}, {"name": "RUSSIA"}],
    }
    latest_values = {
        "scenario_draft": draft,
        "participants": ["FRANCE", "RUSSIA"],
        "gradient_options": [{"id": 1, "text": "Hold"}],
    }
    fake_state = types.SimpleNamespace(
        values=latest_values, metadata={}, created_at="now"
    )
    history = [fake_state]

    scenario_dir = tmp_path / "scenarios"
    history_dir = tmp_path / "histories"
    scenario_dir.mkdir()
    history_dir.mkdir()

    async def fake_history():
        for item in history:
            yield item

    ok = asyncio.run(
        runner.save_scenario_and_history(
            scenario=None,
            scenario_graph=types.SimpleNamespace(aget_state_history=lambda config: fake_history()),
            config={},
            persona_job_filename="rec_99",
            scenario_path_base=str(scenario_dir),
            history_path_base=str(history_dir),
        )
    )

    assert ok is True
    saved = json.loads((scenario_dir / "rec_99.json").read_text())
    assert saved["scenario"] == "Fallback_Scenario"
    assert saved["whose_option"] == "FRANCE"
    assert "gradient_options" in saved
