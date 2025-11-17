# File: tests/test_diplomacy_prepare_participants.py
# Purpose: TDD - ensure Diplomacy graph sets participants from involved_powers
#
# This test calls prepare_diplomacy_from_raw directly (no API calls).

import runpy


def _load_prepare():
    mod = runpy.run_path(
        "data_creation/scenario_creation/langgraph_creation/diplomacy_scenario_creation_graph.py"
    )
    return mod["prepare_diplomacy_from_raw"]


def test_participants_set_from_involved_powers_plural():
    prepare = _load_prepare()
    state = {
        "game_name": "Escalation_Game",
        "participants": [],
        "raw_record": {
            "involved_powers": ["ENGLAND", "FRANCE"],
            "phase": "S1901M",
            "destination": "ENG",
            "orders_to_dest": [],
            "units_near_dest": {},
        },
    }
    out = prepare(state)
    assert out["participants"] == ["ENGLAND", "FRANCE"]
    assert isinstance(out.get("map_summary"), str) and "Spring 1901" in out["map_summary"]
    assert "Contesting ENG" in out["map_summary"]


def test_participants_set_from_involved_power_singular_key():
    prepare = _load_prepare()
    state = {
        "game_name": "Escalation_Game",
        "participants": [],
        "raw_record": {
            "involved_power": ["RUSSIA", "TURKEY"],
            "phase": "S1901M",
            "destination": "BLA",
        },
    }
    out = prepare(state)
    assert out["participants"] == ["RUSSIA", "TURKEY"]
    assert "Contesting BLA" in out.get("map_summary", "")

