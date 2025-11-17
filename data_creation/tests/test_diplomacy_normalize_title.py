# File: tests/test_diplomacy_normalize_title.py
# Purpose: TDD - ensure scenario title is grounded to map (dest/phase) and powers.

import runpy


def _load_helpers():
    mod = runpy.run_path(
        "data_creation/scenario_creation/langgraph_creation/diplomacy_scenario_creation_graph.py"
    )
    return mod["normalize_scenario_title"], mod["prepare_diplomacy_from_raw"]


def test_title_normalization_uses_dest_and_powers_and_phase():
    normalize_title, prepare = _load_helpers()
    raw = {"phase": "F1902M", "destination": "SWE", "involved_powers": ["ENGLAND", "RUSSIA"]}
    state = {"game_name": "Escalation_Game", "participants": [], "raw_record": raw}
    out = prepare(state)
    players = out["participants"]
    draft = {"scenario": "River_Deltas_Fishing_Dispute"}
    new_draft = normalize_title(draft, players, raw)
    title = new_draft["scenario"]
    assert "ENGLAND" in title and "RUSSIA" in title
    assert "SWE" in title
    assert "1902" in title and "Fall" in title

