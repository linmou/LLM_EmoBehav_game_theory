# File: tests/test_diplomacy_name_sanitization.py
# Purpose: TDD - ensure 'Alice'/'Ben' are sanitized to involved powers in scenario text and payoffs.

import runpy


def _load_funcs():
    mod = runpy.run_path(
        "data_creation/scenario_creation/langgraph_creation/diplomacy_scenario_creation_graph.py"
    )
    return mod["sanitize_names"], mod["normalize_scenario_participants"]


def test_sanitize_replaces_alice_ben_in_text_and_payoff():
    sanitize, normalize = _load_funcs()
    draft = {
        "description": "Alice meets Ben at the board.",
        "participants": [{"name": "Alice"}, {"name": "Ben"}],
        "behavior_choices": {"withdraw": "X", "escalate": "Y"},
        "payoff_matrix_description": {
            "player 1: withdraw , player 2: withdraw": [
                "player 1 gets 0: Alice holds position.",
                "player 2 gets 0: Ben holds position.",
            ]
        },
    }
    players = ["RUSSIA", "TURKEY"]
    out = sanitize(normalize(draft, players), players)
    assert "Alice" not in out["description"]
    assert "Ben" not in out["description"]
    v0, v1 = out["payoff_matrix_description"][
        "player 1: withdraw , player 2: withdraw"
    ]
    assert "Alice" not in v0 and "Ben" not in v1
    assert "RUSSIA" in out["description"] and "TURKEY" in out["description"]

