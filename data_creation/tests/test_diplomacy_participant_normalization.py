# File: tests/test_diplomacy_participant_normalization.py
# Purpose: TDD - ensure we normalize scenario_draft.participants to involved powers.

import runpy


def _load_normalizer():
    mod = runpy.run_path(
        "data_creation/scenario_creation/langgraph_creation/diplomacy_scenario_creation_graph.py"
    )
    return mod["normalize_scenario_participants"]


def test_normalize_overwrites_wrong_names_dict_form():
    normalize = _load_normalizer()
    draft = {
        "scenario": "Neighborhood_Park_Picnic_Space",
        "participants": [
            {"name": "Alice", "role": "Resident Planning a Picnic"},
            {"name": "Ben", "role": "Resident Planning a Picnic"},
        ],
    }
    out = normalize(draft, ["RUSSIA", "TURKEY"])
    assert isinstance(out["participants"], list)
    assert out["participants"][0]["name"] == "RUSSIA"
    assert out["participants"][1]["name"] == "TURKEY"


def test_normalize_sets_when_missing():
    normalize = _load_normalizer()
    draft = {"scenario": "X"}
    out = normalize(draft, ["ENGLAND", "FRANCE"])
    assert out["participants"][0]["name"] == "ENGLAND"
    assert out["participants"][1]["name"] == "FRANCE"

