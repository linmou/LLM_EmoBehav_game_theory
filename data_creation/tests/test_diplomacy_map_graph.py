# File: tests/test_diplomacy_map_graph.py
# Purpose: TDD - validate Diplomacy map-state → scenario generation (no API calls).
#
# This test covers:
# - Generating a scenario dict from a structured Diplomacy map state
# - Presence of balanced tradeoff sentences in the description
# - Options 1..5 rendered as natural-language orders (no "cooperate"/"defect")
# - Header fields your_country/target_country/phase carried through
#
# The implementation lives in:
# data_creation/scenario_creation/langgraph_creation/scenario_creation_graph_diplomacy.py

from __future__ import annotations

import re
from typing import Dict, Any, List

import pytest


def _has_balanced_tradeoff(text: str) -> bool:
    # Lightweight checks for both sides (restraint/pressure) and cost/benefit signals.
    t = text.lower()
    return (
        ("restraint" in t or "quieter stance" in t or "lower pressure" in t)
        and ("pressure" in t or "advance" in t or "contest" in t or "projection" in t)
        and ("cost" in t or "but" in t)
    )


def _extract_options(prompt: str) -> List[str]:
    opts = []
    for line in prompt.splitlines():
        m = re.match(r"\s*Option\s*(\d+)[\.:)]\s*(.+)", line, flags=re.IGNORECASE)
        if m:
            opts.append(m.group(2).strip())
    return opts


def test_generate_scenario_from_map_basic():
    # Import by path to avoid package-level imports that require azure deps.
    import runpy
    mod = runpy.run_path("data_creation/scenario_creation/langgraph_creation/scenario_creation_graph_diplomacy.py")
    generate_scenario_from_map = mod["generate_scenario_from_map"]

    map_state: Dict[str, Any] = {
        "id": "MAP-ENG-FRA-S1902-01",
        "game": "Diplomacy-Standard",
        "your_country": "England",
        "target_country": "France",
        "phase": {"year": 1902, "season": "Spring", "subphase": "Orders"},
        "your_units": [
            {"type": "F", "location": "English Channel"},
            {"type": "F", "location": "North Sea"},
            {"type": "A", "location": "Liverpool"},
        ],
        "target_units": [
            {"type": "F", "location": "Brest"},
        ],
        # Optional hints for surface text; generator should work without them too.
        "focus_sea": "Mid-Atlantic Ocean",
        "coast_focus": "Brest",
    }

    scenario = generate_scenario_from_map(map_state)

    # Header fields preserved
    assert scenario["your_country"] == "England"
    assert scenario["target_country"] == "France"
    assert isinstance(scenario["phase"], dict) and scenario["phase"]["season"] == "Spring"
    assert scenario["game"] == "Diplomacy-Standard"

    # Scenario description contains balanced tradeoff framing
    desc = scenario["scenario"]
    assert isinstance(desc, str) and len(desc) > 40
    assert _has_balanced_tradeoff(desc)

    # Options present (1..5) with natural-language orders, no explicit cooperate/defect words
    options = scenario["options"]
    assert isinstance(options, list) and 3 <= len(options) <= 5
    # Accept 3..5; prefer 5 when sufficient context is present.
    for i, opt in enumerate(options, start=1):
        assert isinstance(opt, dict) and "text" in opt and "id" in opt
        assert opt["id"] == i
        text = str(opt["text"]).lower()
        assert "cooperate" not in text and "defect" not in text
        # Reasonable length, looks like an order set
        assert len(text) > 12 and any(w in text for w in ["move", "hold", "keep", "advance", "shift"])

    # Sanity: a minimal prompt-like rendering would include "Option 1."
    rendered = "\n".join([f"Option {o['id']}. {o['text']}" for o in options])
    ex_opts = _extract_options(rendered)
    assert len(ex_opts) == len(options)

