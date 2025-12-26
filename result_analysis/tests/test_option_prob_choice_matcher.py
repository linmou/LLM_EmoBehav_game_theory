# Tests for result_analysis/score_game_theory_option_prob_match.py
# Purpose: verify option-string construction and argmax-vs-score matching logic.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import pytest

from result_analysis.score_game_theory_option_prob_match import (
    build_option_strings,
    pick_argmax_option_id,
    score_record_match,
)


@dataclass(frozen=True)
class DummyRecord:
    prompt: str
    chosen_option_id: int
    options: List[Dict[str, object]]


class DummyScorer:
    def __init__(self, mapping: Dict[Tuple[str, str], float]) -> None:
        self._mapping = mapping

    def score_options(self, prompt: str, option_strings: Dict[int, str]) -> Dict[int, float]:
        return {opt_id: self._mapping[(prompt, s)] for opt_id, s in option_strings.items()}


def test_build_option_strings_uses_metadata_id_and_text() -> None:
    options = [
        {"id": 2, "text": "B"},
        {"id": 1, "text": "A"},
    ]
    assert build_option_strings(options) == {2: "Option 2. B", 1: "Option 1. A"}


def test_pick_argmax_option_id_breaks_ties_by_lowest_id() -> None:
    assert pick_argmax_option_id({2: -1.0, 1: -1.0}) == 1


def test_score_record_match_returns_correct_true_when_argmax_matches_score() -> None:
    record = DummyRecord(
        prompt="P",
        chosen_option_id=2,
        options=[{"id": 1, "text": "A"}, {"id": 2, "text": "B"}],
    )
    scorer = DummyScorer({("P", "Option 1. A"): -2.0, ("P", "Option 2. B"): -1.0})
    out = score_record_match(
        prompt=record.prompt,
        chosen_option_id=record.chosen_option_id,
        options=record.options,
        scorer=scorer,
    )
    assert out["predicted_option_id"] == 2
    assert out["is_match"] is True


def test_score_record_match_returns_correct_false_when_argmax_differs() -> None:
    record = DummyRecord(
        prompt="P",
        chosen_option_id=1,
        options=[{"id": 1, "text": "A"}, {"id": 2, "text": "B"}],
    )
    scorer = DummyScorer({("P", "Option 1. A"): -2.0, ("P", "Option 2. B"): -1.0})
    out = score_record_match(
        prompt=record.prompt,
        chosen_option_id=record.chosen_option_id,
        options=record.options,
        scorer=scorer,
    )
    assert out["predicted_option_id"] == 2
    assert out["is_match"] is False

