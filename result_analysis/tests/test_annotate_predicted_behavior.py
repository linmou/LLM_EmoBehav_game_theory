# Tests for result_analysis/annotate_predicted_behavior.py
# Purpose: ensure option text -> behavior classification and annotation works.

from __future__ import annotations

from result_analysis.annotate_predicted_behavior import _unpack_option_payload, classify_pd_behavior


def test_classify_pd_behavior_defect_keywords() -> None:
    assert classify_pd_behavior("I will defect.") == "defect"
    assert classify_pd_behavior("Betray the other player") == "defect"


def test_classify_pd_behavior_cooperate_keywords() -> None:
    assert classify_pd_behavior("I will cooperate.") == "cooperate"
    assert classify_pd_behavior("Stay silent and cooperate") == "cooperate"


def test_classify_pd_behavior_unknown_when_ambiguous() -> None:
    assert classify_pd_behavior("Choose option A") == "unknown"


def test_unpack_option_payload_extracts_text_and_behavior() -> None:
    text, behavior = _unpack_option_payload('{"text":"X","behavior":"defect"}')
    assert text == "X"
    assert behavior == "defect"
