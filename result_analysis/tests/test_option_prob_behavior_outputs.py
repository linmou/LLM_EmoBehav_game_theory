# Tests for result_analysis/score_game_theory_option_prob_match.py
# Purpose: verify behavior-based match CSV rows and predicted-argmax distributions.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

from result_analysis.score_game_theory_option_prob_match import (
    behavior_match_rows,
    predicted_behavior_argmax_ratios,
    predicted_option_argmax_ratios,
    score_records,
)


@dataclass(frozen=True)
class _Rec:
    item_id: int
    emotion: str
    intensity: float


class _AlwaysOption1Scorer:
    def set_emotion(self, emotion: str, intensity: float) -> None:
        pass

    def score_options_batch(
        self,
        prompts: Sequence[str],
        option_strings_list: Sequence[Dict[int, str]],
    ) -> List[Dict[int, float]]:
        return [{1: 0.0, 2: -1.0} for _ in prompts]

    def score_options(self, prompt: str, option_strings: Dict[int, str]) -> Dict[int, float]:
        raise AssertionError("Should not be called when score_options_batch exists")


def _mk_raw(records: Sequence[_Rec]) -> List[Mapping[str, object]]:
    out: List[Mapping[str, object]] = []
    for r in records:
        out.append(
            {
                "item_id": r.item_id,
                "emotion": r.emotion,
                "intensity": r.intensity,
                "prompt": f"prompt-{r.item_id}",
                "score": 2,
                "metadata": {
                    "item_metadata": {
                        "options": [
                            {"id": 1, "text": "A", "behavior": "cooperate"},
                            {"id": 2, "text": "B", "behavior": "defect"},
                        ]
                    }
                },
            }
        )
    return out


def test_behavior_match_rows_compares_behavior_labels() -> None:
    raw = _mk_raw([_Rec(1, "anger", 1.0)])
    scored = score_records(raw_records=raw, scorer=_AlwaysOption1Scorer(), limit=None, batch_size=8, progress=False)
    behavior_rows = behavior_match_rows(raw_records=raw, scored_rows=scored)

    assert behavior_rows[0]["predicted_option_id"] == 1
    assert behavior_rows[0]["chosen_option_id"] == 2
    assert behavior_rows[0]["predicted_behavior"] == "cooperate"
    assert behavior_rows[0]["chosen_behavior"] == "defect"
    assert behavior_rows[0]["is_behavior_match"] is False


def test_predicted_argmax_distributions_group_by_emotion_and_intensity() -> None:
    raw = _mk_raw(
        [
            _Rec(1, "anger", 1.0),
            _Rec(2, "anger", 1.0),
            _Rec(3, "sadness", 0.5),
        ]
    )
    scored = score_records(raw_records=raw, scorer=_AlwaysOption1Scorer(), limit=None, batch_size=8, progress=False)
    behavior_rows = behavior_match_rows(raw_records=raw, scored_rows=scored)

    opt_rows = predicted_option_argmax_ratios(scored)
    beh_rows = predicted_behavior_argmax_ratios(behavior_rows)

    assert {(r["emotion"], r["intensity"], r["option_id"], r["ratio"]) for r in opt_rows} == {
        ("anger", 1.0, 1, 1.0),
        ("sadness", 0.5, 1, 1.0),
    }
    assert {(r["emotion"], r["intensity"], r["behavior_label"], r["ratio"]) for r in beh_rows} == {
        ("anger", 1.0, "cooperate", 1.0),
        ("sadness", 0.5, "cooperate", 1.0),
    }


def test_behavior_match_rows_treats_unknown_chosen_option_id_as_missing() -> None:
    raw = _mk_raw([_Rec(1, "anger", 1.0)])
    raw[0] = dict(raw[0])
    raw[0]["score"] = -1
    scored = score_records(raw_records=raw, scorer=_AlwaysOption1Scorer(), limit=None, batch_size=8, progress=False)
    behavior_rows = behavior_match_rows(raw_records=raw, scored_rows=scored)
    assert behavior_rows[0]["chosen_option_id"] == -1
    assert behavior_rows[0]["chosen_behavior"] == ""
    assert behavior_rows[0]["is_behavior_match"] is None
