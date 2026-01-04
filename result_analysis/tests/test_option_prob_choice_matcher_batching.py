# Tests for result_analysis/score_game_theory_option_prob_match.py
# Purpose: verify grouping by (emotion,intensity) and batched scoring calls.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

from result_analysis.score_game_theory_option_prob_match import score_records


@dataclass(frozen=True)
class _Rec:
    item_id: int
    emotion: str
    intensity: float


class DummyBatchScorer:
    def __init__(self) -> None:
        self.set_calls: List[Tuple[str, float]] = []
        self.batch_calls: List[int] = []

    def set_emotion(self, emotion: str, intensity: float) -> None:
        self.set_calls.append((emotion, float(intensity)))

    def score_options_batch(
        self,
        prompts: Sequence[str],
        option_strings_list: Sequence[Dict[int, str]],
    ) -> List[Dict[int, float]]:
        self.batch_calls.append(len(prompts))
        # Always prefer option 1
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
                "score": 1,
                "metadata": {"item_metadata": {"options": [{"id": 1, "text": "A"}, {"id": 2, "text": "B"}]}},
            }
        )
    return out


def test_score_records_groups_by_emotion_and_intensity_and_batches() -> None:
    raw = _mk_raw(
        [
            _Rec(1, "anger", 1.0),
            _Rec(2, "anger", 1.0),
            _Rec(3, "sadness", 0.5),
            _Rec(4, "sadness", 0.5),
            _Rec(5, "sadness", 0.5),
        ]
    )
    scorer = DummyBatchScorer()
    rows = score_records(raw_records=raw, scorer=scorer, limit=None, batch_size=2, progress=False)

    assert [r["item_id"] for r in rows] == [1, 2, 3, 4, 5]
    # One set_emotion per group
    assert scorer.set_calls == [("anger", 1.0), ("sadness", 0.5)]
    # Batching within each group, batch_size=2 => [2] then [2,1]
    assert scorer.batch_calls == [2, 2, 1]


def test_score_records_allows_missing_score() -> None:
    raw = _mk_raw([_Rec(1, "anger", 1.0)])
    raw[0] = dict(raw[0])
    raw[0]["score"] = None
    scorer = DummyBatchScorer()
    rows = score_records(raw_records=raw, scorer=scorer, limit=None, batch_size=2, progress=False)
    assert rows[0]["chosen_option_id"] is None
    assert rows[0]["is_match"] is None
