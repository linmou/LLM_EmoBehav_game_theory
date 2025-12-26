# Tests `result_analysis/trust_game_trustor_expected_score.py`: skip rows lacking neutral baseline.

import pandas as pd

from result_analysis.trust_game_expected_score import TRUSTOR_SPEC, _compute_item_expected_score_deltas


def test_compute_item_expected_score_deltas_skips_missing_neutral_baseline() -> None:
    df = pd.DataFrame(
        [
            # item_id=1 has neutral baseline.
            {
                "model": "m",
                "item_id": 1,
                "emotion": "neutral",
                "intensity": 0.0,
                "behavior": "trust_none",
                "decision_score": 0,
            },
            {
                "model": "m",
                "item_id": 1,
                "emotion": "anger",
                "intensity": 1.0,
                "behavior": "trust_low",
                "decision_score": 1,
            },
            # item_id=2 is missing neutral baseline; should be skipped (not crash).
            {
                "model": "m",
                "item_id": 2,
                "emotion": "anger",
                "intensity": 1.0,
                "behavior": "trust_high",
                "decision_score": 2,
            },
        ]
    )

    rate = _compute_item_expected_score_deltas(df, spec=TRUSTOR_SPEC)

    assert len(rate) == 1
    row = rate.iloc[0]
    assert row["model"] == "m"
    assert int(row["item_id"]) == 1
    assert row["emotion"] == "anger"
    assert float(row["delta_decision_score"]) == 1.0
