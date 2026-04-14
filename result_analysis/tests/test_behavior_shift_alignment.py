"""
result_analysis/tests/test_behavior_shift_alignment.py
Purpose: TDD for result_analysis/behavior_shift_alignment.py class-based alignment specs.
Targets: result_analysis/behavior_shift_alignment.py.
"""

from constants import Emotions, GameNames
from result_analysis.behavior_shift_alignment import (
    AlignmentSpec,
    DEFAULT_ALIGNMENT_SPECS,
    compute_behavior_shift_alignment,
    validate_alignment_specs,
)


def test_default_alignment_specs_use_enums_and_behavior_choice_field_names() -> None:
    pd_spec = DEFAULT_ALIGNMENT_SPECS[GameNames.PRISONERS_DILEMMA]
    stag_spec = DEFAULT_ALIGNMENT_SPECS[GameNames.STAG_HUNT]
    trustor_spec = DEFAULT_ALIGNMENT_SPECS[GameNames.TRUST_GAME_TRUSTOR]
    trustee_spec = DEFAULT_ALIGNMENT_SPECS[GameNames.TRUST_GAME_TRUSTEE]
    proposer_spec = DEFAULT_ALIGNMENT_SPECS[GameNames.ULTIMATUM_GAME_PROPOSER]
    beauty_spec = DEFAULT_ALIGNMENT_SPECS[GameNames.BEAUTY_CONTEST]

    assert isinstance(pd_spec, AlignmentSpec)
    assert pd_spec.game is GameNames.PRISONERS_DILEMMA
    assert pd_spec.focal_behavior == "cooperate"
    assert pd_spec.expected_by_emotion[Emotions.ANGER] == -1
    assert stag_spec.game is GameNames.STAG_HUNT
    assert stag_spec.focal_behavior == "cooperate"
    assert stag_spec.expected_by_emotion[Emotions.HAPPINESS] == 1
    assert stag_spec.expected_by_emotion[Emotions.ANGER] == -1
    assert stag_spec.expected_by_emotion[Emotions.DISGUST] == -1
    assert stag_spec.expected_by_emotion[Emotions.FEAR] == 0
    assert trustor_spec.focal_behavior == "trust_high"
    assert trustee_spec.focal_behavior == "return_high"
    assert trustee_spec.expected_by_emotion[Emotions.HAPPINESS] == 1
    assert trustee_spec.expected_by_emotion[Emotions.ANGER] == -1
    assert proposer_spec.focal_behavior == "offer_high"
    assert proposer_spec.expected_by_emotion[Emotions.HAPPINESS] == 1
    assert proposer_spec.expected_by_emotion[Emotions.ANGER] == -1
    assert beauty_spec.focal_behavior == "commit_3"


def test_compute_behavior_shift_alignment_supports_default_trustee_and_proposer_specs() -> None:
    rows = [
        {
            "task": "Trust_Game_Trustee",
            "behavior": "return_high",
            "emotion": "happiness",
            "delta": 0.30,
            "significant": True,
        },
        {
            "task": "Ultimatum_Game_Proposer",
            "behavior": "offer_high",
            "emotion": "anger",
            "delta": -0.20,
            "significant": True,
        },
    ]

    result = compute_behavior_shift_alignment(rows)

    assert result["overall"]["covered_rows"] == 2
    assert result["by_task"]["Trust_Game_Trustee"]["raw_alignment"] == 1.0
    assert result["by_task"]["Ultimatum_Game_Proposer"]["raw_alignment"] == 1.0


def test_validate_alignment_specs_accepts_default_specs() -> None:
    validate_alignment_specs(DEFAULT_ALIGNMENT_SPECS)


def test_validate_alignment_specs_rejects_invalid_focal_behavior() -> None:
    bad_spec = AlignmentSpec(
        game=GameNames.PRISONERS_DILEMMA,
        focal_behavior="trust_high",
        expected_by_emotion={Emotions.ANGER: -1},
    )

    try:
        validate_alignment_specs({GameNames.PRISONERS_DILEMMA: bad_spec})
    except ValueError as exc:
        assert "invalid focal_behavior" in str(exc)
    else:
        raise AssertionError("expected validate_alignment_specs to reject invalid focal behavior")


def test_compute_behavior_shift_alignment_uses_default_alignment_specs() -> None:
    rows = [
        {
            "task": "Prisoners_Dilemma",
            "behavior": "cooperate",
            "emotion": "anger",
            "delta": -0.20,
            "q_value": 0.01,
        },
        {
            "task": "Prisoners_Dilemma",
            "behavior": "cooperate",
            "emotion": "fear",
            "delta": 0.25,
            "q_value": 0.01,
        },
        {
            "task": "Prisoners_Dilemma",
            "behavior": "cooperate",
            "emotion": "sad",
            "delta": 0.10,
            "q_value": 0.50,
        },
    ]

    result = compute_behavior_shift_alignment(rows)

    assert DEFAULT_ALIGNMENT_SPECS[GameNames.PRISONERS_DILEMMA].expected_by_emotion[Emotions.ANGER] == -1
    assert DEFAULT_ALIGNMENT_SPECS[GameNames.PRISONERS_DILEMMA].expected_by_emotion[Emotions.FEAR] == 0
    assert result["overall"]["covered_rows"] == 3
    assert result["overall"]["raw_alignment"] == 1.0 / 3.0
    assert result["overall"]["normalized_alignment"] == 2.0 / 3.0
    assert result["overall"]["matched_rows"] == 2
    assert result["overall"]["mismatched_rows"] == 1


def test_compute_behavior_shift_alignment_marks_nonsignificant_task_as_notsig() -> None:
    rows = [
        {
            "task": "Beauty_Contest",
            "behavior": "commit_3",
            "emotion": "anger",
            "delta": -0.10,
            "significant": False,
        },
        {
            "task": "Beauty_Contest",
            "behavior": "commit_3",
            "emotion": "fear",
            "delta": 0.20,
            "significant": False,
        },
    ]

    result = compute_behavior_shift_alignment(rows)

    task_summary = result["by_task"]["Beauty_Contest"]
    assert task_summary["label"] == "NotSig"
    assert task_summary["raw_alignment"] == 0.5
    assert task_summary["normalized_alignment"] == 0.75


def test_compute_behavior_shift_alignment_skips_unmapped_rows_and_supports_custom_map() -> None:
    rows = [
        {
            "task": "Escalation_Game",
            "behavior": "escalate",
            "emotion": "anger",
            "delta": 0.30,
            "significant": True,
        },
        {
            "task": "Unknown_Game",
            "behavior": "other",
            "emotion": "anger",
            "delta": 0.30,
            "significant": True,
        },
    ]

    result = compute_behavior_shift_alignment(
        rows,
        alignment_specs={
            GameNames.ESCALATION_GAME: AlignmentSpec(
                game=GameNames.ESCALATION_GAME,
                focal_behavior="escalate",
                expected_by_emotion={Emotions.ANGER: 1},
            ),
        },
    )

    assert result["overall"]["covered_rows"] == 1
    assert result["overall"]["skipped_rows"] == 1
    assert result["overall"]["raw_alignment"] == 1.0
