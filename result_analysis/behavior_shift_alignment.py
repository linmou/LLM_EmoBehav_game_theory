#!/usr/bin/env python3
# Purpose: compute behaviour-shift alignment from significance-table rows using enum-backed alignment specs.

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from statistics import mean
from typing import Any

from constants import Emotions, GameNames
from games.game_configs import GAME_CONFIGS


@dataclass(frozen=True)
class AlignmentSpec:
    game: GameNames
    focal_behavior: str
    expected_by_emotion: dict[Emotions, int]



DEFAULT_ALIGNMENT_SPECS: dict[GameNames, AlignmentSpec] = {
    GameNames.PRISONERS_DILEMMA: AlignmentSpec(
        game=GameNames.PRISONERS_DILEMMA,
        focal_behavior="cooperate",
        expected_by_emotion={
            Emotions.HAPPINESS: -1,
            Emotions.ANGER: -1,
            Emotions.DISGUST: -1,
            Emotions.FEAR: 0,
            Emotions.SADNESS: 0,
            Emotions.SURPRISE: 0,
        },
    ),
    GameNames.STAG_HUNT: AlignmentSpec(
        game=GameNames.STAG_HUNT,
        focal_behavior="cooperate",
        expected_by_emotion={
            Emotions.HAPPINESS: 1,
            Emotions.ANGER: 0,
            Emotions.DISGUST: 0,
            Emotions.FEAR: -1,
            Emotions.SADNESS: 0,
            Emotions.SURPRISE: 0,
        },
    ),
    GameNames.ESCALATION_GAME: AlignmentSpec(
        game=GameNames.ESCALATION_GAME,
        focal_behavior="escalate",
        expected_by_emotion={
            Emotions.HAPPINESS: 0,
            Emotions.ANGER: 1,
            Emotions.DISGUST: 0,
            Emotions.FEAR: -1,
            Emotions.SADNESS: 0,
            Emotions.SURPRISE: 0,
        },
    ),
    GameNames.TRUST_GAME_TRUSTOR: AlignmentSpec(
        game=GameNames.TRUST_GAME_TRUSTOR,
        focal_behavior="trust_high",
        expected_by_emotion={
            Emotions.HAPPINESS: 1,
            Emotions.ANGER: -1,
            Emotions.DISGUST: -1,
            Emotions.FEAR: -1,
            Emotions.SADNESS: 0,
            Emotions.SURPRISE: 0,
        },
    ),
    GameNames.TRUST_GAME_TRUSTEE: AlignmentSpec(
        game=GameNames.TRUST_GAME_TRUSTEE,
        focal_behavior="return_high",
        expected_by_emotion={
            Emotions.HAPPINESS: 1,
            Emotions.ANGER: -1,
            Emotions.DISGUST: -1,
            Emotions.FEAR: -1,
            Emotions.SADNESS: 0,
            Emotions.SURPRISE: 0,
        },
    ),
    GameNames.ULTIMATUM_GAME_PROPOSER: AlignmentSpec(
        game=GameNames.ULTIMATUM_GAME_PROPOSER,
        focal_behavior="offer_high",
        expected_by_emotion={
            Emotions.HAPPINESS: 1,
            Emotions.ANGER: -1,
            Emotions.DISGUST: -1,
            Emotions.FEAR: 1,
            Emotions.SADNESS: -1,
            Emotions.SURPRISE: 0,
        },
    ),
    GameNames.ULTIMATUM_GAME_RESPONDER: AlignmentSpec(
        game=GameNames.ULTIMATUM_GAME_RESPONDER,
        focal_behavior="reject",
        expected_by_emotion={
            Emotions.HAPPINESS: -1,
            Emotions.ANGER: 1,
            Emotions.DISGUST: 1,
            Emotions.FEAR: 1,
            Emotions.SADNESS: 1,
            Emotions.SURPRISE: 0,
        },
    ),
    GameNames.SEALED_AUCTION: AlignmentSpec(
        game=GameNames.SEALED_AUCTION,
        focal_behavior="devote_high",
        expected_by_emotion={
            Emotions.HAPPINESS: 1,
            Emotions.ANGER: 1,
            Emotions.DISGUST: -1,
            Emotions.FEAR: -1,
            Emotions.SADNESS: 1,
            Emotions.SURPRISE: 1,
        },
    ),
    GameNames.BEAUTY_CONTEST: AlignmentSpec(
        game=GameNames.BEAUTY_CONTEST,
        focal_behavior="commit_3",
        expected_by_emotion={
            Emotions.HAPPINESS: 0,
            Emotions.ANGER: -1,
            Emotions.DISGUST: 0,
            Emotions.FEAR: 0,
            Emotions.SADNESS: 0,
            Emotions.SURPRISE: 0,
        },
    ),
}


def _bool_from_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n", ""}:
            return False
    raise ValueError(f"Cannot interpret significance flag: {value!r}")


def _float_from_row(row: Mapping[str, object], key: str) -> float:
    value = row.get(key)
    if value is None:
        raise KeyError(f"Missing required column: {key}")
    if not isinstance(value, (int, float, str)):
        raise ValueError(f"Column {key} must be numeric-compatible, got {type(value).__name__}")
    return float(value)


def _infer_model_sign(row: Mapping[str, object], significance_threshold: float) -> int:
    delta = _float_from_row(row, "delta")
    significant_value = row.get("significant")
    if significant_value is not None:
        significant = _bool_from_value(significant_value)
    elif row.get("q_value") is not None:
        significant = _float_from_row(row, "q_value") < significance_threshold
    elif row.get("p_value") is not None:
        significant = _float_from_row(row, "p_value") < significance_threshold
    else:
        raise KeyError("Row must include one of: significant, q_value, p_value")

    if not significant or delta == 0.0:
        return 0
    return 1 if delta > 0.0 else -1


def _alignment_value(model_sign: int, expected_sign: int) -> float:
    if expected_sign == 0:
        return 1.0 if model_sign == 0 else -1.0
    if model_sign == 0:
        return 0.0
    return 1.0 if model_sign == expected_sign else -1.0


def _behavior_fields_for_game(game: GameNames) -> set[str]:
    config = GAME_CONFIGS[game]
    scenario_class = config["scenario_class"]
    if game is GameNames.TRUST_GAME_TRUSTOR:
        field_name = "trustor_behavior_choices"
    elif game is GameNames.TRUST_GAME_TRUSTEE:
        field_name = "trustee_behavior_choices"
    elif game is GameNames.ULTIMATUM_GAME_PROPOSER:
        field_name = "proposer_behavior_choices"
    elif game is GameNames.ULTIMATUM_GAME_RESPONDER:
        field_name = "responder_behavior_choices"
    else:
        field_name = "behavior_choices"
    behavior_model = scenario_class.model_fields[field_name].annotation
    return set(behavior_model.model_fields.keys())


def validate_alignment_specs(alignment_specs: Mapping[GameNames, AlignmentSpec]) -> None:
    for game, spec in alignment_specs.items():
        if spec.game is not game:
            raise ValueError(f"Alignment spec key {game.value} does not match payload game {spec.game.value}")
        valid_behaviors = _behavior_fields_for_game(game)
        if spec.focal_behavior not in valid_behaviors:
            raise ValueError(
                f"{game.value}: invalid focal_behavior={spec.focal_behavior!r}; "
                f"expected one of {sorted(valid_behaviors)}"
            )
        unexpected_emotions = set(spec.expected_by_emotion) - set(Emotions)
        if unexpected_emotions:
            raise ValueError(f"{game.value}: unexpected emotions {sorted(e.value for e in unexpected_emotions)}")
        for emotion, expected_sign in spec.expected_by_emotion.items():
            if expected_sign not in {-1, 0, 1}:
                raise ValueError(
                    f"{game.value}: invalid sign {expected_sign!r} for emotion {emotion.value}; expected one of -1, 0, 1"
                )


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw_values = [float(row["raw_alignment"]) for row in rows]
    normalized_values = [float(row["normalized_alignment"]) for row in rows]
    model_signs = [int(row["model_sign"]) for row in rows]
    matched_rows = sum(1 for value in raw_values if value > 0.0)
    mismatched_rows = sum(1 for value in raw_values if value < 0.0)
    raw_alignment = mean(raw_values)
    return {
        "covered_rows": len(rows),
        "matched_rows": matched_rows,
        "mismatched_rows": mismatched_rows,
        "raw_alignment": raw_alignment,
        "normalized_alignment": mean(normalized_values),
        "label": "NotSig" if all(sign == 0 for sign in model_signs) else f"{raw_alignment:.3f}",
    }


def compute_behavior_shift_alignment(
    significance_rows: Iterable[Mapping[str, object]],
    *,
    alignment_specs: Mapping[GameNames, AlignmentSpec] | None = None,
    significance_threshold: float = 0.05,
) -> dict[str, Any]:
    specs = dict(alignment_specs if alignment_specs is not None else DEFAULT_ALIGNMENT_SPECS)
    validate_alignment_specs(specs)

    covered_rows: list[dict[str, Any]] = []
    skipped_rows = 0

    for row in significance_rows:
        try:
            game = GameNames.from_string(str(row["task"]))
            emotion = Emotions.from_string(str(row["emotion"]))
        except ValueError:
            skipped_rows += 1
            continue
        spec = specs.get(game)
        behavior = str(row["behavior"])
        if spec is None or behavior != spec.focal_behavior:
            skipped_rows += 1
            continue
        expected_sign = spec.expected_by_emotion.get(emotion)
        if expected_sign is None:
            skipped_rows += 1
            continue

        model_sign = _infer_model_sign(row, significance_threshold)
        raw_alignment = _alignment_value(model_sign, expected_sign)
        covered_rows.append(
            {
                "task": game.value,
                "behavior": behavior,
                "emotion": emotion.value,
                "expected_sign": expected_sign,
                "model_sign": model_sign,
                "raw_alignment": raw_alignment,
                "normalized_alignment": (raw_alignment + 1.0) / 2.0,
            }
        )

    if not covered_rows:
        raise ValueError("No significance rows matched the alignment specs")

    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in covered_rows:
        by_task.setdefault(str(row["task"]), []).append(row)

    overall = _summarize_rows(covered_rows)
    overall["skipped_rows"] = skipped_rows

    return {
        "overall": overall,
        "by_task": {task: _summarize_rows(rows) for task, rows in by_task.items()},
        "rows": covered_rows,
    }
