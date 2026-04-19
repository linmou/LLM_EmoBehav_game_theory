#!/usr/bin/env python3
# Purpose: transform curated social game rows into loadable game scenario datasets with resume and audit artifacts.

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from constants import GameNames
from games.game_configs import get_game_config


SUPPORTED_SOCIAL_GAMES = {
    "beauty_contest": {
        "game_name": GameNames.BEAUTY_CONTEST,
        "success_file": "beauty_contest.success.json",
        "failure_file": "beauty_contest.failures.jsonl",
        "skip_file": "beauty_contest.skipped.jsonl",
        "prompt_target": "Beauty Contest scenario",
    },
    "escalation_game": {
        "game_name": GameNames.ESCALATION_GAME,
        "success_file": "escalation_game.success.json",
        "failure_file": "escalation_game.failures.jsonl",
        "skip_file": "escalation_game.skipped.jsonl",
        "prompt_target": "Escalation Game scenario",
    },
    "prisoners_dilemma": {
        "game_name": GameNames.PRISONERS_DILEMMA,
        "success_file": "prisoners_dilemma.success.json",
        "failure_file": "prisoners_dilemma.failures.jsonl",
        "skip_file": "prisoners_dilemma.skipped.jsonl",
        "prompt_target": "Prisoners' Dilemma scenario",
    },
    "trust_game_trustor": {
        "game_name": GameNames.TRUST_GAME_TRUSTOR,
        "success_file": "trust_game_trustor.success.json",
        "failure_file": "trust_game_trustor.failures.jsonl",
        "skip_file": "trust_game_trustor.skipped.jsonl",
        "prompt_target": "Trust Game trustor scenario",
    },
    "trust_game_trustee": {
        "game_name": GameNames.TRUST_GAME_TRUSTEE,
        "success_file": "trust_game_trustee.success.json",
        "failure_file": "trust_game_trustee.failures.jsonl",
        "skip_file": "trust_game_trustee.skipped.jsonl",
        "prompt_target": "Trust Game trustee scenario",
    },
    "ultimatum_game_proposer": {
        "game_name": GameNames.ULTIMATUM_GAME_PROPOSER,
        "success_file": "ultimatum_game_proposer.success.json",
        "failure_file": "ultimatum_game_proposer.failures.jsonl",
        "skip_file": "ultimatum_game_proposer.skipped.jsonl",
        "prompt_target": "Ultimatum Game proposer scenario",
    },
    "ultimatum_game_responder": {
        "game_name": GameNames.ULTIMATUM_GAME_RESPONDER,
        "success_file": "ultimatum_game_responder.success.json",
        "failure_file": "ultimatum_game_responder.failures.jsonl",
        "skip_file": "ultimatum_game_responder.skipped.jsonl",
        "prompt_target": "Ultimatum Game responder scenario",
    },
}
DEFAULT_TRANSFORM_SAMPLE_ROOT = (
    Path(__file__).resolve().parent / "transform_to_natural_lannguage_samples" / "diplomacy"
)
DEFAULT_RUBRIC_PATH = DEFAULT_TRANSFORM_SAMPLE_ROOT / "transform_rubrics.md"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def default_few_shot_path(social_game: str) -> Path:
    return DEFAULT_TRANSFORM_SAMPLE_ROOT / f"{social_game}_few_shot_examples.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="transform_social_game_cases")
    parser.add_argument("--social-game", required=True)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--few-shot-path", default=None)
    parser.add_argument("--rubric-path", default=str(DEFAULT_RUBRIC_PATH))
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=None)
    return parser


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, default=json_default),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True, default=json_default) for row in rows) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_existing_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return load_jsonl(path)


def social_game_config(social_game: str) -> dict[str, Any]:
    cfg = SUPPORTED_SOCIAL_GAMES.get(social_game)
    if cfg is None:
        raise ValueError(
            f"Unsupported social game: {social_game}. Supported values: {', '.join(sorted(SUPPORTED_SOCIAL_GAMES))}"
        )
    game_config = get_game_config(cast(GameNames, cfg["game_name"]))
    return {
        **cfg,
        "target_game_name": game_config["game_name"],
        "scenario_class": game_config["scenario_class"],
        "payoff_matrix": game_config["payoff_matrix"],
    }


def extract_run_variants(source_rows: list[dict[str, Any]]) -> set[str]:
    variants: set[str] = set()
    for source_row in source_rows:
        variant_name = source_row.get("variant_name")
        if isinstance(variant_name, str) and variant_name.strip():
            variants.add(variant_name)
    return variants


def load_prompt_pack(
    social_game: str,
    rubric_path: Path,
    few_shot_path: Path,
    run_variants: set[str] | None = None,
) -> dict[str, Any]:
    cfg = social_game_config(social_game)
    if not rubric_path.exists():
        raise FileNotFoundError(f"Rubric file not found: {rubric_path}")
    if not few_shot_path.exists():
        raise FileNotFoundError(f"Few-shot file not found: {few_shot_path}")
    few_shot_examples = read_json(few_shot_path)
    if not isinstance(few_shot_examples, list):
        raise ValueError(f"Few-shot file must contain a JSON list: {few_shot_path}")
    if run_variants:
        filtered_examples = []
        for example in few_shot_examples:
            if not isinstance(example, dict):
                continue
            example_input = example.get("input")
            if not isinstance(example_input, dict):
                continue
            variant_name = example_input.get("variant_name")
            if isinstance(variant_name, str) and variant_name in run_variants:
                filtered_examples.append(example)
        few_shot_examples = filtered_examples
    return {
        "social_game": social_game,
        "rubric_path": rubric_path,
        "rubric_text": rubric_path.read_text(encoding="utf-8").strip(),
        "few_shot_path": few_shot_path,
        "few_shot_examples": few_shot_examples,
        "run_variants": sorted(run_variants) if run_variants else [],
        "target_game_name": cfg["target_game_name"],
        "scenario_class": cfg["scenario_class"],
        "payoff_matrix": cfg["payoff_matrix"],
        "prompt_target": cfg["prompt_target"],
    }


def build_system_prompt(prompt_pack: dict[str, Any]) -> str:
    few_shot_block = json.dumps(prompt_pack["few_shot_examples"], ensure_ascii=True, indent=2)
    return (
        f"{prompt_pack['rubric_text']}\n\n"
        "Use the following few-shot examples as style and shape references.\n"
        "Return only one JSON object for the transformed case.\n\n"
        f"{few_shot_block}"
    )


def build_identity_key(source_row: dict[str, Any]) -> str:
    source = source_row.get("source")
    if not isinstance(source, dict):
        raise ValueError("source row must include an object 'source'")
    case_id = source_row.get("id")
    source_game_id = source.get("game_id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("source row must include a non-empty top-level 'id'")
    if not isinstance(source_game_id, str) or not source_game_id.strip():
        raise ValueError("source row must include a non-empty 'source.game_id'")
    return f"{case_id}::{source_game_id}"


def few_shot_variant_name(example: dict[str, Any]) -> str | None:
    example_input = example.get("input")
    if not isinstance(example_input, dict):
        return None
    variant_name = example_input.get("variant_name")
    if isinstance(variant_name, str) and variant_name.strip():
        return variant_name
    return None


def validate_row_few_shot_pool(source_row: dict[str, Any], prompt_pack: dict[str, Any]) -> None:
    row_variant = source_row.get("variant_name")
    if not isinstance(row_variant, str) or not row_variant.strip():
        raise ValueError("source row must include a non-empty 'variant_name'")
    same_variant_count = sum(
        1
        for example in prompt_pack["few_shot_examples"]
        if isinstance(example, dict) and few_shot_variant_name(example) == row_variant
    )
    if same_variant_count < 1:
        raise ValueError("few-shot pool must supply at least 1 same-variant example for each source row")
    run_variants = {
        variant_name
        for variant_name in prompt_pack.get("run_variants", [])
        if isinstance(variant_name, str) and variant_name
    }
    if len(run_variants - {row_variant}) == 0:
        return
    cross_variant_count = sum(
        1
        for example in prompt_pack["few_shot_examples"]
        if isinstance(example, dict) and few_shot_variant_name(example) != row_variant
    )
    if cross_variant_count < 2:
        raise ValueError("few-shot pool must supply at least 2 cross-variant examples for each source row")


def build_row_prompt_pack(source_row: dict[str, Any], prompt_pack: dict[str, Any]) -> dict[str, Any]:
    row_variant = source_row.get("variant_name")
    assert isinstance(row_variant, str) and row_variant.strip()
    same_variant_examples = [
        example
        for example in prompt_pack["few_shot_examples"]
        if isinstance(example, dict) and few_shot_variant_name(example) == row_variant
    ]
    cross_variant_examples = [
        example
        for example in prompt_pack["few_shot_examples"]
        if isinstance(example, dict) and few_shot_variant_name(example) != row_variant
    ]

    row_prompt_pack = dict(prompt_pack)
    row_prompt_pack["few_shot_examples"] = (
        rank_examples_by_ngram_gain(same_variant_examples)
        + rank_examples_by_ngram_gain(cross_variant_examples)[:2]
    )
    return row_prompt_pack


def render_progress(done: int, total: int) -> str:
    total = max(total, 1)
    pct = int((done / total) * 100)
    return f"progress={done}/{total} ({pct}%)"


def load_api_key() -> str:
    api_key = os.environ.get("DPSK_API", "").strip().strip("'").strip('"')
    if not api_key:
        raise RuntimeError("DPSK_API is required in the environment or .env")
    return api_key


def load_dotenv_if_available(env_path: Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_path, override=False)


def build_openai_client() -> Any:
    from openai import OpenAI

    return OpenAI(
        api_key=load_api_key(),
        base_url=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL),
    )


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9'.-]+", text.lower())


def _ngrams(text: str, n: int) -> list[str]:
    words = _word_tokens(text)
    if len(words) < n:
        return []
    return [" ".join(words[idx:idx + n]) for idx in range(len(words) - n + 1)]


def _distinct_ratio(descriptions: list[str], n: int) -> float:
    all_ngrams: list[str] = []
    for description in descriptions:
        all_ngrams.extend(_ngrams(description, n))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def example_lexical_surface(example: dict[str, Any]) -> str:
    output = example.get("output")
    if not isinstance(output, dict):
        return ""
    description = output.get("description")
    behavior_choices = output.get("behavior_choices")
    choice_values: list[str] = []
    if isinstance(behavior_choices, dict):
        for value in behavior_choices.values():
            if isinstance(value, str):
                choice_values.append(value)
    parts: list[str] = []
    if isinstance(description, str) and description.strip():
        parts.append(description)
    if choice_values:
        parts.append(" ".join(choice_values))
    return "\n".join(parts)


def weighted_ngram_gain_score(text: str, selected_texts: list[str]) -> tuple[int, int, str]:
    selected_3grams = set()
    selected_4grams = set()
    selected_5grams = set()
    for selected_text in selected_texts:
        selected_3grams.update(_ngrams(selected_text, 3))
        selected_4grams.update(_ngrams(selected_text, 4))
        selected_5grams.update(_ngrams(selected_text, 5))

    grams_3 = set(_ngrams(text, 3))
    grams_4 = set(_ngrams(text, 4))
    grams_5 = set(_ngrams(text, 5))
    gain = (
        len(grams_3 - selected_3grams)
        + 2 * len(grams_4 - selected_4grams)
        + 3 * len(grams_5 - selected_5grams)
    )
    overlap = (
        len(grams_3 & selected_3grams)
        + 2 * len(grams_4 & selected_4grams)
        + 3 * len(grams_5 & selected_5grams)
    )
    return gain - overlap, len(grams_3) + len(grams_4) + len(grams_5), text


def rank_examples_by_ngram_gain(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    remaining = list(examples)
    ranked: list[dict[str, Any]] = []
    selected_texts: list[str] = []
    while remaining:
        next_example = max(
            remaining,
            key=lambda example: weighted_ngram_gain_score(
                example_lexical_surface(example),
                selected_texts,
            ),
        )
        ranked.append(next_example)
        selected_texts.append(example_lexical_surface(next_example))
        remaining.remove(next_example)
    return ranked


def compute_description_diversity_report(descriptions: list[str]) -> dict[str, Any]:
    repeated_3grams: dict[str, int] = {}
    for description in descriptions:
        for gram in _ngrams(description, 3):
            repeated_3grams[gram] = repeated_3grams.get(gram, 0) + 1

    repeated_3gram_rows = [
        {"ngram": gram, "count": count}
        for gram, count in sorted(
            repeated_3grams.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if count > 1
    ]

    return {
        "description_count": len(descriptions),
        "selected_description_metrics": {
            "distinct_1": _distinct_ratio(descriptions, 1),
            "distinct_2": _distinct_ratio(descriptions, 2),
            "distinct_3": _distinct_ratio(descriptions, 3),
        },
        "repeated_3grams": repeated_3gram_rows,
    }


def extract_response_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("model response did not contain choices")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model response did not contain message content")
    return content.strip()


def parse_json_text(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("model response must parse to a JSON object")
    return payload


def inject_game_fields(row: dict[str, Any], prompt_pack: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    payload["game_name"] = prompt_pack["target_game_name"]
    payload["payoff_matrix"] = prompt_pack["payoff_matrix"]
    return payload


def _participant_choice_sets(payload: dict[str, Any]) -> dict[str, dict[str, str]]:
    participants = payload.get("participants")
    if not isinstance(participants, list):
        return {}

    shared_choices = payload.get("behavior_choices")
    if isinstance(shared_choices, dict):
        choice_sets: dict[str, dict[str, str]] = {}
        for participant in participants:
            if not isinstance(participant, dict):
                continue
            name = participant.get("name")
            if isinstance(name, str) and name:
                choice_sets[name] = shared_choices
        return choice_sets

    role_choice_sets: dict[str, dict[str, str]] = {}
    for role, field_name in (
        ("Trustor", "trustor_behavior_choices"),
        ("Trustee", "trustee_behavior_choices"),
        ("Proposer", "proposer_behavior_choices"),
        ("Responder", "responder_behavior_choices"),
    ):
        choices = payload.get(field_name)
        if isinstance(choices, dict):
            role_choice_sets[role] = choices

    choice_sets = {}
    for participant in participants:
        if not isinstance(participant, dict):
            continue
        name = participant.get("name")
        participant_role = participant.get("role")
        if (
            isinstance(name, str)
            and name
            and isinstance(participant_role, str)
            and participant_role in role_choice_sets
        ):
            choice_sets[name] = role_choice_sets[participant_role]
    return choice_sets


def _behavior_marker_score(key: str, text: str) -> int:
    marker_groups = {
        "none": ("0", "0%", "zero", "none", "nothing", "no"),
        "low": ("30", "30%", "low", "limited", "light", "modest", "moderate"),
        "medium": ("40", "40-50", "40-50%", "40%", "45%", "medium", "moderate", "limited"),
        "high": ("80", "80%", "high", "generous", "substantial", "strong", "major"),
        "withdraw": ("normal", "hold", "steady", "stay", "remain", "keep"),
        "escalate": ("escalate", "increase", "advance", "push", "pump", "attack"),
        "accept": ("accept", "agree", "take"),
        "reject": ("reject", "decline", "refuse"),
    }
    score = 0
    lower_key = key.lower()
    lower_text = text.lower()
    for marker_key, markers in marker_groups.items():
        if marker_key in lower_key:
            score += sum(1 for marker in markers if marker in lower_text)
    return score


def _canonicalize_behavior_choice(action_text: str, choices: dict[str, str]) -> str | None:
    if action_text in choices.values():
        return action_text

    action_tokens = set(_word_tokens(action_text))
    ranked: list[tuple[int, int, str]] = []
    for key, choice_text in choices.items():
        choice_tokens = set(_word_tokens(choice_text))
        overlap = len(action_tokens & choice_tokens)
        marker_score = _behavior_marker_score(key, action_text)
        ranked.append((marker_score, overlap, choice_text))

    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    best_marker, best_overlap, best_choice = ranked[0]
    if best_marker == 0 and best_overlap == 0:
        return None
    if len(ranked) > 1 and ranked[1][:2] == (best_marker, best_overlap):
        return None
    return best_choice


def canonicalize_previous_actions_against_behavior_choices(payload: dict[str, Any]) -> dict[str, Any]:
    previous_actions = payload.get("previous_actions")
    if not isinstance(previous_actions, list) or not previous_actions:
        return payload

    choice_sets = _participant_choice_sets(payload)
    if not choice_sets:
        return payload

    normalized_previous_actions: list[Any] = []
    for action in previous_actions:
        if isinstance(action, dict):
            round_actions = action.get("actions")
            if not isinstance(round_actions, list):
                normalized_previous_actions.append(action)
                continue
            normalized_round_actions: list[dict[str, Any]] = []
            for round_action in round_actions:
                if not isinstance(round_action, dict):
                    normalized_round_actions.append(round_action)
                    continue
                participant = round_action.get("participant")
                decision = round_action.get("action")
                if (
                    isinstance(participant, str)
                    and isinstance(decision, str)
                    and participant in choice_sets
                ):
                    canonical = _canonicalize_behavior_choice(decision, choice_sets[participant])
                    if canonical is not None:
                        normalized_round_actions.append(
                            {**round_action, "action": canonical}
                        )
                        continue
                normalized_round_actions.append(round_action)
            normalized_previous_actions.append({**action, "actions": normalized_round_actions})
            continue

        if isinstance(action, (list, tuple)) and len(action) == 2:
            participant, decision = action
            if (
                isinstance(participant, str)
                and isinstance(decision, str)
                and participant in choice_sets
            ):
                canonical = _canonicalize_behavior_choice(decision, choice_sets[participant])
                if canonical is not None:
                    normalized_previous_actions.append((participant, canonical))
                    continue
        normalized_previous_actions.append(action)

    payload["previous_actions"] = normalized_previous_actions
    return payload


def validate_loadable_with_game_contract(row: dict[str, Any], prompt_pack: dict[str, Any]) -> None:
    scenario_payload = dict(row)
    prompt_pack["scenario_class"](**scenario_payload)


def transform_source_row(
    *,
    source_row: dict[str, Any],
    prompt_pack: dict[str, Any],
    model_name: str,
    max_retries: int = 0,
    temperature: float = 0.0,
    request_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    system_prompt = build_system_prompt(prompt_pack)
    user_prompt = (
        f"Transform the following curated social-game case into one loadable {prompt_pack['prompt_target']}.\n"
        f"{json.dumps(source_row, ensure_ascii=True)}"
    )
    last_error: Exception | None = None
    for _ in range(max_retries + 1):
        try:
            client = build_openai_client()
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                timeout=request_timeout_seconds,
            )
            payload = inject_game_fields(parse_json_text(extract_response_text(response)), prompt_pack)
            payload = canonicalize_previous_actions_against_behavior_choices(payload)
            payload.setdefault("provenance", {})
            payload["provenance"]["id"] = source_row["id"]
            payload["provenance"]["source_game_id"] = source_row["source"]["game_id"]
            payload["provenance"]["source_dataset"] = source_row["source"].get("dataset")
            payload["provenance"]["source_line_number"] = source_row["source"].get("line_number")
            validate_loadable_with_game_contract(payload, prompt_pack)
            return payload
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    assert last_error is not None
    raise last_error


def transform_source_row_candidates(
    *,
    source_row: dict[str, Any],
    prompt_pack: dict[str, Any],
    model_name: str,
    max_retries: int = 0,
    num_candidates: int = 1,
    temperature: float = 0.0,
    request_timeout_seconds: float | None = None,
) -> list[dict[str, Any]]:
    return [
        transform_source_row(
            source_row=source_row,
            prompt_pack=prompt_pack,
            model_name=model_name,
            max_retries=max_retries,
            temperature=temperature,
            request_timeout_seconds=request_timeout_seconds,
        )
        for _ in range(max(1, num_candidates))
    ]


def select_best_candidate(
    candidates: list[dict[str, Any]],
    selected_descriptions: list[str],
) -> dict[str, Any]:
    def overlap_score(description: str) -> tuple[int, int]:
        selected_ngrams = set()
        for selected_description in selected_descriptions:
            selected_ngrams.update(_ngrams(selected_description, 3))
            selected_ngrams.update(_ngrams(selected_description, 4))
        description_ngrams = set(_ngrams(description, 3)) | set(_ngrams(description, 4))
        overlap = len(description_ngrams & selected_ngrams)
        return overlap, len(description_ngrams)

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            overlap_score(candidate.get("description", ""))[0],
            -overlap_score(candidate.get("description", ""))[1],
            candidate.get("description", ""),
        ),
    )
    return ranked[0]


def build_failure_record(
    *,
    source_row: dict[str, Any],
    stage: str,
    error: Exception,
) -> dict[str, Any]:
    source_obj = source_row.get("source")
    source: dict[str, Any] = source_obj if isinstance(source_obj, dict) else {}
    return {
        "identity_key": (
            f"{source_row.get('id')}::{source.get('game_id')}"
            if source_row.get("id") and source.get("game_id")
            else None
        ),
        "id": source_row.get("id"),
        "source_game_id": source.get("game_id"),
        "stage": stage,
        "error_type": error.__class__.__name__,
        "message": str(error),
        "source_snapshot": {
            "id": source_row.get("id"),
            "source": source_row.get("source"),
            "episode_type": source_row.get("episode_type"),
            "variant_name": source_row.get("variant_name"),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def validate_source_row(source_row: dict[str, Any]) -> str:
    if not isinstance(source_row, dict):
        raise ValueError("source row must be a JSON object")
    return build_identity_key(source_row)


def load_existing_success_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"success dataset must contain a JSON list: {path}")
    return [row for row in payload if isinstance(row, dict)]


def identity_from_success_row(row: dict[str, Any]) -> str | None:
    provenance = row.get("provenance")
    if not isinstance(provenance, dict):
        return None
    row_id = provenance.get("id")
    source_game_id = provenance.get("source_game_id")
    if isinstance(row_id, str) and row_id and isinstance(source_game_id, str) and source_game_id:
        return f"{row_id}::{source_game_id}"
    return None


def completed_identities_from_outputs(
    success_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    run_metadata: dict[str, Any] | None,
) -> set[str]:
    identities = {identity for row in success_rows if (identity := identity_from_success_row(row))}
    for row in failure_rows:
        identity = row.get("identity_key")
        if isinstance(identity, str) and identity:
            identities.add(identity)
    if isinstance(run_metadata, dict):
        for identity in run_metadata.get("completed_identities", []):
            if isinstance(identity, str) and identity:
                identities.add(identity)
    return identities


def artifact_paths(output_dir: Path, social_game: str) -> dict[str, Path]:
    cfg = social_game_config(social_game)
    return {
        "success": output_dir / cfg["success_file"],
        "failure": output_dir / cfg["failure_file"],
        "skip": output_dir / cfg["skip_file"],
        "metadata": output_dir / "run_metadata.json",
        "candidates": output_dir / f"{social_game}.candidates.jsonl",
        "diversity": output_dir / "diversity_report.json",
    }


def run_transform(args: argparse.Namespace) -> int:
    social_game_config(args.social_game)
    load_dotenv_if_available(Path(".env"))

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(output_dir, args.social_game)
    resolved_rubric_path = Path(args.rubric_path)
    resolved_few_shot_path = Path(args.few_shot_path) if args.few_shot_path else default_few_shot_path(args.social_game)
    source_rows = load_jsonl(input_path, limit=args.limit)
    run_variants = extract_run_variants(source_rows)

    prompt_pack = load_prompt_pack(
        social_game=args.social_game,
        rubric_path=resolved_rubric_path,
        few_shot_path=resolved_few_shot_path,
        run_variants=run_variants,
    )

    existing_success_rows = [] if args.rerun else load_existing_success_rows(paths["success"])
    existing_failure_rows = [] if args.rerun else load_existing_jsonl(paths["failure"])
    existing_skip_rows = [] if args.rerun else load_existing_jsonl(paths["skip"])
    existing_metadata = None if args.rerun or not paths["metadata"].exists() else read_json(paths["metadata"])

    success_rows = list(existing_success_rows)
    failure_rows = list(existing_failure_rows)
    skipped_rows = list(existing_skip_rows)
    completed_identities = completed_identities_from_outputs(success_rows, failure_rows, existing_metadata)

    total = len(source_rows)
    done = 0

    pending_rows: list[tuple[int, str, dict[str, Any]]] = []
    for index, source_row in enumerate(source_rows):
        try:
            identity_key = validate_source_row(source_row)
        except Exception as exc:  # noqa: BLE001
            failure_rows.append(build_failure_record(source_row=source_row, stage="input_validation", error=exc))
            done += 1
            print(render_progress(done, total))
            continue

        if not args.rerun and identity_key in completed_identities:
            skipped_rows.append(
                {
                    "identity_key": identity_key,
                    "stage": "resume_skip",
                    "message": "identity already completed in prior artifacts",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            done += 1
            print(render_progress(done, total))
            continue

        try:
            validate_row_few_shot_pool(source_row, prompt_pack)
        except Exception as exc:  # noqa: BLE001
            failure_rows.append(build_failure_record(source_row=source_row, stage="few_shot_selection", error=exc))
            completed_identities.add(identity_key)
            done += 1
            print(render_progress(done, total))
            continue

        pending_rows.append((index, identity_key, source_row))

    success_by_index: dict[int, dict[str, Any]] = {}
    failure_by_index: dict[int, dict[str, Any]] = {}
    candidates_by_index: dict[int, list[dict[str, Any]]] = {}

    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
        future_map = {
            executor.submit(
                transform_source_row_candidates,
                source_row=source_row,
                prompt_pack=build_row_prompt_pack(source_row, prompt_pack),
                model_name=args.model,
                max_retries=args.max_retries,
                num_candidates=args.num_candidates,
                temperature=args.temperature,
                request_timeout_seconds=args.request_timeout_seconds,
            ): (index, identity_key, source_row)
            for index, identity_key, source_row in pending_rows
        }

        for future in as_completed(future_map):
            index, identity_key, source_row = future_map[future]
            try:
                candidates_by_index[index] = future.result()
            except Exception as exc:  # noqa: BLE001
                failure_by_index[index] = build_failure_record(source_row=source_row, stage="transform", error=exc)
            completed_identities.add(identity_key)
            done += 1
            print(render_progress(done, total))

    candidate_rows: list[dict[str, Any]] = []
    selected_descriptions = [row.get("description", "") for row in success_rows]
    for index, _, _ in pending_rows:
        if index in candidates_by_index:
            selected = select_best_candidate(candidates_by_index[index], selected_descriptions)
            for candidate_index, candidate in enumerate(candidates_by_index[index]):
                candidate_rows.append(
                    {
                        "candidate_index": candidate_index,
                        "selected": candidate == selected,
                        **candidate,
                    }
                )
            success_rows.append(selected)
            selected_descriptions.append(selected.get("description", ""))
        if index in failure_by_index:
            failure_rows.append(failure_by_index[index])

    write_json(paths["success"], success_rows)
    write_jsonl(paths["failure"], failure_rows)
    write_jsonl(paths["skip"], skipped_rows)
    write_jsonl(paths["candidates"], candidate_rows)

    diversity_report = compute_description_diversity_report(
        [row.get("description", "") for row in success_rows]
    )
    diversity_report["candidate_counts"] = {
        "generated": len(candidate_rows),
        "selected": len(success_rows),
    }
    write_json(paths["diversity"], diversity_report)

    metadata = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "social_game": args.social_game,
        "input_path": str(input_path.resolve()),
        "success_output_path": str(paths["success"].resolve()),
        "failure_output_path": str(paths["failure"].resolve()),
        "skip_output_path": str(paths["skip"].resolve()),
        "model_name": args.model,
        "num_candidates": args.num_candidates,
        "temperature": args.temperature,
        "request_timeout_seconds": args.request_timeout_seconds,
        "rubric_path": str(resolved_rubric_path.resolve()),
        "few_shot_path": str(resolved_few_shot_path.resolve()),
        "run_variants": sorted(run_variants),
        "completed_identities": sorted(completed_identities),
        "counts": {
            "total": total,
            "success": len(success_rows),
            "failed": len(failure_rows),
            "skipped": len(skipped_rows),
        },
    }
    write_json(paths["metadata"], metadata)

    print(
        f"social_game={args.social_game} total={total} success={len(success_rows)} "
        f"failed={len(failure_rows)} skipped={len(skipped_rows)}"
    )
    print(f"success_output={paths['success']}")
    print(f"failure_output={paths['failure']}")
    print(f"metadata_output={paths['metadata']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        social_game_config(args.social_game)
    except ValueError as exc:
        parser.error(str(exc))
    return run_transform(args)


if __name__ == "__main__":
    raise SystemExit(main())
