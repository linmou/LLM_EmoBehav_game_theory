"""Generate StarCraft scenarios with Gemini, grounded in intent-map JSONL slices.

For each generated scenario, we read one JSONL line from the intent-map dataset and
attach `image_path` in the output so downstream multimodal pipelines can load the
corresponding frame.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from auto_experiments.sc2_dataset.generate_sc2_scenarios_with_gemini import (  # reuse parsing + few-shot loader
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    _extract_json,
    init_gemini_model,
    load_existing_scenarios,
    load_few_shot_examples,
    write_scenarios,
)

from games.game_configs import get_game_config


IntentMapRecord = Dict[str, Any]


def scenario_type_for_intent_category(intent_category: str) -> str:
    m = {
        "air": "AirControl",
        "drop": "Airdrop",
        "base": "BaseRace",
        "gold": "GoldMineralCompetition",
    }
    try:
        return m[intent_category]
    except KeyError as exc:
        raise ValueError(f"Unknown intent_category: {intent_category!r}") from exc


def iter_intent_map_records(path: Path) -> Iterable[IntentMapRecord]:
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            raw = json.loads(s)
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_num} must be a JSON object")

            map_image = raw.get("map_image")
            description = raw.get("description")
            intent_category = raw.get("intent_category")
            if not (
                isinstance(map_image, str)
                and map_image.strip()
                and isinstance(description, str)
                and description.strip()
                and isinstance(intent_category, str)
                and intent_category.strip()
            ):
                raise ValueError(
                    f"{path}:{line_num} missing required fields: map_image/description/intent_category"
                )

            meta = raw.get("meta")
            meta_dict: Dict[str, Any] = meta if isinstance(meta, dict) else {}
            yield {
                "map_image": map_image.strip(),
                "description": description.strip(),
                "intent_category": intent_category.strip(),
                "metadata": {
                    "source_jsonl": str(path),
                    "source_line_num": line_num,
                    **meta_dict,
                },
            }


def _parse_one_scenario(text: str) -> Dict[str, Any]:
    payload = json.loads(_extract_json(text))
    if isinstance(payload, dict) and isinstance(payload.get("scenarios"), list):
        scenarios = payload["scenarios"]
    elif isinstance(payload, list):
        scenarios = payload
    elif isinstance(payload, dict):
        scenarios = [payload]
    else:
        raise ValueError("Expected JSON object, JSON list, or {'scenarios':[...]} from model")

    if len(scenarios) != 1 or not isinstance(scenarios[0], dict):
        raise ValueError("Expected exactly 1 scenario object from model")
    return scenarios[0]


def _behavior_choice_keys_for_game_name(game_name: str) -> tuple[str, ...]:
    scenario_cls = get_game_config(game_name)["scenario_class"]
    example = scenario_cls.example()
    if not isinstance(example, dict):
        raise ValueError(f"Invalid scenario example for game {game_name!r}: expected dict")
    behavior_choices = example.get("behavior_choices")
    if not isinstance(behavior_choices, dict) or not behavior_choices:
        raise ValueError(
            f"Invalid scenario example for game {game_name!r}: missing behavior_choices dict"
        )
    return tuple(str(k) for k in behavior_choices.keys())


def _validate_scenario_format(
    sc: Dict[str, Any], *, behavior_choice_keys: Sequence[str]
) -> None:
    scenario = sc.get("scenario")
    if not (isinstance(scenario, str) and scenario.strip()):
        raise ValueError("Missing/empty 'scenario'")

    desc = sc.get("description")
    if not (isinstance(desc, str) and desc.strip()):
        raise ValueError("Missing/empty 'description'")

    participants = sc.get("participants")
    if not (
        isinstance(participants, list)
        and len(participants) == 2
        and all(
            isinstance(p, dict)
            and isinstance(p.get("name"), str)
            and p.get("name", "").strip()
            for p in participants
        )
    ):
        raise ValueError("Invalid 'participants' (need 2 names)")

    bc = sc.get("behavior_choices")
    if not isinstance(bc, dict):
        raise ValueError("Missing 'behavior_choices'")
    for k in behavior_choice_keys:
        v = bc.get(k)
        if not (isinstance(v, str) and v.strip()):
            raise ValueError(f"Missing/empty behavior_choices.{k}")


def _build_prompt(
    instruction: Sequence[str],
    examples: Sequence[Dict[str, Any]],
    record: IntentMapRecord,
    *,
    game_name: str,
) -> str:
    instr = "\n".join(f"- {line}" for line in instruction)
    scenario_type = scenario_type_for_intent_category(record["intent_category"])
    behavior_keys = _behavior_choice_keys_for_game_name(game_name)
    schema_example = get_game_config(game_name)["scenario_class"].example()
    return (
        "You generate StarCraft scenarios as game-theory cases.\n\n"
        "Rules:\n"
        f"{instr}\n\n"
        "Return exactly 1 new scenario.\n"
        "Return ONLY valid JSON for a single scenario object with keys:\n"
        "scenario, description, participants, behavior_choices.\n"
        f"GameName MUST be exactly: {game_name}\n"
        f"behavior_choices keys MUST be: {', '.join(behavior_keys)}\n"
        f"ScenarioType MUST be exactly: {scenario_type}\n\n"
        "Ground the scenario in this map slice:\n"
        f"- intent_category: {record['intent_category']}\n"
        f"- observation: {record['description']}\n"
        f"- map_image_path: {record['map_image']}\n\n"
        "Game schema example (follow this shape):\n"
        f"{json.dumps(schema_example, ensure_ascii=False)}\n\n"
        "Few-shot examples:\n"
        f"{json.dumps(list(examples), ensure_ascii=False)}"
    )


def generate_one_scenario(
    *,
    model: Any,
    instruction: Sequence[str],
    examples: Sequence[Dict[str, Any]],
    record: IntentMapRecord,
    game_name: str,
) -> Dict[str, Any]:
    behavior_choice_keys = _behavior_choice_keys_for_game_name(game_name)
    prompt = _build_prompt(instruction, examples, record, game_name=game_name)
    response = model.generate_content(
        [{"role": "user", "parts": [prompt]}],
        generation_config={
            "temperature": DEFAULT_TEMPERATURE,
            "response_mime_type": "application/json",
        },
    )
    text = getattr(response, "text", "") or ""
    scenario = _parse_one_scenario(text)

    forbidden = ("sealed-auction", "sealed auction")
    if any(term in str(scenario.get("description", "")).lower() for term in forbidden):
        raise ValueError("Scenario contains forbidden term; regenerate.")

    scenario["intent_category"] = record["intent_category"]
    scenario["image_path"] = record["map_image"]
    if "metadata" in record:
        scenario["metadata"] = record["metadata"]
    _validate_scenario_format(scenario, behavior_choice_keys=behavior_choice_keys)
    expected_suffix = "_" + scenario_type_for_intent_category(record["intent_category"])
    if not str(scenario.get("scenario", "")).endswith(expected_suffix):
        raise ValueError(f"Scenario 'scenario' must end with {expected_suffix!r}")
    return scenario


def generate_from_intent_map_parallel(
    *,
    api_key: str,
    model_name: str,
    instruction: Sequence[str],
    examples: Sequence[Dict[str, Any]],
    records: Sequence[IntentMapRecord],
    game_name: str,
    concurrency: int,
    max_retries: int,
    show_progress: bool,
) -> List[Dict[str, Any]]:
    if concurrency <= 0:
        raise ValueError("--concurrency must be > 0")
    if max_retries < 0:
        raise ValueError("--max_retries must be >= 0")

    def _worker(rec: IntentMapRecord) -> Dict[str, Any]:
        last_err: Exception | None = None
        for _ in range(max_retries + 1):
            try:
                model = init_gemini_model(api_key=api_key, model_name=model_name)
                return generate_one_scenario(
                    model=model,
                    instruction=instruction,
                    examples=examples,
                    record=rec,
                    game_name=game_name,
                )
            except Exception as exc:
                last_err = exc
        assert last_err is not None
        raise last_err

    out: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(_worker, rec) for rec in records]
        total = len(futures)
        done = 0
        for fut in as_completed(futures):
            out.append(fut.result())
            done += 1
            if show_progress:
                width = 30
                filled = int(width * done / total) if total else width
                bar = "#" * filled + "-" * (width - filled)
                print(f"\r[{bar}] {done}/{total} items", end="", flush=True)
        if show_progress:
            print(flush=True)

    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--few_shot",
        type=Path,
        default=Path(__file__).with_name("few_shot_examples.json"),
    )
    p.add_argument(
        "--intent_map_jsonl",
        type=Path,
        default=Path("datasets/intent_map_dataset_air72_base96p6_drop94p6_gold50.jsonl"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "data_creation/scenario_creation/langgraph_creation/SC2_Sealed_Auction_all_data_samples.json"
        ),
    )
    p.add_argument("--game_name", type=str, default="Sealed_Auction")
    p.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of scenarios to generate (1 per JSONL line).",
    )
    p.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many JSONL records before generating.",
    )
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--max_retries", type=int, default=2)
    p.add_argument("--no_progress", action="store_true")
    p.add_argument(
        "--resume",
        action="store_true",
        help="If --out exists, append until total --n scenarios; JSONL offset advances accordingly.",
    )
    return p


def _confirm_overwrite(path: Path, *, input_fn=input) -> bool:
    if not path.exists():
        return True
    ans = (input_fn(f"Output file '{path}' already exists. Replace it? [y/N] ") or "").strip()
    return ans.lower() in {"y", "yes"}


def main(argv: Sequence[str] | None = None) -> None:
    p = build_arg_parser()
    args = p.parse_args(list(argv) if argv is not None else None)

    api_key = os.environ.get("GEMINI_API_KEY", "")
    instruction, examples = load_few_shot_examples(args.few_shot)
    behavior_choice_keys = _behavior_choice_keys_for_game_name(args.game_name)
    for idx, ex in enumerate(examples):
        bc = ex.get("behavior_choices")
        if not isinstance(bc, dict):
            raise ValueError(f"Few-shot example {idx} missing behavior_choices dict")
        for k in behavior_choice_keys:
            v = bc.get(k)
            if not (isinstance(v, str) and v.strip()):
                raise ValueError(f"Few-shot example {idx} missing/empty behavior_choices.{k}")

    existing: List[Dict[str, Any]] = []
    if args.resume and args.out.exists():
        existing = load_existing_scenarios(args.out)
    elif args.out.exists() and not _confirm_overwrite(args.out):
        raise SystemExit(1)

    remaining = args.n - len(existing)
    if remaining <= 0:
        write_scenarios(args.out, existing[: args.n])
        return

    start = args.offset + len(existing)
    if start < 0:
        raise ValueError("--offset must be >= 0")

    records: List[IntentMapRecord] = []
    for idx, rec in enumerate(iter_intent_map_records(args.intent_map_jsonl)):
        if idx < start:
            continue
        records.append(rec)
        if len(records) >= remaining:
            break

    if len(records) != remaining:
        raise RuntimeError(
            f"Not enough intent-map records: need {remaining} starting at offset {start}, got {len(records)}"
        )

    new_items = generate_from_intent_map_parallel(
        api_key=api_key,
        model_name=args.model,
        instruction=instruction,
        examples=examples,
        records=records,
        game_name=args.game_name,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        show_progress=not args.no_progress,
    )
    write_scenarios(args.out, existing + new_items)


if __name__ == "__main__":
    main()
