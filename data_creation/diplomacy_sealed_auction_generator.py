import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

DEFAULT_MODEL = "gemini-2.5-flash"
REQUIRED_CHOICE_KEYS = ("devote_low", "devote_medium", "devote_high")
REQUIRED_RECORD_KEYS = (
    "scenario",
    "description",
    "participants",
    "behavior_choices",
    "game_category",
    "game_name",
)


def render_progress(done: int, total: int, width: int = 24) -> str:
    total = max(total, 1)
    done = min(max(done, 0), total)
    ratio = done / total
    filled = int(ratio * width)
    bar = "=" * filled + "-" * (width - filled)
    pct = int(ratio * 100)
    return f"[{bar}] {pct:3d}% {done}/{total}"


def _is_tty() -> bool:
    try:
        return sys.stderr.isatty()
    except Exception:
        return False


def load_fewshot_examples(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Fewshot file must contain a JSON list: {path}")
    for item in data:
        if not isinstance(item, dict):
            raise ValueError(f"Fewshot file must contain objects only: {path}")
    return data


def load_existing_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Output file must contain a JSON list: {path}")
    return [item for item in data if isinstance(item, dict)]


def build_prompt(
    fewshot: list[dict], avoid_titles: list[str], seed: Optional[int] = None
) -> str:
    fewshot_block = list(fewshot)
    if seed is not None:
        random.Random(seed).shuffle(fewshot_block)
    fewshot_block = json.dumps(fewshot_block, indent=2)
    avoid_block = ""
    if avoid_titles:
        avoid_block = "\nAvoid reusing these scenario titles:\n- " + "\n- ".join(
            avoid_titles
        )
    return f"""
You are writing Diplomacy-themed sealed auction scenarios. Avoid any wording that suggests a game or competition.

Fewshot examples (for style only):
{fewshot_block}
{avoid_block}

Write a new scenario as a JSON object with exactly these keys:
- game_category: "SEALED_BID_AUCTION_MULTIPARTY"
- scenario: short title
- description: 3-5 sentences, Diplomacy setting, no negotiation
- participants: list with 3-5 dict entries, each with a "name" field
- behavior_choices: dict with exactly three keys:
  - devote_low
  - devote_medium
  - devote_high
- game_name: "NoReturn_SealedBid_Auction"

Each behavior choice should be a concrete action or resource commitment.
Do not mention "game", "competition", "contest" or any emotional words.
Return only the JSON object.
""".strip()


def parse_model_output(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


def request_content(
    client: Any,
    model: str,
    prompt: str,
    temperature: float,
) -> Optional[str]:
    from google.genai import types

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
        ),
    )
    return getattr(response, "text", None)


def resolve_api_key(explicit_key: Optional[str]) -> Optional[str]:
    if explicit_key:
        return explicit_key
    api_key = None
    try:
        from api_configs import GEMINI_CONFIG  # type: ignore

        if isinstance(GEMINI_CONFIG, dict):
            api_key = GEMINI_CONFIG.get("api_key")
    except Exception:
        api_key = None
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return api_key


def validate_record(record: dict) -> dict:
    missing = [key for key in REQUIRED_RECORD_KEYS if key not in record]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    if record["game_name"] != "NoReturn_SealedBid_Auction":
        raise ValueError("game_name must be NoReturn_SealedBid_Auction")
    if record["game_category"] != "SEALED_BID_AUCTION_MULTIPARTY":
        raise ValueError("game_category must be SEALED_BID_AUCTION_MULTIPARTY")

    if not isinstance(record["scenario"], str) or not record["scenario"].strip():
        raise ValueError("scenario must be a non-empty string")
    if not isinstance(record["description"], str) or not record["description"].strip():
        raise ValueError("description must be a non-empty string")

    participants = record["participants"]
    if not isinstance(participants, list) or len(participants) < 2:
        raise ValueError("participants must be a list with at least two entries")
    for entry in participants:
        if not isinstance(entry, dict) or "name" not in entry:
            raise ValueError("each participant must be a dict with a name field")
        if not isinstance(entry["name"], str) or not entry["name"].strip():
            raise ValueError("participant name must be a non-empty string")

    behavior_choices = record["behavior_choices"]
    if not isinstance(behavior_choices, dict):
        raise ValueError("behavior_choices must be a dict")
    if set(behavior_choices.keys()) != set(REQUIRED_CHOICE_KEYS):
        raise ValueError("behavior_choices must contain devote_low/medium/high only")
    for key in REQUIRED_CHOICE_KEYS:
        value = behavior_choices[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non-empty string")

    return record


def generate_one(
    client: Any,
    model: str,
    fewshot: list[dict],
    temperature: float,
    avoid_titles: list[str],
    seed: Optional[int],
) -> Optional[dict]:
    prompt = build_prompt(fewshot, avoid_titles, seed=seed)
    for attempt in range(2):
        content = request_content(client, model, prompt, temperature)
        if not content:
            continue
        try:
            data = parse_model_output(content)
            return validate_record(data)
        except (json.JSONDecodeError, ValueError):
            if attempt == 0:
                prompt = (
                    prompt
                    + "\nYour previous response was invalid JSON. Return only valid JSON."
                )
            else:
                raise
    return None


def write_json(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=True, indent=2)
    tmp_path.replace(path)


def trim_records(records: list[dict], max_keep: int) -> list[dict]:
    if max_keep <= 0:
        return records
    if len(records) <= max_keep:
        return records
    return records[-max_keep:]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Diplomacy sealed auction scenarios."
    )
    parser.add_argument(
        "--fewshot",
        type=Path,
        default=Path("data_creation/scenario_creation/diplomacy_sealed_auction_fewshot.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data_creation/scenario_creation/langgraph_creation/Diplomacy_Sealed_Auction_all_data_samples.json"
        ),
    )
    parser.add_argument("--num", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument(
        "--max-keep",
        type=int,
        default=500,
        help="Keep only the most recent N records in --output (0 disables).",
    )
    args = parser.parse_args()

    if args.num <= 0:
        raise ValueError("--num must be a positive integer")

    fewshot = load_fewshot_examples(args.fewshot)
    existing_records = load_existing_records(args.output)
    avoid_titles = [
        record.get("scenario", "")
        for record in existing_records
        if isinstance(record.get("scenario", ""), str)
    ]

    api_key = resolve_api_key(args.api_key)
    if not api_key:
        raise RuntimeError(
            "Missing Gemini API key. Provide --api_key or set GEMINI_API_KEY/GOOGLE_API_KEY."
        )

    from google import genai

    client = genai.Client(api_key=api_key)
    base_seed = int(time.time())

    results: list[dict] = []
    done = 0
    total = args.num
    start = time.time()
    last_draw = 0.0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                generate_one,
                client,
                args.model,
                fewshot,
                args.temperature,
                avoid_titles,
                base_seed + idx,
            )
            for idx in range(args.num)
        ]
        for future in as_completed(futures):
            record = future.result()
            if record:
                results.append(record)
            done += 1
            if _is_tty():
                now = time.time()
                if done == total or (now - last_draw) > 0.1:
                    elapsed = max(now - start, 1e-6)
                    rate = done / elapsed
                    sys.stderr.write(
                        "\r" + render_progress(done, total) + f"  {rate:5.1f}/s"
                    )
                    sys.stderr.flush()
                    last_draw = now

    if _is_tty():
        sys.stderr.write("\n")

    write_json(
        args.output,
        trim_records(existing_records + results, args.max_keep),
    )


if __name__ == "__main__":
    main()
