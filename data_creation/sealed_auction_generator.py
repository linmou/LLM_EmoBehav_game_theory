import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Optional

DEFAULT_MODEL = "gemini-2.5-flash"
REQUIRED_CHOICE_KEYS = ("devote_low", "devote_medium", "devote_high")
REQUIRED_RECORD_KEYS = (
    "scenario",
    "description",
    "participants",
    "behavior_choices",
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


def load_persona_jobs(path: Path) -> list[str]:
    jobs: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line).get("item")
            if item:
                jobs.append(item)
    return jobs


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


def read_processed_jobs(path: Path) -> set[str]:
    processed: set[str] = set()
    for record in load_existing_records(path):
        job = record.get("job")
        if job:
            processed.add(job)
    return processed


def filter_unprocessed_jobs(jobs: Iterable[str], processed: set[str]) -> list[str]:
    return [job for job in jobs if job not in processed]


def normalize_participants(job: str) -> list[dict]:
    you_name = f"You ({job})" if job else "You"
    others = [
        {"name": f"Other {job} counterpart {idx}"} if job else {"name": f"Other bidder {idx}"}
        for idx in range(1, 4)
    ]
    return [{"name": you_name}] + others


def build_prompt(job: str, fewshot: list[dict]) -> str:
    fewshot_block = f"Fewshot examples (for style only):\n{json.dumps(fewshot, indent=2)}" if fewshot else ""
    return f"""
You are writing job-themed sealed auction scenarios. Avoid any wording that suggests a game or competition.

Role focus: {job}

{fewshot_block}

Write a new scenario as a JSON object with exactly these keys:
- scenario: short title
- description: 3-5 sentences, reflecting the job situation , no negotiation
- participants: list with exactly four dict entries, each with a "name" field:
  - {{"name": "You (<job>)"}}
  - {{"name": "Other <job> counterpart 1"}}
  - {{"name": "Other <job> counterpart 2"}}
  - {{"name": "Other <job> counterpart 3"}}
- behavior_choices: dict with exactly three keys:
  - devote_low
  - devote_medium
  - devote_high
- game_category: "SEALED_BID_AUCTION_MULTIPARTY"
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

    if not isinstance(record["scenario"], str) or not record["scenario"].strip():
        raise ValueError("scenario must be a non-empty string")
    if not isinstance(record["description"], str) or not record["description"].strip():
        raise ValueError("description must be a non-empty string")

    participants = record["participants"]
    if not isinstance(participants, list) or len(participants) < 2:
        raise ValueError("participants must be a list with at least two entries")

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


def normalize_record(record: dict, job: str) -> dict:
    validate_record(record)
    normalized = dict(record)
    normalized["participants"] = normalize_participants(job)
    normalized["job"] = job
    normalized["game_name"] = "NoReturn_SealedBid_Auction"
    normalized["game_category"] = "SEALED_BID_AUCTION_MULTIPARTY"
    return normalized


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


def generate_one(
    client: Any,
    model: str,
    job: str,
    fewshot: Optional[list[dict]],
    temperature: float,
) -> Optional[dict]:
    prompt = build_prompt(job, fewshot)
    for attempt in range(2):
        content = request_content(client, model, prompt, temperature)
        if not content:
            continue
        try:
            data = parse_model_output(content)
            return normalize_record(data, job)
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
        description="Generate sealed auction scenarios with job-specific context."
    )
    parser.add_argument(
        "--jobs",
        type=Path,
        default=Path("data_creation/persona_jobs_all.jsonl"),
    )
    parser.add_argument(
        "--fewshot",
        type=Optional[Path],
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data_creation/scenario_creation/langgraph_creation/Sealed_Auction_all_data_samples.json"
        ),
    )
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument(
        "--max-keep",
        type=int,
        default=500,
        help="Keep only the most recent N records in --output (0 disables).",
    )
    args = parser.parse_args()

    fewshot = load_fewshot_examples(args.fewshot) if args.fewshot else None
    jobs = load_persona_jobs(args.jobs)
    existing_records = load_existing_records(args.output)
    normalized_existing = [
        normalize_record(record, str(record.get("job") or ""))
        for record in existing_records
    ]
    processed = {record.get("job") for record in normalized_existing if record.get("job")}
    pending = filter_unprocessed_jobs(jobs, processed)

    random.Random(args.seed).shuffle(pending)
    if args.num is not None:
        pending = pending[: args.num]

    if not pending:
        write_json(args.output, trim_records(normalized_existing, args.max_keep))
        return

    api_key = resolve_api_key(args.api_key)
    if not api_key:
        raise RuntimeError(
            "Missing Gemini API key. Provide --api_key or set GEMINI_API_KEY/GOOGLE_API_KEY."
        )

    from google import genai

    client = genai.Client(api_key=api_key)

    results: list[dict] = []
    done = 0
    total = len(pending)
    start = time.time()
    last_draw = 0.0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                generate_one,
                client,
                args.model,
                job,
                fewshot,
                args.temperature,
            )
            for job in pending
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
        trim_records(normalized_existing + results, args.max_keep),
    )


if __name__ == "__main__":
    main()
