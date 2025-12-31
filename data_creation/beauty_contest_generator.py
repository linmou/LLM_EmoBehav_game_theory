import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, List, Optional

DEFAULT_MODEL = "gemini-2.5-flash"


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


def load_persona_jobs(path: Path) -> List[str]:
    jobs: List[str] = []
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
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_existing_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        raw = handle.read().strip()
    if not raw:
        return []

    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(f"Output file must contain a JSON list: {path}")
        return [item for item in data if isinstance(item, dict)]
    except json.JSONDecodeError:
        records: list[dict] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                records.append(item)
        return records


def read_processed_jobs(path: Path) -> set[str]:
    processed: set[str] = set()
    for record in load_existing_records(path):
        job = record.get("job")
        if job:
            processed.add(job)
    return processed


def filter_unprocessed_jobs(jobs: Iterable[str], processed: set[str]) -> List[str]:
    return [job for job in jobs if job not in processed]


def normalize_participants(job: str) -> list[dict]:
    you_name = f"You ({job})" if job else "You"
    group_name = f"10 Other {job} counterparts" if job else "10 Others"
    return [{"name": you_name}, {"name": group_name}]


def normalize_behavior_choices(raw: Any) -> dict:
    if isinstance(raw, dict) and all(f"commit_{i}" in raw for i in range(4)):
        return {f"commit_{i}": str(raw[f"commit_{i}"]) for i in range(4)}

    if isinstance(raw, dict) and all(k in raw for k in ("option_low", "option_medium", "option_high")):
        return {
            "commit_0": "Make no commitment and keep the current plan unchanged.",
            "commit_1": str(raw["option_low"]),
            "commit_2": str(raw["option_medium"]),
            "commit_3": str(raw["option_high"]),
        }

    if isinstance(raw, list):
        items = [str(item) for item in raw if item is not None]
        while len(items) < 4:
            items.append("Make a moderate commitment consistent with the team's baseline.")
        items = items[:4]
        return {f"commit_{i}": items[i] for i in range(4)}

    raise ValueError("behavior_choices must be a dict or list")


def normalize_record(record: dict, job: str) -> dict:
    normalized = dict(record)
    normalized["participants"] = normalize_participants(job)
    normalized["behavior_choices"] = normalize_behavior_choices(record.get("behavior_choices"))
    normalized["job"] = job
    normalized["game_name"] = "Beauty_Contest"
    return normalized


def build_prompt(job: str, fewshot: list[dict]) -> str:
    fewshot_block = json.dumps(fewshot, indent=2)
    return f"""
You are writing realistic workplace or daily-life scenarios. Avoid any wording that suggests a game or competition.

Role focus: {job}

Fewshot examples (for style only):
{fewshot_block}

Write a new scenario as a JSON object with exactly these keys:
- scenario: short title
- description: 3-5 sentences, everyday context
- participants: list with exactly two dict entries, each with a "name" field:
  - {{"name": "You (<job>)"}}
  - {{"name": "<N> other <job plural/similar group>"}}
- behavior_choices: dict with exactly four keys:
  - commit_0
  - commit_1
  - commit_2
  - commit_3

Each behavior choice should be a concrete action or numeric commitment. Do not mention "game", "competition", "contest" or any emotional words.
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


def generate_one(
    client: Any,
    model: str,
    job: str,
    fewshot: list[dict],
    temperature: float,
) -> Optional[dict]:
    prompt = build_prompt(job, fewshot)

    from google.genai import types

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
        ),
    )
    content = getattr(response, "text", None)
    if not content:
        return None
    data = parse_model_output(content)
    return normalize_record(data, job)


def write_json(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=True, indent=2)
    tmp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Beauty Contest scenarios.")
    parser.add_argument(
        "--jobs",
        type=Path,
        default=Path("data_creation/persona_jobs_all.jsonl"),
    )
    parser.add_argument(
        "--fewshot",
        type=Path,
        default=Path("data_creation/scenario_creation/beauty_contest_fewshot.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_creation/scenario_creation/beauty_contest_generated.json"),
    )
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--api_key", type=str, default=None)
    args = parser.parse_args()

    fewshot = load_fewshot_examples(args.fewshot)
    jobs = load_persona_jobs(args.jobs)
    existing_records = load_existing_records(args.output)
    normalized_existing = [
        normalize_record(record, str(record.get("job") or "")) for record in existing_records
    ]
    processed = {record.get("job") for record in normalized_existing if record.get("job")}
    pending = filter_unprocessed_jobs(jobs, processed)

    random.Random(args.seed).shuffle(pending)
    if args.num is not None:
        pending = pending[: args.num]

    if not pending:
        write_json(args.output, normalized_existing)
        return

    api_key = args.api_key
    if not api_key:
        try:
            from api_configs import GEMINI_CONFIG  # type: ignore

            api_key = GEMINI_CONFIG.get("api_key")  # type: ignore[assignment]
        except Exception:
            api_key = None
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing Gemini API key. Provide --api_key, or set GEMINI_CONFIG['api_key'] "
            "in api_configs.py, or set GOOGLE_API_KEY / GEMINI_API_KEY."
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

    write_json(args.output, normalized_existing + results)


if __name__ == "__main__":
    main()
