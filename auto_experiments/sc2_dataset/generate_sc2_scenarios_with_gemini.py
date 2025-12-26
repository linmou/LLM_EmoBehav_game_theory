"""Generate StarCraft scenarios with Gemini 2.5 Flash from `few_shot_examples.json`."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


DEFAULT_MODEL_NAME = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 1.0


def load_few_shot_examples(path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict JSON in {path}")

    instruction = raw.get("instruction")
    examples = raw.get("examples")

    if not isinstance(instruction, list) or not all(
        isinstance(x, str) and x.strip() for x in instruction
    ):
        raise ValueError(f"Invalid or missing 'instruction' list in {path}")

    if not isinstance(examples, list) or not all(isinstance(x, dict) for x in examples):
        raise ValueError(f"Invalid or missing 'examples' list in {path}")

    return [x.strip() for x in instruction], list(examples)


def _extract_json(text: str) -> str:
    s = (text or "").strip()
    if not s:
        raise ValueError("Empty model response")
    if s.startswith("{") or s.startswith("["):
        return s

    start_obj = s.find("{")
    end_obj = s.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        return s[start_obj : end_obj + 1]

    start_arr = s.find("[")
    end_arr = s.rfind("]")
    if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        return s[start_arr : end_arr + 1]

    return s


def parse_scenarios(text: str) -> List[Dict[str, Any]]:
    payload = json.loads(_extract_json(text))
    if isinstance(payload, list):
        scenarios = payload
    elif isinstance(payload, dict) and isinstance(payload.get("scenarios"), list):
        scenarios = payload["scenarios"]
    else:
        raise ValueError("Expected JSON list or {'scenarios': [...]} from model")

    if not all(isinstance(x, dict) for x in scenarios):
        raise ValueError("Each scenario must be a JSON object")
    return list(scenarios)

def _validate_scenarios_format(scenarios: Sequence[Dict[str, Any]]) -> None:
    for i, sc in enumerate(scenarios):
        desc = sc.get("description")
        if not (isinstance(desc, str) and desc.strip()):
            raise ValueError(f"Scenario {i} missing/empty 'description'")

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
            raise ValueError(f"Scenario {i} has invalid 'participants' (need 2 names)")

        bc = sc.get("behavior_choices")
        if not isinstance(bc, dict):
            raise ValueError(f"Scenario {i} missing 'behavior_choices'")
        for k in ("devote_none", "devote_low", "devote_high"):
            v = bc.get(k)
            if not (isinstance(v, str) and v.strip()):
                raise ValueError(f"Scenario {i} missing/empty behavior_choices.{k}")


def _build_prompt(instruction: Sequence[str], examples: Sequence[Dict[str, Any]], n: int) -> str:
    instr = "\n".join(f"- {line}" for line in instruction)
    return (
        "You generate StarCraft scenarios.\n\n"
        "Rules:\n"
        f"{instr}\n\n"
        f"Return exactly {n} new scenarios.\n"
        "Return ONLY valid JSON with schema: {\"scenarios\": [ ... ]}.\n"
        "Each scenario must match the examples exactly: keys description, participants, behavior_choices.\n\n"
        "Few-shot examples:\n"
        f"{json.dumps(list(examples), ensure_ascii=False)}"
    )


def generate_scenarios(
    *,
    model: Any,
    instruction: Sequence[str],
    examples: Sequence[Dict[str, Any]],
    n: int,
) -> List[Dict[str, Any]]:
    prompt = _build_prompt(instruction, examples, n)
    response = model.generate_content(
        [{"role": "user", "parts": [prompt]}],
        generation_config={
            "temperature": DEFAULT_TEMPERATURE,
            "response_mime_type": "application/json",
        },
    )
    text = getattr(response, "text", "") or ""
    scenarios = parse_scenarios(text)

    _validate_scenarios_format(scenarios)

    forbidden = ("sealed-auction", "sealed auction")
    for idx, sc in enumerate(scenarios):
        desc = str(sc.get("description", ""))
        lower = desc.lower()
        if any(term in lower for term in forbidden):
            raise ValueError(
                f"Scenario {idx} description contains forbidden term; regenerate."
            )

    return scenarios


def write_scenarios(path: Path, scenarios: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(scenarios), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def init_gemini_model(*, api_key: str, model_name: str = DEFAULT_MODEL_NAME) -> Any:
    import google.generativeai as genai  # type: ignore[import-not-found]

    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY env var")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def generate_scenarios_parallel(
    *,
    api_key: str,
    model_name: str,
    instruction: Sequence[str],
    examples: Sequence[Dict[str, Any]],
    n: int,
    batch_size: int,
    concurrency: int,
    max_retries: int,
    show_progress: bool,
) -> List[Dict[str, Any]]:
    if n <= 0:
        return []
    if batch_size <= 0:
        raise ValueError("--batch_size must be > 0")
    if concurrency <= 0:
        raise ValueError("--concurrency must be > 0")
    if max_retries < 0:
        raise ValueError("--max_retries must be >= 0")

    sizes: List[int] = []
    remaining = n
    while remaining > 0:
        cur = min(batch_size, remaining)
        sizes.append(cur)
        remaining -= cur

    def _worker(batch_n: int) -> List[Dict[str, Any]]:
        last_err: Exception | None = None
        for _ in range(max_retries + 1):
            try:
                model = init_gemini_model(api_key=api_key, model_name=model_name)
                scenarios = generate_scenarios(
                    model=model, instruction=instruction, examples=examples, n=batch_n
                )
                if len(scenarios) != batch_n:
                    raise ValueError(
                        f"Model returned {len(scenarios)} scenarios, expected {batch_n}"
                    )
                return scenarios
            except Exception as exc:
                last_err = exc
        assert last_err is not None
        raise last_err

    out: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(_worker, batch_n) for batch_n in sizes]
        total = len(futures)
        done = 0
        for fut in as_completed(futures):
            out.extend(fut.result())
            done += 1
            if show_progress:
                width = 30
                filled = int(width * done / total) if total else width
                bar = "#" * filled + "-" * (width - filled)
                print(f"\r[{bar}] {done}/{total} batches", end="", flush=True)
        if show_progress:
            print(flush=True)

    if len(out) != n:
        raise ValueError(f"Generated {len(out)} scenarios, expected {n}")
    return out


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--few_shot", type=Path, default=Path(__file__).with_name("few_shot_examples.json"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max_retries", type=int, default=2)
    p.add_argument("--no_progress", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)

    api_key = os.environ.get("GEMINI_API_KEY", "")
    instruction, examples = load_few_shot_examples(args.few_shot)
    scenarios = generate_scenarios_parallel(
        api_key=api_key,
        model_name=args.model,
        instruction=instruction,
        examples=examples,
        n=args.n,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        show_progress=not args.no_progress,
    )
    write_scenarios(args.out, scenarios)


if __name__ == "__main__":
    main()
