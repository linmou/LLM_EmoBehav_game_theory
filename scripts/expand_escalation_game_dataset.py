#!/usr/bin/env python3
"""Expand data/sc2/escalation_game.json to 10x size using GPT-5 and OAI_CONFIG."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

from openai import OpenAI

from api_configs import OAI_CONFIG


SYSTEM_PROMPT = (
    "You are generating high-quality StarCraft II escalation scenarios. "
    "Follow the JSON schema exactly and avoid duplicates or near-duplicates of the provided descriptions. "
    "Preserve a balance of Protoss, Terran, and Zerg perspectives."
)


def build_openai_client(
    openai_cls: Callable[..., Any] = OpenAI,
    config: Mapping[str, str] | None = None,
) -> Any:
    """Instantiate an OpenAI client with the provided config (defaults to OAI_CONFIG)."""
    cfg = dict(config or OAI_CONFIG)
    return openai_cls(**cfg)


def _load_dataset(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of scenarios in {path}")
    return data


def _build_prompt(samples: Sequence[Dict[str, Any]], needed: int) -> str:
    sample_block = json.dumps(list(samples), indent=2)
    return (
        f"Expand the SC2 escalation dataset by creating {needed} new scenarios.\n"
        "Use the exact structure shown in the samples: description, you_play_as, behaviour_decisions "
        "(escalate/withdraw lists), players (player_1/player_2 with race, role, economy, army, advantage), "
        "options (ids 1 and 2 with escalation_strength 2 and -2), and all_options (ids 1-4 with strengths [2,1,-1,-2]).\n"
        "Keep text concise, avoid repeating the sample descriptions, and vary tech, timing, and map states.\n"
        "Return a JSON object with a 'scenarios' key containing a list of the new scenario objects.\n"
        f"Sample scenarios for reference:\n{sample_block}"
    )


def _parse_generated_scenarios(content: str) -> List[Dict[str, Any]]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        snippet = content[:200]
        raise ValueError(f"Invalid JSON from model: {snippet}") from exc
    scenarios = parsed.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("Expected JSON object with 'scenarios' list from GPT-5 response")
    return scenarios


def _merge_datasets(
    existing: Iterable[Dict[str, Any]],
    generated: Iterable[Dict[str, Any]],
    target_total: int,
    strict: bool = True,
) -> List[Dict[str, Any]]:
    combined: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for scenario in existing:
        desc = scenario.get("description")
        if isinstance(desc, str):
            seen.add(desc.strip())
        combined.append(scenario)

    for scenario in generated:
        if len(combined) >= target_total:
            break
        desc = scenario.get("description")
        if not isinstance(desc, str):
            continue
        key = desc.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        combined.append(scenario)

    if strict and len(combined) < target_total:
        raise ValueError(
            f"Only collected {len(combined)} scenarios; need {target_total} after expansion"
        )
    return combined


def _render_progress_bar(current: int, total: int, width: int = 40) -> str:
    if total <= 0:
        total = 1
    ratio = max(0.0, min(1.0, current / total))
    filled = int(width * ratio)
    bar = "#" * filled + "." * (width - filled)
    pct = int(ratio * 100)
    return f"[{bar}] {current}/{total} ({pct}%)"


def _write_dataset(path: Path, data: List[Dict[str, Any]]) -> None:
    """Persist the current dataset snapshot to disk."""
    path.write_text(json.dumps(data, indent=2))


def expand_escalation_game_dataset(
    input_path: Path | str = Path("data/sc2/escalation_game.json"),
    output_path: Path | str | None = None,
    client: Any | None = None,
    model: str = "gpt-5.1",
    target_total: int | None = None,
    concurrency: int = 1,
    batch_size: int | None = None,
    max_completion_tokens: int | None = None,
    max_retries: int = 2,
    max_rounds: int = 30,
    show_progress: bool = False,
) -> List[Dict[str, Any]]:
    in_path = Path(input_path)
    out_path = Path(output_path) if output_path is not None else in_path

    base_dataset = _load_dataset(in_path)
    base_len = len(base_dataset)
    if base_len == 0:
        raise ValueError(f"{in_path} is empty; need seed scenarios to expand")

    if target_total is None:
        target_total = base_len * 10
    if target_total <= base_len:
        raise ValueError(
            f"target_total={target_total} must be greater than current dataset size {base_len}"
        )
    needed = target_total - base_len

    llm_client = client or build_openai_client()
    samples = base_dataset[:4]

    def _request_new(count: int) -> List[Dict[str, Any]]:
        prompt = _build_prompt(samples, count)
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        if max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = max_completion_tokens

        last_err: Exception | None = None
        for _ in range(max(1, max_retries)):
            completion = llm_client.chat.completions.create(**kwargs)
            content = completion.choices[0].message.content
            if not isinstance(content, str):
                last_err = ValueError("Unexpected completion format: missing message content")
                continue
            try:
                return _parse_generated_scenarios(content)
            except ValueError as exc:
                last_err = exc
                continue
        if last_err:
            raise last_err
        raise RuntimeError("Failed to obtain scenarios from model")

    dataset: List[Dict[str, Any]] = list(base_dataset)
    batch = max(1, batch_size or needed)
    workers = max(1, concurrency)

    rounds = 0
    last_error: Exception | None = None
    while len(dataset) < target_total and rounds < max_rounds:
        rounds += 1
        remaining = target_total - len(dataset)
        batch = min(batch, remaining)
        task_count = min(workers, (remaining + batch - 1) // batch)

        errors: List[Exception] = []
        with ThreadPoolExecutor(max_workers=task_count) as executor:
            futures = [executor.submit(_request_new, batch) for _ in range(task_count)]
            for fut in as_completed(futures):
                try:
                    generated = fut.result()
                except Exception as exc:
                    errors.append(exc)
                    continue
                dataset = _merge_datasets(
                    dataset, generated, target_total=target_total, strict=False
                )
                if len(dataset) >= target_total:
                    break

        if show_progress:
            print(
                _render_progress_bar(len(dataset), target_total),
                end="\r" if len(dataset) < target_total else "\n",
                flush=True,
            )

        # Persist checkpoint after each round so partial progress is not lost if interrupted.
        out_path.write_text(json.dumps(dataset, indent=2))

        if errors:
            last_error = errors[-1]

    if len(dataset) < target_total:
        if last_error:
            raise last_error
        raise RuntimeError(
            f"Only collected {len(dataset)} scenarios after {rounds} rounds; "
            f"need {target_total} after expansion"
        )

    dataset = dataset[:target_total]
    out_path.write_text(json.dumps(dataset, indent=2))
    return dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand SC2 escalation dataset to 10x size using GPT-5 and OAI_CONFIG."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/sc2/escalation_game.json"),
        help="Path to the base escalation_game.json file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the expanded dataset (defaults to overwriting input).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.1",
        help="GPT-5 model name to use for generation.",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=None,
        help="Total number of scenarios desired in the output dataset (must be > current count).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Number of parallel completion requests to issue.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="How many new scenarios to request per completion call.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=6000,
        help="Per-call completion token cap (set lower if the model enforces limits).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Per-call retry attempts when the model returns invalid JSON.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=30,
        help="Maximum batch rounds before aborting expansion.",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable progress bar output during expansion.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    expand_escalation_game_dataset(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        target_total=args.target_total,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        max_completion_tokens=args.max_completion_tokens,
        max_retries=args.max_retries,
        max_rounds=args.max_rounds,
        show_progress=not args.no_progress_bar,
    )


if __name__ == "__main__":
    main()
