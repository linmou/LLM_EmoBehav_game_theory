#!/usr/bin/env python3
# Purpose: transform curated social game rows into loadable game scenario datasets with resume and audit artifacts.

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from games.beauty_contest import BeautyContestScenario


SUPPORTED_SOCIAL_GAMES = {
    "beauty_contest": {
        "target_game_name": "Beauty_Contest",
        "success_file": "beauty_contest.success.json",
        "failure_file": "beauty_contest.failures.jsonl",
        "skip_file": "beauty_contest.skipped.jsonl",
    }
}
DEFAULT_RUBRIC_PATH = Path(__file__).resolve().parent.parent / "transform_rubrics.md"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="transform_social_game_cases")
    parser.add_argument("--social-game", required=True)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--few-shot-path", required=True)
    parser.add_argument("--rubric-path", default=str(DEFAULT_RUBRIC_PATH))
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--max-retries", type=int, default=0)
    return parser


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n",
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


def social_game_config(social_game: str) -> dict[str, str]:
    cfg = SUPPORTED_SOCIAL_GAMES.get(social_game)
    if cfg is None:
        raise ValueError(
            f"Unsupported social game: {social_game}. Supported values: {', '.join(sorted(SUPPORTED_SOCIAL_GAMES))}"
        )
    return cfg


def load_prompt_pack(
    social_game: str,
    rubric_path: Path,
    few_shot_path: Path,
) -> dict[str, Any]:
    cfg = social_game_config(social_game)
    if not rubric_path.exists():
        raise FileNotFoundError(f"Rubric file not found: {rubric_path}")
    if not few_shot_path.exists():
        raise FileNotFoundError(f"Few-shot file not found: {few_shot_path}")
    few_shot_examples = read_json(few_shot_path)
    if not isinstance(few_shot_examples, list):
        raise ValueError(f"Few-shot file must contain a JSON list: {few_shot_path}")
    return {
        "social_game": social_game,
        "rubric_path": rubric_path,
        "rubric_text": rubric_path.read_text(encoding="utf-8").strip(),
        "few_shot_path": few_shot_path,
        "few_shot_examples": few_shot_examples,
        "target_game_name": cfg["target_game_name"],
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


def validate_loadable_with_game_contract(row: dict[str, Any]) -> None:
    scenario_payload = dict(row)
    scenario_payload.setdefault("payoff_matrix", {})
    BeautyContestScenario(**scenario_payload)


def transform_source_row(
    *,
    source_row: dict[str, Any],
    prompt_pack: dict[str, Any],
    model_name: str,
    max_retries: int = 0,
) -> dict[str, Any]:
    system_prompt = build_system_prompt(prompt_pack)
    user_prompt = (
        "Transform the following curated social-game case into one loadable Beauty Contest scenario.\n"
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
                temperature=0.0,
            )
            payload = parse_json_text(extract_response_text(response))
            payload.setdefault("provenance", {})
            payload["provenance"]["id"] = source_row["id"]
            payload["provenance"]["source_game_id"] = source_row["source"]["game_id"]
            payload["provenance"]["source_dataset"] = source_row["source"].get("dataset")
            payload["provenance"]["source_line_number"] = source_row["source"].get("line_number")
            validate_loadable_with_game_contract(payload)
            return payload
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    assert last_error is not None
    raise last_error


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

    prompt_pack = load_prompt_pack(
        social_game=args.social_game,
        rubric_path=Path(args.rubric_path),
        few_shot_path=Path(args.few_shot_path),
    )

    existing_success_rows = [] if args.rerun else load_existing_success_rows(paths["success"])
    existing_failure_rows = [] if args.rerun else load_existing_jsonl(paths["failure"])
    existing_skip_rows = [] if args.rerun else load_existing_jsonl(paths["skip"])
    existing_metadata = None if args.rerun or not paths["metadata"].exists() else read_json(paths["metadata"])

    success_rows = list(existing_success_rows)
    failure_rows = list(existing_failure_rows)
    skipped_rows = list(existing_skip_rows)
    completed_identities = completed_identities_from_outputs(success_rows, failure_rows, existing_metadata)

    source_rows = load_jsonl(input_path, limit=args.limit)
    total = len(source_rows)
    done = 0

    # Keep the concurrency primitive visible for later extension, while preserving deterministic execution now.
    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)):
        for source_row in source_rows:
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
                transformed = transform_source_row(
                    source_row=source_row,
                    prompt_pack=prompt_pack,
                    model_name=args.model,
                    max_retries=args.max_retries,
                )
                success_rows.append(transformed)
                completed_identities.add(identity_key)
            except Exception as exc:  # noqa: BLE001
                failure_rows.append(build_failure_record(source_row=source_row, stage="transform", error=exc))
                completed_identities.add(identity_key)
            done += 1
            print(render_progress(done, total))

    write_json(paths["success"], success_rows)
    write_jsonl(paths["failure"], failure_rows)
    write_jsonl(paths["skip"], skipped_rows)

    metadata = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "social_game": args.social_game,
        "input_path": str(input_path.resolve()),
        "success_output_path": str(paths["success"].resolve()),
        "failure_output_path": str(paths["failure"].resolve()),
        "skip_output_path": str(paths["skip"].resolve()),
        "model_name": args.model,
        "rubric_path": str(Path(args.rubric_path).resolve()),
        "few_shot_path": str(Path(args.few_shot_path).resolve()),
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
