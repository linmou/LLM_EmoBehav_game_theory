"""
scripts/transform_crowd_enVent_to_text_style_gemini.py

Purpose:
  Rewrite `data/stimulus/crowd-enVent/*.json` into a shorter, cleaner, text-style
  format similar to `data/stimulus/text/*.json`, using Gemini few-shot examples.

Input:
  - crowd-enVent emotion files: list[str] per emotion
  - few-shot pairs JSONL: records with {"emotion","crowd","text"}

Output:
  - out_dir/{emotion}.json: list[str] (same length, same order as input)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]


def _init_model(model_name: str):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import google.generativeai as genai  # type: ignore[import-not-found]

    from api_configs import GEMINI_CONFIG

    api_key = GEMINI_CONFIG.get("api_key")
    if not api_key:
        raise RuntimeError("GEMINI_CONFIG.api_key is empty; set it in api_configs.py")

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def _load_json_list(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError(f"Expected a JSON list[str] in {path}")
    return data


def _load_fewshot_pairs(path: Path) -> Dict[str, List[Dict[str, str]]]:
    by_emotion: Dict[str, List[Dict[str, str]]] = {e: [] for e in EMOTIONS}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if not isinstance(rec, dict):
            continue
        emo = rec.get("emotion")
        crowd = rec.get("crowd")
        text = rec.get("text")
        if emo in by_emotion and isinstance(crowd, str) and isinstance(text, str):
            by_emotion[emo].append({"crowd": crowd, "text": text})
    return by_emotion


def _extract_json_array(raw_text: str) -> List[str]:
    raw_text = (raw_text or "").strip()
    start = raw_text.find("[")
    end = raw_text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model output did not contain a JSON array")
    candidate = raw_text[start : end + 1]
    data = json.loads(candidate)
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError("Expected JSON list[str] from model")
    return [s.strip() for s in data]


def _rewrite_batch(
    model: Any,
    emotion: str,
    inputs: List[str],
    fewshot: List[Dict[str, str]],
) -> List[str]:
    system_instructions = (
        "You rewrite emotion stimulus prompts.\n"
        "Rewrite each input into a short, clean, single-sentence prompt in the same style as the examples.\n"
        "Rules:\n"
        "- Preserve the core situation; do not add new facts.\n"
        "- Prefer second-person (\"You ...\") or generic (\"Someone ...\"). Do NOT use first-person (\"I ...\").\n"
        "- Remove ellipses (\"...\") and filler; avoid overly specific personal details.\n"
        "- Keep it one sentence, plain English, properly capitalized, and end with a period.\n"
        "- Output ONLY a JSON array of strings, same length/order as inputs.\n"
    )

    payload = {
        "emotion": emotion,
        "fewshot_pairs": fewshot,
        "inputs": inputs,
    }

    response = model.generate_content(
        [
            {
                "role": "user",
                "parts": [
                    system_instructions,
                    "\n\nINPUT JSON:\n",
                    json.dumps(payload, ensure_ascii=False),
                ],
            }
        ]
    )
    return _extract_json_array(getattr(response, "text", "") or "")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transform crowd-enVent stimuli into text-style prompts using Gemini."
    )
    parser.add_argument("--in_dir", default="data/stimulus/crowd-enVent")
    parser.add_argument("--out_dir", default="data/stimulus/crowd-enVent_textlike")
    parser.add_argument(
        "--fewshot_path",
        default="data/stimulus/few_shot/crowd_enVent_to_text_pairs.v1.jsonl",
    )
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument(
        "--max_items",
        type=int,
        default=0,
        help="If >0, only rewrite the first N items per emotion (debug)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would run (no API calls, no writes)",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    fewshot_path = Path(args.fewshot_path)
    if not fewshot_path.exists():
        raise FileNotFoundError(f"Few-shot file not found: {fewshot_path}")

    fewshot_by_emotion = _load_fewshot_pairs(fewshot_path)

    if args.dry_run:
        for emo in EMOTIONS:
            src = in_dir / f"{emo}.json"
            inputs = _load_json_list(src)
            if args.max_items > 0:
                inputs = inputs[: args.max_items]
            print(f"{emo}: {len(inputs)} inputs, {len(fewshot_by_emotion[emo])} few-shot pairs")
        return

    model = _init_model(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    for emo in EMOTIONS:
        src = in_dir / f"{emo}.json"
        inputs = _load_json_list(src)
        if args.max_items > 0:
            inputs = inputs[: args.max_items]

        fewshot = fewshot_by_emotion.get(emo, [])
        if not fewshot:
            raise RuntimeError(f"No few-shot pairs found for emotion={emo} in {fewshot_path}")

        outputs = _rewrite_batch(model, emo, inputs, fewshot)
        if len(outputs) != len(inputs):
            raise RuntimeError(
                f"Length mismatch for {emo}: got {len(outputs)} outputs for {len(inputs)} inputs"
            )

        (out_dir / f"{emo}.json").write_text(
            json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Wrote {len(outputs)} -> {out_dir / f'{emo}.json'}")


if __name__ == "__main__":
    main()
