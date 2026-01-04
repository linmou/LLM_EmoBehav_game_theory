"""
Quick one-off script to clean Diplomacy escalation behavior choices using Gemini.

Source:
    data_creation/scenario_creation/langgraph_creation/
        diplomacy_Escalation_Game_all_data_samples.json.bk

Output:
    data_creation/scenario_creation/langgraph_creation/
        diplomacy_Escalation_Game_all_data_samples.cleaned.json

Behavior:
    - For each record, send the existing `behavior_choices` dict to Gemini.
    - Ask Gemini to rewrite each choice as a neutral description of the action.
    - Explicitly remove emotion / attitude wording (e.g., "aggressively",
      "cautiously", "fearful", "angry", "optimistic", etc.).
    - Keep the underlying action semantics unchanged.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import google.generativeai as genai  # type: ignore[import-not-found]

from api_configs import GEMINI_CONFIG


ROOT = Path("data_creation/scenario_creation/langgraph_creation")
INPUT_PATH = ROOT / "diplomacy_Escalation_Game_all_data_samples.json.bk"
FEWSHOT_PATH = ROOT / "diplomacy_Escalation_Game_all_data_samples.json"
OUTPUT_PATH = ROOT / "diplomacy_Escalation_Game_all_data_samples.cleaned.json"

# Use the same family as in config/diplomacy_escalation_game.yaml
DEFAULT_MODEL_NAME = "gemini-2.5-flash-lite"
MAX_WORKERS = 8
FEWSHOT_LIMIT = 8


def _init_model(model_name: str = DEFAULT_MODEL_NAME):
    api_key = GEMINI_CONFIG.get("api_key")
    if not api_key:
        raise RuntimeError("GEMINI_CONFIG.api_key is empty; set it in api_configs.py")

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def _load_fewshot_behavior_examples(limit: int = FEWSHOT_LIMIT) -> List[Dict[str, str]]:
    """
    Load a small set of neutral-style behavior_choices as few-shot examples.

    Uses the current diplomacy_Escalation_Game_all_data_samples.json file as
    a style reference. It does not assume one-to-one alignment with the .bk file;
    it only provides examples of the desired wording.
    """
    if not FEWSHOT_PATH.exists():
        return []

    raw = json.loads(FEWSHOT_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []

    fewshot: List[Dict[str, str]] = []
    for record in raw:
        bc = record.get("behavior_choices")
        if isinstance(bc, dict):
            fewshot.append({str(k): str(v) for k, v in bc.items()})
        if len(fewshot) >= limit:
            break
    return fewshot


def _rewrite_behavior_choices(
    model: Any,
    behavior_choices: Dict[str, str],
    scenario: str,
    description: str,
    fewshot_examples: List[Dict[str, str]],
) -> Dict[str, str]:
    """Call Gemini to rewrite behavior_choices in a neutral, non-emotional style."""
    system_instructions = (
        "You edit game-theoretic behavior options for a Diplomacy escalation game. "
        "Your job is to rewrite each behavior choice so that it describes only the "
        "concrete action being taken, without any explicit emotional or attitudinal "
        "phrasing.\n\n"
        "Remove or avoid words that describe feelings, attitudes, or intensity such as "
        "\"aggressively\", \"cautiously\", \"fearful\", \"angry\", \"eager\", "
        "\"hesitant\", \"provocative\", \"calm\", etc. "
        "Do NOT change who moves where or what order is issued. "
        "Keep each option clear, concise, and in plain English. "
        "Use the style of the example behavior choices when possible."
    )

    prompt = {
        "scenario_id": scenario,
        "scenario_description": description,
        "behavior_choices": behavior_choices,
        "fewshot_examples": fewshot_examples,
        "instructions": (
            "Rewrite behavior_choices so that each value is a neutral description "
            "of the corresponding action, with no emotional or attitudinal wording. "
            "Return ONLY a JSON object with the SAME KEYS as behavior_choices and "
            "string values."
        ),
    }

    response = model.generate_content(
        [
            {
                "role": "user",
                "parts": [
                    system_instructions,
                    "\n\nINPUT JSON:\n",
                    json.dumps(prompt),
                ],
            }
        ]
    )

    raw_text = getattr(response, "text", "") or ""
    raw_text = raw_text.strip()

    # Best effort: extract the first {...} block
    if not (raw_text.startswith("{") and raw_text.endswith("}")):
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw_text = raw_text[start : end + 1]

    cleaned: Dict[str, Any] = json.loads(raw_text)

    # Ensure we return a simple str->str mapping for known keys
    result: Dict[str, str] = {}
    for k, v in behavior_choices.items():
        new_val = cleaned.get(k, v)
        if not isinstance(new_val, str):
            new_val = str(new_val)
        result[k] = new_val.strip()
    return result


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    print(f"Loading input from {INPUT_PATH}")
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {INPUT_PATH}, got {type(data)}")

    total = len(data)
    print(f"Total records: {total}")
    model = _init_model()
    fewshot_examples = _load_fewshot_behavior_examples(12)
    print(f"Loaded {len(fewshot_examples)} few-shot behavior_choices examples")

    updated: List[Dict[str, Any]] = [dict(r) for r in data]

    def _worker(args: Tuple[int, Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
        idx, record = args
        bc = record.get("behavior_choices")
        if not isinstance(bc, dict):
            return idx, record

        scenario = str(record.get("scenario", ""))
        description = str(record.get("description", ""))

        try:
            new_bc = _rewrite_behavior_choices(
                model, bc, scenario, description, fewshot_examples
            )
            record = dict(record)
            record["behavior_choices"] = new_bc
        except Exception as exc:
            print(
                f"[{idx + 1}/{total}] WARNING: failed to clean behavior_choices "
                f"for {scenario}: {exc}"
            )
        return idx, record

    print(f"Cleaning behavior_choices in parallel with max_workers={MAX_WORKERS}")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_worker, (idx, rec)): idx
            for idx, rec in enumerate(updated)
        }
        for i, fut in enumerate(as_completed(futures), start=1):
            idx, rec = fut.result()
            updated[idx] = rec
            if i % 10 == 0 or i == total:
                print(f"Progress: {i}/{total} records processed")

    print(f"Writing cleaned data to {OUTPUT_PATH}")
    OUTPUT_PATH.write_text(json.dumps(updated, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Done.")


if __name__ == "__main__":
    main()
