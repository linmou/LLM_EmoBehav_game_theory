#!/usr/bin/env python3
"""
scripts/augment_game_choice_bins_gemini.py

Purpose:
- Offline augmentation for game-theory scenario JSON files by expanding anchor
  choices into more granular bins using Gemini.

Design:
- Writes a NEW JSON file (does not mutate the input file).
- Skips records that already contain `augmented_options_v1` of the requested length.
- For 2-anchor choices and bins=4: uses a hidden pivot M, then generates X1/X2 and
  outputs [A, X1, X2, B], discarding M (but optionally recording provenance).
- For 3-anchor choices and bins=5: generates one option between low↔medium and
  one between medium↔high, outputting [L, X_LM, M, X_MH, H].
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api_configs import GEMINI_CONFIG
from emotion_experiment_engine.evaluation_utils import llm_evaluate_response


AUG_OPTIONS_FIELD = "augmented_options_v1"
AUG_META_FIELD = "augmented_options_meta_v1"


SYSTEM_PROMPT = """You generate ONE intermediate decision option for a game-theory scenario.
Return ONLY valid JSON. No markdown. No extra keys.

Rules:
- Output must be a single actionable option text (imperative or clear action).
- Must fit the scenario context and match the style/length of the anchors.
- Do NOT mention "Option", "anchor", "A/B", "intermediate", "between", "pivot", "Gemini", or anything meta.
- Do NOT add payoff talk, math, or probabilities unless the anchors already contain them.
- Do NOT add new entities; reuse the same actor(s) implied by the anchors.
- Avoid refusal/inaction unless anchors already allow it.
- Keep it concise: 1 sentence.
JSON schema:
{"option": "<string>"}
"""


FEWSHOT_NEUTRAL = """Few-shot examples:

Example 1:
Scenario: You must choose between releasing a stable update on schedule (A) or releasing an untested feature early (B).
Anchor A: Release the stable, tested update on schedule.
Anchor B: Release the feature-rich update early even though it is less tested.
Output:
{"option":"Release a limited update that includes only the safest improvements while postponing the riskiest new features."}

Example 2:
Scenario: You can either fully share your resources with a rival (A) or keep everything for yourself (B).
Anchor A: Share the resources openly with the rival.
Anchor B: Keep all resources and share nothing.
Output:
{"option":"Share only the minimum necessary resources while keeping the most valuable assets reserved."}
"""


FEWSHOT_BETWEEN_A_M = """Few-shot examples:

Example 1:
Scenario: Shipping decision with stable vs early risky release.
Anchor A: Release the stable, tested update on schedule.
Pivot M: Release a limited update that includes only the safest improvements while postponing the riskiest new features.
Output:
{"option":"Release the stable update on schedule, but hold back any risky components and add an extra round of validation for borderline changes."}

Example 2:
Scenario: Resource sharing vs hoarding.
Anchor A: Share the resources openly with the rival.
Pivot M: Share only the minimum necessary resources while keeping the most valuable assets reserved.
Output:
{"option":"Share the resources, but limit access to the most sensitive parts until trust is established."}
"""


FEWSHOT_BETWEEN_M_B = """Few-shot examples:

Example 1:
Scenario: Shipping decision with stable vs early risky release.
Pivot M: Release a limited update that includes only the safest improvements while postponing the riskiest new features.
Anchor B: Release the feature update early even though it is less tested.
Output:
{"option":"Release the feature update early, but restrict it to a controlled rollout so issues can be caught before full deployment."}

Example 2:
Scenario: Resource sharing vs hoarding.
Pivot M: Share only the minimum necessary resources while keeping the most valuable assets reserved.
Anchor B: Keep all resources and share nothing.
Output:
{"option":"Keep the resources for now, but provide a small, low-risk concession to prevent immediate conflict."}
"""


FEWSHOT_BETWEEN_L_M = """Few-shot examples:

Example 1:
Scenario: Trust decision with low/medium/high levels.
Low: Trust nothing and keep everything.
Medium: Trust somewhat and share a limited amount.
Output:
{"option":"Share a small amount to test trust while keeping most resources protected."}
"""


def _record_context(record: Dict[str, Any]) -> str:
    scenario = record.get("scenario") or ""
    description = record.get("description") or ""
    extra = record.get("additional_context") or ""
    parts = [str(scenario).strip(), str(description).strip(), str(extra).strip()]
    parts = [p for p in parts if p]
    return "\n\n".join(parts) if parts else ""


def _is_choice_dict(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    return all(isinstance(v, str) and v.strip() for v in value.values())


def _guess_choice_field(record: Dict[str, Any]) -> Optional[str]:
    preferred = [
        "behavior_choices",
        "trustor_behavior_choices",
        "trustee_behavior_choices",
        "proposer_behavior_choices",
        "responder_behavior_choices",
    ]
    for key in preferred:
        if _is_choice_dict(record.get(key)):
            return key
    for key, value in record.items():
        if key.endswith("_choices") and _is_choice_dict(value):
            return key
    return None


def _ordered_anchors(choice_field: str, choices: Dict[str, str]) -> List[Tuple[str, str]]:
    items = list(choices.items())
    keys = [k.lower() for k, _ in items]
    joined = " ".join(keys)

    def rank(k: str) -> int:
        k = k.lower()
        if "none" in k:
            return 0
        if "low" in k:
            return 1
        if "medium" in k or "mid" in k:
            return 2
        if "high" in k:
            return 3
        if "cooperate" in k:
            return 1
        if "defect" in k:
            return 3
        if "accept" in k:
            return 1
        if "reject" in k:
            return 3
        return 99

    if any(tok in joined for tok in ("low", "medium", "high", "none", "mid")):
        return sorted(items, key=lambda kv: (rank(kv[0]), kv[0]))

    return items


def _gemini_call(query: str, model: str) -> str:
    api = GEMINI_CONFIG["api_key"]
    if not api:
        raise RuntimeError("GEMINI_CONFIG['api_key'] is empty/missing")

    payload = llm_evaluate_response(
        system_prompt=SYSTEM_PROMPT,
        query=query,
        llm_eval_config={"client": "gemini", "model": model},
    )
    option = payload.get("option")
    if not isinstance(option, str) or not option.strip():
        raise RuntimeError(f"Gemini returned invalid payload: {payload!r}")
    return option.strip()


def _gen_pivot(record_ctx: str, a: str, b: str, model: str) -> str:
    query = (
        "Task: Create a \"do neither A nor B\" pivot option.\n"
        "It must be a real move in the story (not refusal), i.e., a hedging/partial step that avoids fully committing to either anchor.\n\n"
        f"Scenario:\n{record_ctx}\n\n"
        f"Anchor A:\n{a}\n\n"
        f"Anchor B:\n{b}\n\n"
        "Return exactly one option in JSON.\n\n"
        f"{FEWSHOT_NEUTRAL}"
    )
    return _gemini_call(query, model=model)


def _gen_between_a_m(record_ctx: str, a: str, m: str, model: str) -> str:
    query = (
        "Task: Generate an option BETWEEN Anchor A and Pivot M.\n"
        "It should lean toward Anchor A, but show hesitation by adding a cautious hedge consistent with Pivot M.\n\n"
        f"Scenario:\n{record_ctx}\n\n"
        f"Anchor A:\n{a}\n\n"
        f"Pivot M:\n{m}\n\n"
        "Return exactly one option in JSON.\n\n"
        f"{FEWSHOT_BETWEEN_A_M}"
    )
    return _gemini_call(query, model=model)


def _gen_between_m_b(record_ctx: str, m: str, b: str, model: str) -> str:
    query = (
        "Task: Generate an option BETWEEN Pivot M and Anchor B.\n"
        "It should lean toward Anchor B, but not fully commit—mitigate the downside that Anchor A tries to avoid.\n\n"
        f"Scenario:\n{record_ctx}\n\n"
        f"Pivot M:\n{m}\n\n"
        f"Anchor B:\n{b}\n\n"
        "Return exactly one option in JSON.\n\n"
        f"{FEWSHOT_BETWEEN_M_B}"
    )
    return _gemini_call(query, model=model)


def _gen_between(record_ctx: str, left: str, right: str, model: str, fewshot: str) -> str:
    query = (
        "Task: Generate an option BETWEEN Left and Right.\n"
        "It must be a plausible action that is meaningfully intermediate in commitment/risk.\n\n"
        f"Scenario:\n{record_ctx}\n\n"
        f"Left:\n{left}\n\n"
        f"Right:\n{right}\n\n"
        "Return exactly one option in JSON.\n\n"
        f"{fewshot}"
    )
    return _gemini_call(query, model=model)


def augment_record(
    record: Dict[str, Any],
    *,
    bins: int,
    model: str,
    choice_field: Optional[str] = None,
    keep_pivot_in_meta: bool = True,
) -> Tuple[Dict[str, Any], bool]:
    out = deepcopy(record)

    existing = out.get(AUG_OPTIONS_FIELD)
    if isinstance(existing, list) and len(existing) == bins and all(isinstance(x, str) for x in existing):
        return out, False

    cf = choice_field or _guess_choice_field(out)
    if not cf:
        raise RuntimeError("Unable to locate *_choices field in record")

    choices = out.get(cf)
    if not _is_choice_dict(choices):
        raise RuntimeError(f"Choice field {cf!r} is not a non-empty dict[str,str]")

    anchors = _ordered_anchors(cf, choices)
    ctx = _record_context(out)
    if not ctx:
        ctx = "(No scenario/description provided)"

    anchor_texts = [t for _, t in anchors]

    if len(anchor_texts) == 2 and bins == 4:
        a, b = anchor_texts
        pivot = _gen_pivot(ctx, a, b, model=model)
        x1 = _gen_between_a_m(ctx, a, pivot, model=model)
        x2 = _gen_between_m_b(ctx, pivot, b, model=model)
        out[AUG_OPTIONS_FIELD] = [a, x1, x2, b]
        out[AUG_META_FIELD] = {
            "method": "pivot_split_v1",
            "bins": bins,
            "choice_field": cf,
            "anchors": anchor_texts,
            "generator": {"client": "gemini", "model": model},
            **({"pivot": pivot} if keep_pivot_in_meta else {}),
        }
        return out, True

    if len(anchor_texts) == 3 and bins == 5:
        low, mid, high = anchor_texts
        x_lm = _gen_between(ctx, low, mid, model=model, fewshot=FEWSHOT_BETWEEN_L_M)
        x_mh = _gen_between(ctx, mid, high, model=model, fewshot=FEWSHOT_BETWEEN_L_M)
        out[AUG_OPTIONS_FIELD] = [low, x_lm, mid, x_mh, high]
        out[AUG_META_FIELD] = {
            "method": "adjacent_interpolation_v1",
            "bins": bins,
            "choice_field": cf,
            "anchors": anchor_texts,
            "generator": {"client": "gemini", "model": model},
        }
        return out, True

    raise RuntimeError(
        f"Unsupported anchors/bins combination: anchors={len(anchor_texts)} bins={bins} "
        f"(choice_field={cf})"
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Augment game-theory scenario JSON with additional choice bins using Gemini (writes a new JSON file)."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input *_all_data_samples.json")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path (new file)")
    parser.add_argument("--bins", required=True, type=int, choices=(4, 5), help="Total bins after augmentation")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name")
    parser.add_argument("--choice-field", default=None, help="Override choice field (e.g., behavior_choices)")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N records (for smoke tests)")
    parser.add_argument("--keep-pivot-in-meta", action="store_true", help="Persist hidden pivot text in metadata")
    parser.add_argument(
        "--preserve-tail",
        action="store_true",
        help="When --limit is used, append the unprocessed tail to keep output length identical to input.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers (threads). >1 accelerates Gemini calls across records.",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    data = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("Input JSON must be a list of scenario records")

    limit = args.limit if args.limit is not None else len(data)
    limit = min(limit, len(data))
    processed_records = data[:limit]

    out_list: List[Optional[Dict[str, Any]]] = [None] * limit
    changed = 0

    def _work(idx: int, rec: Dict[str, Any]) -> Tuple[int, Dict[str, Any], bool]:
        updated, did_change = augment_record(
            rec,
            bins=args.bins,
            model=args.model,
            choice_field=args.choice_field,
            keep_pivot_in_meta=bool(args.keep_pivot_in_meta),
        )
        return idx, updated, did_change

    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    if args.workers == 1:
        for idx, rec in enumerate(processed_records):
            if not isinstance(rec, dict):
                raise SystemExit(f"Record {idx} is not an object")
            i, updated, did_change = _work(idx, rec)
            out_list[i] = updated
            if did_change:
                changed += 1
                print(f"[{idx}] augmented", file=sys.stderr)
            else:
                print(
                    f"[{idx}] skipped (already has {AUG_OPTIONS_FIELD})",
                    file=sys.stderr,
                )
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = []
            for idx, rec in enumerate(processed_records):
                if not isinstance(rec, dict):
                    raise SystemExit(f"Record {idx} is not an object")
                futures.append(ex.submit(_work, idx, rec))

            for fut in as_completed(futures):
                idx, updated, did_change = fut.result()
                out_list[idx] = updated
                if did_change:
                    changed += 1
                    print(f"[{idx}] augmented", file=sys.stderr)
                else:
                    print(
                        f"[{idx}] skipped (already has {AUG_OPTIONS_FIELD})",
                        file=sys.stderr,
                    )

    if any(item is None for item in out_list):
        raise SystemExit("Internal error: missing augmented records")

    final_list: List[Dict[str, Any]] = [item for item in out_list if item is not None]

    if args.preserve_tail and limit < len(data):
        final_list.extend(deepcopy(data[limit:]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(final_list, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"wrote: {args.output} (processed={limit} changed={changed})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
