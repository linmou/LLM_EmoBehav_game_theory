#!/usr/bin/env python3
"""
Convert original FANToM JSON files to project JSONL formats used by FantomDataset.

Source directory (read-only):
    /data/home/jjl7137/fantom/data/fantom

Outputs (JSONL) in repo:
    data/fantom/fantom_short_answerability_binary_inaccessible.jsonl
    data/fantom/fantom_short_belief_choice_inaccessible.jsonl

Usage:
    python scripts/convert_fantom.py            # convert both
    python scripts/convert_fantom.py --which binary
    python scripts/convert_fantom.py --which choice

Design (KISS):
    - Reads original JSON (array) and writes newline-delimited JSON (JSONL)
    - Minimal field mapping with safe guards; skips malformed items
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


SRC_DIR = Path("/data/home/jjl7137/fantom/data/fantom")
OUT_DIR = Path("data/fantom")


def _load_json_array(path: Path) -> list:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    return data


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def transform_binary() -> int:
    src = SRC_DIR / "short_answerability_binary_inaccessible_fantom.json"
    out = OUT_DIR / "fantom_short_answerability_binary_inaccessible.jsonl"

    raw = _load_json_array(src)
    rows = []
    for ex in raw:
        if not isinstance(ex, dict):
            continue
        context = ex.get("context")
        question = ex.get("question")
        fact_q = ex.get("fact_question", "")
        correct = ex.get("correct_answer")

        if not isinstance(context, str) or not isinstance(question, str) or not isinstance(correct, str):
            continue

        # Compose a clear question string
        q = question.strip()
        if fact_q and isinstance(fact_q, str):
            fq = fact_q.strip()
            if fq and fq not in q:
                q = f"{q} {fq}"

        # Map variants like 'no:long' to 'no'
        ans_raw = correct.strip().lower()
        ans = "yes" if ans_raw.startswith("yes") else ("no" if ans_raw.startswith("no") else None)
        if ans is None:
            continue

        rows.append({
            "context": context.strip(),
            "question": q,
            "answer": ans,
        })

    return _write_jsonl(out, rows)


def _parse_choices_from_text(txt: str) -> list[str]:
    # Fallback if choices_list is missing; parse lines starting with (a)/(b)/1./2./A./B.
    import re
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    opts: list[str] = []
    for ln in lines:
        m = re.match(r"^\(?[a-zA-Z0-9]\)?[\.|\)]\s+(.*)$", ln)
        if m:
            opts.append(m.group(1).strip())
    return opts


def transform_choice() -> int:
    src = SRC_DIR / "short_belief_choice_inaccessible_fantom.json"
    out = OUT_DIR / "fantom_short_belief_choice_inaccessible.jsonl"

    raw = _load_json_array(src)
    rows = []
    for ex in raw:
        if not isinstance(ex, dict):
            continue
        context = ex.get("context")
        question = ex.get("question")

        if not isinstance(context, str) or not isinstance(question, str):
            continue

        # Prefer choices_list; else parse choices_text
        options = ex.get("choices_list")
        if not isinstance(options, list):
            choices_text = ex.get("choices_text", "")
            if isinstance(choices_text, str):
                options = _parse_choices_from_text(choices_text)
            else:
                options = []

        options = [str(o).strip() for o in options if str(o).strip()]
        if len(options) < 2:
            continue

        # correct_answer may be index (0-based). Ensure int and within range
        correct_idx = ex.get("correct_answer")
        try:
            correct_idx = int(correct_idx)
        except Exception:
            continue
        if not (0 <= correct_idx < len(options)):
            continue

        # Clean question (remove helper suffix if present)
        q = question.strip()
        if q.endswith("Choose an answer from above:"):
            q = q[: -len("Choose an answer from above:")].strip()

        rows.append({
            "context": context.strip(),
            "question": q,
            "options": options,
            "correct_index": correct_idx,
        })

    return _write_jsonl(out, rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["both", "binary", "choice"], default="both")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    if args.which in ("both", "binary"):
        n = transform_binary()
        print(f"Wrote {n} items to {OUT_DIR / 'fantom_short_answerability_binary_inaccessible.jsonl'}")
        total += n
    if args.which in ("both", "choice"):
        n = transform_choice()
        print(f"Wrote {n} items to {OUT_DIR / 'fantom_short_belief_choice_inaccessible.jsonl'}")
        total += n
    print(f"Done. Total items written: {total}")


if __name__ == "__main__":
    main()

