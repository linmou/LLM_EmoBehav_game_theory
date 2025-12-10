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


def _norm_yes_no(s: str) -> Optional[str]:
    s = s.strip().lower()
    if s.startswith("yes"):
        return "yes"
    if s.startswith("no"):
        return "no"
    return None


def transform_binary_generic(src_name: str, out_task: str) -> int:
    src = SRC_DIR / src_name
    out = OUT_DIR / f"fantom_{out_task}.jsonl"

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
        q = question.strip()
        if fact_q and isinstance(fact_q, str):
            fq = fact_q.strip()
            if fq and fq not in q:
                q = f"{q} {fq}"
        ans = _norm_yes_no(correct)
        if ans is None:
            continue
        rows.append({"context": context.strip(), "question": q, "answer": ans})
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


def transform_choice_generic(src_name: str, out_task: str) -> int:
    src = SRC_DIR / src_name
    out = OUT_DIR / f"fantom_{out_task}.jsonl"

    raw = _load_json_array(src)
    rows = []
    for ex in raw:
        if not isinstance(ex, dict):
            continue
        context = ex.get("context")
        question = ex.get("question")
        if not isinstance(context, str) or not isinstance(question, str):
            continue
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
        correct_idx = ex.get("correct_answer")
        try:
            correct_idx = int(correct_idx)
        except Exception:
            continue
        if not (0 <= correct_idx < len(options)):
            continue
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


def transform_list_generic(src_name: str, out_task: str) -> int:
    src = SRC_DIR / src_name
    out = OUT_DIR / f"fantom_{out_task}.jsonl"
    raw = _load_json_array(src)
    rows = []
    for ex in raw:
        if not isinstance(ex, dict):
            continue
        context = ex.get("context")
        question = ex.get("question")
        answers = ex.get("correct_answer")
        wrong = ex.get("wrong_answer")
        if not isinstance(context, str) or not isinstance(question, str) or not isinstance(answers, list):
            continue
        answers = [str(a).strip() for a in answers if str(a).strip()]
        if not answers:
            continue
        row = {
            "context": context.strip(),
            "question": question.strip(),
            "answers": answers,
        }
        if isinstance(wrong, list) and wrong:
            row["wrong_answer"] = [str(a).strip() for a in wrong if str(a).strip()]
        rows.append(row)
    return _write_jsonl(out, rows)


def transform_text_generic(src_name: str, out_task: str) -> int:
    src = SRC_DIR / src_name
    out = OUT_DIR / f"fantom_{out_task}.jsonl"
    raw = _load_json_array(src)
    rows = []
    for ex in raw:
        if not isinstance(ex, dict):
            continue
        context = ex.get("context")
        question = ex.get("question")
        answer = ex.get("correct_answer")
        wrong = ex.get("wrong_answer")
        if not isinstance(context, str) or not isinstance(question, str) or not isinstance(answer, str):
            continue
        row = {
            "context": context.strip(),
            "question": question.strip(),
            "answer": answer.strip(),
        }
        if "belief_gen" in out_task and isinstance(wrong, str) and wrong.strip():
            row["wrong_answer"] = wrong.strip()
        rows.append(row)
    return _write_jsonl(out, rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["both", "binary", "choice", "all"], default="both")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    if args.which in ("both", "binary", "all"):
        total += transform_binary_generic("short_answerability_binary_inaccessible_fantom.json", "short_answerability_binary_inaccessible")
        total += transform_binary_generic("short_infoaccessibility_binary_inaccessible_fantom.json", "short_infoaccessibility_binary_inaccessible")
        # Accessible variants
        total += transform_binary_generic("short_answerability_binary_accessible_fantom.json", "short_answerability_binary_accessible")
        total += transform_binary_generic("short_infoaccessibility_binary_accessible_fantom.json", "short_infoaccessibility_binary_accessible")
        # Full variants (if present)
        for name in [
            ("full_answerability_binary_inaccessible_fantom.json", "full_answerability_binary_inaccessible"),
            ("full_answerability_binary_accessible_fantom.json", "full_answerability_binary_accessible"),
            ("full_infoaccessibility_binary_inaccessible_fantom.json", "full_infoaccessibility_binary_inaccessible"),
            ("full_infoaccessibility_binary_accessible_fantom.json", "full_infoaccessibility_binary_accessible"),
        ]:
            p = SRC_DIR / name[0]
            if p.exists():
                total += transform_binary_generic(name[0], name[1])

    if args.which in ("both", "choice", "all"):
        total += transform_choice_generic("short_belief_choice_inaccessible_fantom.json", "short_belief_choice_inaccessible")
        total += transform_choice_generic("short_belief_choice_accessible_fantom.json", "short_belief_choice_accessible")
        # Full variants
        for name in [
            ("full_belief_choice_inaccessible_fantom.json", "full_belief_choice_inaccessible"),
            ("full_belief_choice_accessible_fantom.json", "full_belief_choice_accessible"),
        ]:
            p = SRC_DIR / name[0]
            if p.exists():
                total += transform_choice_generic(name[0], name[1])

    if args.which in ("all",):
        # list tasks
        total += transform_list_generic("short_answerability_list_inaccessible_fantom.json", "short_answerability_list_inaccessible")
        total += transform_list_generic("short_answerability_list_accessible_fantom.json", "short_answerability_list_accessible")
        total += transform_list_generic("short_infoaccessibility_list_inaccessible_fantom.json", "short_infoaccessibility_list_inaccessible")
        total += transform_list_generic("short_infoaccessibility_list_accessible_fantom.json", "short_infoaccessibility_list_accessible")
        for name in [
            ("full_answerability_list_inaccessible_fantom.json", "full_answerability_list_inaccessible"),
            ("full_answerability_list_accessible_fantom.json", "full_answerability_list_accessible"),
            ("full_infoaccessibility_list_inaccessible_fantom.json", "full_infoaccessibility_list_inaccessible"),
            ("full_infoaccessibility_list_accessible_fantom.json", "full_infoaccessibility_list_accessible"),
        ]:
            p = SRC_DIR / name[0]
            if p.exists():
                total += transform_list_generic(name[0], name[1])

        # text generation tasks
        total += transform_text_generic("short_belief_gen_inaccessible_fantom.json", "short_belief_gen_inaccessible")
        total += transform_text_generic("short_belief_gen_accessible_fantom.json", "short_belief_gen_accessible")
        total += transform_text_generic("short_fact_fantom.json", "short_fact")
        for name in [
            ("full_belief_gen_inaccessible_fantom.json", "full_belief_gen_inaccessible"),
            ("full_belief_gen_accessible_fantom.json", "full_belief_gen_accessible"),
            ("full_fact_fantom.json", "full_fact"),
        ]:
            p = SRC_DIR / name[0]
            if p.exists():
                total += transform_text_generic(name[0], name[1])

    print(f"Done. Total items written: {total}")


if __name__ == "__main__":
    main()
