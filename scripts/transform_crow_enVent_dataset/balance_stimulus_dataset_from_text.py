"""
scripts/balance_stimulus_dataset_from_text.py

Purpose:
  Create a balanced stimulus dataset by taking an existing per-emotion dataset
  (list[str] per emotion) and "topping up" smaller emotions with additional
  items sampled from `data/stimulus/text/{emotion}.json`.

Rules:
  - Keep all existing items in-place.
  - Add only items that are not already present (normalized match) in the
    source emotion list.
  - Write a new dataset dir with {emotion}.json files.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Set


EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _load_json_list(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError(f"Expected a JSON list[str] in {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Balance a per-emotion stimulus dataset by topping up from data/stimulus/text."
    )
    parser.add_argument("--in_dir", default="data/stimulus/crowd-enVent_textlike")
    parser.add_argument("--text_dir", default="data/stimulus/text")
    parser.add_argument(
        "--out_dir", default="data/stimulus/crowd-enVent_textlike_balanced_v1"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=0,
        help="Target items per emotion; 0 means use max count across in_dir",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    text_dir = Path(args.text_dir)
    out_dir = Path(args.out_dir)

    in_data: Dict[str, List[str]] = {}
    counts: Dict[str, int] = {}
    for emo in EMOTIONS:
        src = in_dir / f"{emo}.json"
        if not src.exists():
            raise FileNotFoundError(f"Missing input file: {src}")
        items = _load_json_list(src)
        in_data[emo] = items
        counts[emo] = len(items)

    target = args.target if args.target > 0 else max(counts.values())
    if target <= 0:
        raise ValueError("Computed target <= 0")

    out_dir.mkdir(parents=True, exist_ok=True)

    for emo in EMOTIONS:
        base = list(in_data[emo])
        need = target - len(base)
        if need < 0:
            raise ValueError(
                f"{emo} has {len(base)} items, which exceeds target={target}; "
                "set a higher --target if you want to keep everything."
            )
        if need == 0:
            (out_dir / f"{emo}.json").write_text(
                json.dumps(base, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"{emo}: kept {len(base)} (target={target})")
            continue

        text_path = text_dir / f"{emo}.json"
        if not text_path.exists():
            raise FileNotFoundError(f"Missing text file: {text_path}")

        candidates = _load_json_list(text_path)
        existing_norm: Set[str] = {_norm(x) for x in base}
        added: List[str] = []
        for cand in candidates:
            n = _norm(cand)
            if n in existing_norm:
                continue
            existing_norm.add(n)
            added.append(cand)
            if len(added) >= need:
                break

        if len(added) < need:
            raise RuntimeError(
                f"{emo}: need {need} extra items, but only found {len(added)} non-duplicates in {text_path}"
            )

        out = base + added
        (out_dir / f"{emo}.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"{emo}: {len(base)} + {len(added)} = {len(out)} (target={target})")


if __name__ == "__main__":
    main()

