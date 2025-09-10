#!/usr/bin/env python3
"""
Transform BFCL dataset JSON into this repo's JSONL format.

Supported categories:
- live_simple  (single function)
- live_multiple (multiple functions, ordered)

Input expectations (based on BFCL_dataloading_doc.md):
- Prompts file (JSON): list of items with fields:
  - id: str
  - question: list[{role, content}] (single user turn for live_simple)
  - function: {name, parameters{...}} OR functions: [{...}, ...]
- Ground truth file (JSON): list of items with fields:
  - id: str
  - ground_truth: list of {"<func>": {"<param>": [allowed_values...]}}

Output: JSONL where each line is a merged record:
{
  "id": ..., "question": [...],
  "function": {...} or "functions": [...],
  "ground_truth": [...]
}

Usage examples:
  python scripts/bfcl_transform.py \
    --category live_simple \
    --input /path/to/BFCL_v4_live_simple.json \
    --ground_truth /path/to/possible_answer/BFCL_v4_live_simple.json \
    --output data/BFCL/bfcl_live_simple.jsonl

  python scripts/bfcl_transform.py \
    --category live_multiple \
    --input /path/to/BFCL_v4_live_multiple.json \
    --ground_truth /path/to/possible_answer/BFCL_v4_live_multiple.json \
    --output data/BFCL/bfcl_live_multiple.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Any:
    """Load JSON or JSONL (auto-detect)."""
    with path.open("r", encoding="utf-8") as f:
        data = f.read()
    # Try full JSON first
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        # Fallback to JSONL
        items = []
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
        return items


def index_by_id(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for it in items:
        id_ = it.get("id")
        if not isinstance(id_, str):
            raise ValueError(f"Missing or non-string id in ground truth: {it}")
        idx[id_] = it
    return idx


def transform_item(category: str, prompt_item: Dict[str, Any], gt_item: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "id": prompt_item.get("id"),
        "question": prompt_item.get("question"),
    }

    # Copy function schema
    # Copy function schema (handle schemas that always use 'functions')
    funcs = prompt_item.get("functions")
    func = prompt_item.get("function")
    if funcs is not None:
        # Some datasets use 'functions' even for live_simple
        if not isinstance(funcs, list):
            raise ValueError(f"Expected 'functions' to be a list, got: {type(funcs)}")
        out["functions"] = funcs
    elif func is not None:
        if isinstance(func, dict):
            out["function"] = func
        elif isinstance(func, list):
            # Normalize list under 'function' key
            if len(func) == 1 and isinstance(func[0], dict):
                out["function"] = func[0]
            else:
                out["functions"] = func
        else:
            raise ValueError(f"Expected 'function' to be object or list, got: {type(func)}")
    else:
        raise ValueError("Missing 'function'/'functions' in prompt item")

    # Ground truth list passthrough
    gt_list = gt_item.get("ground_truth")
    if not isinstance(gt_list, list):
        raise ValueError(f"Expected ground_truth list for id={prompt_item.get('id')}")
    out["ground_truth"] = gt_list
    return out


def transform(category: str, prompts_path: Path, gt_path: Path, output_path: Path) -> int:
    prompts = load_json(prompts_path)
    gts = load_json(gt_path)

    if not isinstance(prompts, list) or not isinstance(gts, list):
        raise ValueError("Both input and ground truth files must be JSON lists")

    gt_index = index_by_id(gts)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as out_f:
        for item in prompts:
            pid = item.get("id")
            if pid is None or pid not in gt_index:
                # Skip items without ground truth
                continue
            merged = transform_item(category, item, gt_index[pid])
            out_f.write(json.dumps(merged, ensure_ascii=False) + "\n")
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Transform BFCL JSON to JSONL for this repo")
    parser.add_argument("--category", required=True, choices=["live_simple", "live_multiple"], help="BFCL task category")
    parser.add_argument("--input", required=True, type=Path, help="Path to BFCL prompts JSON (category file)")
    parser.add_argument("--ground_truth", required=True, type=Path, help="Path to BFCL ground truth JSON (possible_answer)")
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL path (default: data/BFCL/bfcl_<category>.jsonl)")

    args = parser.parse_args()
    out = args.output or Path("data/BFCL") / f"bfcl_{args.category}.jsonl"

    n = transform(args.category, args.input, args.ground_truth, out)
    print(f"✅ Wrote {n} items to {out}")


if __name__ == "__main__":
    main()
