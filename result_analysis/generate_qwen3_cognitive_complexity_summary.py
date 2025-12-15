#!/usr/bin/env python3
"""
Generate a first‑pass markdown report stratified by cognitive complexity
for Qwen3‑32B across a couple of tasks.

Output: result_analysis/qwen3_cognitive_complexity_summary.md

This script computes per‑item cognitive_complexity = avg_thinking_neutral / avg_no_thinking_neutral
and reports tertile summaries by task. It does not compute emotion deltas yet.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Tuple

from result_analysis.cognitive_complexity.metrics import neutral_avg_by_item


ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "qwen3_cognitive_complexity_summary.md"


def _task_name_from_csv(path: Path) -> str:
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            tn = row.get("task_name")
            if tn:
                return tn
    raise ValueError(f"No task_name found in {path}")


def _find_qwen32b_files() -> Tuple[List[Path], List[Path]]:
    think = sorted(Path("results/fantom_qwen3/thinking-nogen").glob("Qwen3-32B-AWQ_*/detailed_results.csv"))
    no = sorted(Path("results/fantom_qwen3/no-thinking-nogen").glob("Qwen3-32B-AWQ_*/detailed_results.csv"))
    return think, no


def _pair_by_task(think_files: List[Path], no_files: List[Path]) -> Dict[str, Tuple[Path, Path]]:
    tmap: Dict[str, Path] = {}
    nmap: Dict[str, Path] = {}
    for p in think_files:
        try:
            tmap[_task_name_from_csv(p)] = p
        except Exception:
            continue
    for p in no_files:
        try:
            nmap[_task_name_from_csv(p)] = p
        except Exception:
            continue
    common = sorted(set(tmap) & set(nmap))
    return {k: (tmap[k], nmap[k]) for k in common}


def _ratios_for_task(tp: Path, np: Path) -> Dict[str, float]:
    at = neutral_avg_by_item(tp)
    an = neutral_avg_by_item(np)
    out: Dict[str, float] = {}
    for item, dn in an.items():
        if dn == 0.0:
            continue
        if item in at:
            out[item] = at[item] / dn
    return out


def _tertiles_by_item(rmap: Dict[str, float]) -> Tuple[List[str], List[str], List[str]]:
    if not rmap:
        return [], [], []
    items_sorted = sorted(rmap.items(), key=lambda kv: kv[1])
    n = len(items_sorted)
    base, rem = n // 3, n % 3
    n_low = base + (1 if rem > 0 else 0)
    n_mid = base + (1 if rem > 1 else 0)
    low = [k for k, _ in items_sorted[:n_low]]
    mid = [k for k, _ in items_sorted[n_low : n_low + n_mid]]
    high = [k for k, _ in items_sorted[n_low + n_mid :]]
    return low, mid, high


def main() -> None:
    think, no = _find_qwen32b_files()
    pairs = _pair_by_task(think, no)
    if not pairs:
        raise SystemExit("No paired Qwen3-32B tasks found.")

    # Choose up to two tasks deterministically
    tasks = sorted(pairs.keys())[:2]

    lines: List[str] = []
    lines.append("# Qwen3‑32B Cognitive Complexity Summary (first pass)\n")
    lines.append("Model: Qwen3‑32B‑AWQ; metric: cognitive_complexity = thinking_neutral / no‑thinking_neutral\n")
    lines.append("Note: tertiles are by item‑level ratios; emotion deltas not computed in this pass.\n")

    for t in tasks:
        tp, np = pairs[t]
        ratios = _ratios_for_task(tp, np)
        low, mid, high = _tertiles_by_item(ratios)
        def stats(keys: List[str]) -> Tuple[int, float, float]:
            vals = [ratios[k] for k in keys]
            return (len(vals), (mean(vals) if vals else float('nan')), (median(vals) if vals else float('nan')))
        nL, mL, mdL = stats(low)
        nM, mM, mdM = stats(mid)
        nH, mH, mdH = stats(high)

        lines.append(f"\n## Task: {t}\n")
        lines.append(f"Files:\n- thinking: {tp}\n- no‑thinking: {np}\n")
        lines.append("Tertile Summary (n, mean, median):\n")
        lines.append(f"- low:  n={nL}, mean={mL:.3f}, median={mdL:.3f}")
        lines.append(f"- mid:  n={nM}, mean={mM:.3f}, median={mdM:.3f}")
        lines.append(f"- high: n={nH}, mean={mH:.3f}, median={mdH:.3f}\n")

        # Top examples
        top = sorted(ratios.items(), key=lambda kv: kv[1], reverse=True)[:5]
        lines.append("Top 5 items by ratio:")
        for k, v in top:
            lines.append(f"- {k}: {v:.3f}")

    OUT_PATH.write_text("\n".join(lines))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

