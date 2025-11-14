"""
Compute paired t-test significance for BFCL runs using summary_by_repeat.csv.

Approach
- For each run dir, read per-repeat mean for neutral (intensity 0.0) and each emotion (intensity 1.5).
- Compute paired differences d_i = mean_emotion_i - mean_neutral_i for matching repeat_id.
- t = dbar / (sd_d / sqrt(n)), df = n-1; significance by two-sided alpha=0.05 via t-critical lookup.

Outputs
- paired_t_from_summary_by_repeat(run_dir) -> emotion -> { t_stat, df, significant, n_pairs, mean_delta }
- aggregate_significance_by_model(runs) -> model -> emotion -> { sig_rate, avg_t, avg_delta }

Notes
- We only use the 'mean' column as requested; other summary fields ignored.
- If repeats differ across emotion/neutral, we align on intersection of repeat_id.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import math


@dataclass
class RunInfo:
    path: Path
    model: str
    task_type: str


def _read_summary_by_repeat(run_dir: Path) -> List[Dict[str, str]]:
    p = Path(run_dir) / "summary_by_repeat.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    with p.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _collect_means_by_repeat(rows: List[Dict[str, str]]) -> Dict[Tuple[str, float], Dict[int, float]]:
    out: Dict[Tuple[str, float], Dict[int, float]] = {}
    for r in rows:
        emo = r["emotion"].strip()
        inten = float(r["intensity"])
        rep = int(r["repeat_id"])
        mean = float(r["mean"])
        out.setdefault((emo, inten), {})[rep] = mean
    return out


def _t_critical_two_sided(df: int, alpha: float = 0.05) -> float:
    # Minimal t-critical table for two-sided alpha=0.05
    # df: 1..30, 40, 60, 120, inf
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
        19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
        25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
        40: 2.021, 60: 2.000, 120: 1.980,
    }
    if df in table:
        return table[df]
    if df < 1:
        return float("inf")
    # Linear interpolate between nearest keys
    keys = sorted(table)
    for i, k in enumerate(keys):
        if df < k:
            k0 = keys[i-1]
            k1 = k
            v0 = table[k0]
            v1 = table[k1]
            t = v0 + (v1 - v0) * ((df - k0) / (k1 - k0))
            return t
    # beyond max, use last
    return table[keys[-1]]


def paired_t_from_summary_by_repeat(run_dir: Path) -> Dict[str, Dict[str, float]]:
    rows = _read_summary_by_repeat(run_dir)
    by_key = _collect_means_by_repeat(rows)
    # neutral baseline
    neutral = by_key.get(("neutral", 0.0))
    if not neutral:
        raise ValueError(f"No neutral repeats found in {run_dir}")
    out: Dict[str, Dict[str, float]] = {}
    for (emo, inten), repmap in by_key.items():
        if emo == "neutral":
            continue
        # align by repeat_id intersection
        common = sorted(set(repmap.keys()) & set(neutral.keys()))
        if len(common) < 2:
            # need at least 2 for df>=1
            continue
        diffs = [repmap[i] - neutral[i] for i in common]
        n = len(diffs)
        mean_d = sum(diffs) / n
        # sample std
        ssd = sum((x - mean_d) ** 2 for x in diffs)
        sd = math.sqrt(ssd / (n - 1)) if n > 1 else float("nan")
        if sd == 0:
            t_stat = float("inf") if mean_d != 0 else 0.0
        else:
            t_stat = mean_d / (sd / math.sqrt(n))
        df = n - 1
        tcrit = _t_critical_two_sided(df)
        significant = abs(t_stat) >= tcrit
        out[emo] = {
            "t_stat": t_stat,
            "df": df,
            "significant": significant,
            "n_pairs": n,
            "mean_delta": mean_d,
        }
    return out


def discover_bfcl_runs(base: Path) -> List[RunInfo]:
    runs: List[RunInfo] = []
    if not base.exists():
        return runs
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        if not (p / "summary_by_repeat.csv").exists():
            continue
        name = p.name
        if "_bfcl_" not in name:
            continue
        model = name.split("_bfcl_", 1)[0]
        task_type = "unknown"
        cfg_path = p / "experiment_config.json"
        try:
            d = csv
            import json
            cfg = json.loads(cfg_path.read_text())
            task_type = cfg.get("benchmark", {}).get("task_type", "unknown")
        except Exception:
            pass
        runs.append(RunInfo(p, model, task_type))
    return runs


def aggregate_significance_by_model(runs: Iterable[RunInfo]) -> Dict[str, Dict[str, Dict[str, float]]]:
    # model -> emo -> list of (significant flag, t, mean_delta)
    acc: Dict[str, Dict[str, List[Tuple[bool, float, float]]]] = {}
    for r in runs:
        res = paired_t_from_summary_by_repeat(r.path)
        mm = acc.setdefault(r.model, {})
        for emo, d in res.items():
            mm.setdefault(emo, []).append((d["significant"], d["t_stat"], d["mean_delta"]))
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model, emolist in acc.items():
        emo_stats: Dict[str, Dict[str, float]] = {}
        for emo, arr in emolist.items():
            if not arr:
                continue
            n = len(arr)
            sig_rate = sum(1 for s, _, _ in arr if s) / n
            avg_t = sum(t for _, t, _ in arr) / n
            avg_delta = sum(d for _, _, d in arr) / n
            emo_stats[emo] = {"sig_rate": sig_rate, "avg_t": avg_t, "avg_delta": avg_delta}
        out[model] = emo_stats
    return out

