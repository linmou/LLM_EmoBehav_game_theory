"""
Compute paired t-test significance for HumanEval runs using detailed_results.csv.

Approach
- For each run dir, read per-item score for neutral and each emotion.
- Compute paired differences d_i = score_emotion_i - score_neutral_i for matching item_id.
- t = dbar / (sd_d / sqrt(n)), df = n-1; significance by two-sided alpha=0.05.

Outputs
- paired_t_vs_neutral_from_detailed(run_dir) -> emotion -> { t_stat, df, significant, n_pairs, mean_delta }
- discover_humaneval_runs(base) -> list of RunInfo(path, model)
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
HUMAN_EVAL_DIR = ROOT / "results" / "humaneval"


@dataclass
class RunInfo:
    path: Path
    model: str


def _t_critical_two_sided(df: int, alpha: float = 0.05) -> float:
    # Minimal t-critical table for two-sided alpha=0.05
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
    keys = sorted(table)
    for i, k in enumerate(keys):
        if df < k:
            k0 = keys[i - 1]
            k1 = k
            v0 = table[k0]
            v1 = table[k1]
            return v0 + (v1 - v0) * ((df - k0) / (k1 - k0))
    return table[keys[-1]]


def _read_detailed(run_dir: Path) -> List[Dict[str, str]]:
    p = Path(run_dir) / "detailed_results.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    with p.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _collect_scores_by_emotion(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for r in rows:
        emo = (r.get("emotion") or "").strip().lower()
        item = (r.get("item_id") or "").strip()
        try:
            score = float(r.get("score") or 0.0)
        except Exception:
            score = 0.0
        if not emo or not item:
            continue
        out.setdefault(emo, {})[item] = score
    return out


def paired_t_vs_neutral_from_detailed(run_dir: Path) -> Dict[str, Dict[str, float]]:
    rows = _read_detailed(run_dir)
    by_emo = _collect_scores_by_emotion(rows)
    neutral = by_emo.get("neutral")
    if not neutral:
        raise ValueError(f"No neutral rows found in {run_dir}")
    out: Dict[str, Dict[str, float]] = {}
    for emo, emap in by_emo.items():
        if emo == "neutral":
            continue
        ids = sorted(set(emap.keys()) & set(neutral.keys()))
        if len(ids) < 2:
            continue
        diffs = [emap[i] - neutral[i] for i in ids]
        n = len(diffs)
        mean_d = sum(diffs) / n
        ssd = sum((x - mean_d) ** 2 for x in diffs)
        sd = math.sqrt(ssd / (n - 1)) if n > 1 else float("nan")
        if sd == 0:
            # Preserve sign for zero-variance case
            t_stat = (math.copysign(float("inf"), mean_d) if mean_d != 0 else 0.0)
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


def discover_humaneval_runs(base: Path | None = None) -> List[RunInfo]:
    base = HUMAN_EVAL_DIR if base is None else Path(base)
    runs: List[RunInfo] = []
    if not base.exists():
        return runs
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        if not (p / "detailed_results.csv").exists():
            continue
        runs.append(RunInfo(path=p, model=p.name))
    return runs


# --- Markdown generation (merged from generator) ---

def _fmt_pp(x: float) -> str:
    return f"{x*100:.2f} pp"


def _read_neutral_mean(run_dir: Path) -> float:
    p = Path(run_dir) / "detailed_results.csv"
    with p.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    vals: List[float] = []
    for r in rows:
        if (r.get("emotion") or "").strip().lower() != "neutral":
            continue
        try:
            vals.append(float(r.get("score") or 0.0))
        except Exception:
            pass
    return sum(vals) / len(vals) if vals else float("nan")


def build_md() -> str:
    runs = discover_humaneval_runs(HUMAN_EVAL_DIR)
    if not runs:
        raise SystemExit(f"No HumanEval runs under {HUMAN_EVAL_DIR}")
    lines: List[str] = []
    lines.append("# HumanEval Emotion Significance (paired t vs neutral across problems)")
    lines.append("")
    import datetime as _dt
    lines.append(f"Last Updated: {_dt.date.today().isoformat()}")
    lines.append("")
    lines.append("Scope")
    lines.append("- Grouping: per model directory under results/humaneval.")
    lines.append("- Test: paired t across problems comparing each emotion against neutral within the same run (repeats not available).")
    lines.append("- Reported: Δ pass@1 (pp), t-stat, significance flag (* ~ p<0.05, normal approx), and emotion mean.")
    lines.append("")

    for r in runs:
        res = paired_t_vs_neutral_from_detailed(r.path)
        neutral_mean = _read_neutral_mean(r.path)
        lines.append(
            f"- {r.model}: {r.path} (neutral={_fmt_pp(neutral_mean)})"
        )
        order = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
        for emo in order:
            if emo not in res:
                continue
            d = res[emo]
            emo_mean = neutral_mean + d["mean_delta"]
            star = "*" if d["significant"] else ""
            lines.append(
                f"  - {emo.title()}: Δ={_fmt_pp(d['mean_delta'])}, t={d['t_stat']:.2f}{star} (mean={_fmt_pp(emo_mean)})"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    out_path = ROOT / "result_analysis" / "humaneval_emotion_significance.md"
    out_path.write_text(build_md())
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
