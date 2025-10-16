"""
Generate BFCL significance summaries using paired t-tests across repeats.

Outputs
- result_analysis/bfcl_emotion_significance_summary.md: per-model, per emotion: avg delta, avg t, and fraction of runs significant at alpha=0.05.
- result_analysis/bfcl_emotion_by_category_significance.md: per-model, per-category version of the above.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .bfcl_significance import (
    discover_bfcl_runs,
    aggregate_significance_by_model,
    paired_t_from_summary_by_repeat,
    RunInfo,
)
from .generate_bfcl_emotion_summary import parse_bfcl_category, BFCL_DIR


ROOT = Path(__file__).resolve().parents[1]
OUT_MODEL = ROOT / "result_analysis" / "bfcl_emotion_significance_summary.md"
OUT_CATEGORY = ROOT / "result_analysis" / "bfcl_emotion_by_category_significance.md"


def _fmt_pp(x: float) -> str:
    return f"{x*100:.2f} pp"


def _references_section(runs: List[RunInfo]) -> str:
    lines: List[str] = []
    lines.append("References (run directories)")
    lines.append(f"- Base: {BFCL_DIR}")
    for r in runs:
        lines.append(f"- {r.model} | {r.task_type} | {r.path}")
    return "\n".join(lines)


def build_model_md(agg: Dict[str, Dict[str, Dict[str, float]]], runs: List[RunInfo]) -> str:
    lines: List[str] = []
    lines.append("# BFCL Significance by Model (paired t across repeats)")
    lines.append("")
    lines.append("Last Updated: 2025-10-01")
    lines.append("")
    lines.append("Scope")
    lines.append("- Per run: paired t-test on per-repeat means (emotion − neutral), df = n_pairs−1, alpha = 0.05 (two-sided).")
    lines.append("- Aggregation: per model, report avg delta, avg t-stat, and fraction of runs marked significant.")
    lines.append("")
    for model in sorted(agg.keys()):
        lines.append(f"## {model}")
        for emo in ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]:
            if emo not in agg[model]:
                continue
            d = agg[model][emo]
            lines.append(
                f"- {emo.title()}: Δ={_fmt_pp(d['avg_delta'])}, t̄={d['avg_t']:.2f}, sig_rate={d['sig_rate']:.2f}"
            )
        lines.append("")
    lines.append("")
    lines.append(_references_section(runs))
    return "\n".join(lines)


def aggregate_by_model_and_category(runs: Iterable[RunInfo]) -> Dict[str, Dict[str, Dict[str, float]]]:
    # model -> cat -> emo -> accumulators (later reduced to avg and sig_rate)
    acc: Dict[str, Dict[str, Dict[str, List[Tuple[bool, float, float]]]]] = {}
    for r in runs:
        cat = parse_bfcl_category(r.task_type)
        res = paired_t_from_summary_by_repeat(r.path)
        mm = acc.setdefault(r.model, {}).setdefault(cat, {})
        for emo, d in res.items():
            mm.setdefault(emo, []).append((d["significant"], d["t_stat"], d["mean_delta"]))

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model, cats in acc.items():
        out[model] = {}
        for cat, emomap in cats.items():
            stats_cat: Dict[str, float] = {}
            # We'll flatten by writing keys like '{cat}:{emo}:sig_rate' in MD builder, so keep nested.
            out[model][cat] = {}
            for emo, arr in emomap.items():
                if not arr:
                    continue
                n = len(arr)
                sig_rate = sum(1 for s, _, _ in arr if s) / n
                avg_t = sum(t for _, t, _ in arr) / n
                avg_delta = sum(d for _, _, d in arr) / n
                out[model][cat][emo] = {"sig_rate": sig_rate, "avg_t": avg_t, "avg_delta": avg_delta}
    return out


def build_category_md(agg: Dict[str, Dict[str, Dict[str, Dict[str, float]]]], runs: List[RunInfo]) -> str:
    lines: List[str] = []
    lines.append("# BFCL Significance by Category (paired t across repeats)")
    lines.append("")
    lines.append("Last Updated: 2025-10-01")
    lines.append("")
    lines.append("Scope")
    lines.append("- Grouping: per model, per category (simple/multiple/parallel/parallel_multiple).")
    lines.append("- Metric: Δ mean, avg t, and sig_rate across runs in each model+category.")
    lines.append("")
    # Index run paths per model+category for inline references
    paths_idx: Dict[str, Dict[str, List[str]]] = {}
    for r in runs:
        cat = parse_bfcl_category(r.task_type)
        paths_idx.setdefault(r.model, {}).setdefault(cat, []).append(str(r.path))

    for model in sorted(agg.keys()):
        lines.append(f"## {model}")
        for cat in ["simple", "multiple", "parallel", "parallel_multiple"]:
            if cat not in agg[model]:
                continue
            ref_paths = ", ".join(paths_idx.get(model, {}).get(cat, []))
            if ref_paths:
                lines.append(f"- {cat}: {ref_paths}")
            else:
                lines.append(f"- {cat}:")
            for emo in ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]:
                d = agg[model][cat].get(emo)
                if not d:
                    continue
                lines.append(
                    f"  - {emo.title()}: Δ={_fmt_pp(d['avg_delta'])}, t̄={d['avg_t']:.2f}, sig_rate={d['sig_rate']:.2f}"
                )
        lines.append("")
    lines.append("")
    lines.append(_references_section(runs))
    return "\n".join(lines)


def main() -> None:
    runs = discover_bfcl_runs(BFCL_DIR)
    if not runs:
        raise SystemExit(f"No runs found under {BFCL_DIR}")
    by_model = aggregate_significance_by_model(runs)
    OUT_MODEL.write_text(build_model_md(by_model, runs))
    by_cat = aggregate_by_model_and_category(runs)
    OUT_CATEGORY.write_text(build_category_md(by_cat, runs))
    print(f"Wrote {OUT_MODEL}")
    print(f"Wrote {OUT_CATEGORY}")


if __name__ == "__main__":
    main()
