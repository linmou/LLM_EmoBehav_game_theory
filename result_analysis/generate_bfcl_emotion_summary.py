"""
Generate emotion impact summaries for BFCL (live) runs under results/bfcl_qwen3/live.

Two outputs (mirroring Fantom):
- result_analysis/bfcl_emotion_summary.md: per-model averages across categories.
- result_analysis/bfcl_emotion_by_category_summary.md: per-model, per-category (simple/multiple/parallel/parallel_multiple).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .fantom_emotion_impacts import read_summary_overall, compute_emotion_deltas


ROOT = Path(__file__).resolve().parents[1]
BFCL_DIR = ROOT / "results" / "bfcl" / "live"
OUT_MODEL_MD = ROOT / "result_analysis" / "bfcl_emotion_summary.md"
OUT_CATEGORY_MD = ROOT / "result_analysis" / "bfcl_emotion_by_category_summary.md"


@dataclass
class RunInfo:
    path: Path
    model: str
    task_type: str  # e.g., live_simple, live_parallel, live_parallel_multiple, live_multiple


def discover_runs(base: Path) -> List[RunInfo]:
    runs: List[RunInfo] = []
    if not base.exists():
        return runs
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        if not (p / "summary_overall.csv").exists():
            continue
        name = p.name
        if "_bfcl_" not in name:
            continue
        model = name.split("_bfcl_", 1)[0]
        task_type = "unknown"
        cfg_path = p / "experiment_config.json"
        try:
            cfg = json.loads(cfg_path.read_text())
            task_type = cfg.get("benchmark", {}).get("task_type", "unknown")
        except Exception:
            # fallback: try to parse after last `_bfcl_`
            suffix = name.split("_bfcl_", 1)[1]
            # tokens like live_simple_2025... -> take first two parts
            parts = suffix.split("_")
            task_type = "_".join(parts[:2]) if len(parts) >= 2 else parts[0]
        runs.append(RunInfo(p, model, task_type))
    return runs


def parse_bfcl_category(task_type: str) -> str:
    """Map BFCL task_type (e.g., 'live_parallel_multiple') to a short category name.

    Returns one of: 'simple', 'multiple', 'parallel', 'parallel_multiple', or 'unknown'.
    """
    if not task_type:
        return "unknown"
    if task_type.startswith("live_"):
        cat = task_type[len("live_"):]
        return cat or "unknown"
    return task_type


def aggregate_by_model(runs: Iterable[RunInfo]) -> Dict[str, Dict[str, float]]:
    per_model_neutral: Dict[str, List[float]] = {}
    per_model_deltas: Dict[str, Dict[str, List[float]]] = {}
    for r in runs:
        rows = read_summary_overall(r.path)
        per_model_neutral.setdefault(r.model, []).append(rows["neutral"].mean_of_means)
        deltas = compute_emotion_deltas(r.path)
        md = per_model_deltas.setdefault(r.model, {})
        for emo, d in deltas.items():
            md.setdefault(emo, []).append(d)
    out: Dict[str, Dict[str, float]] = {}
    for model, arr in per_model_neutral.items():
        stats: Dict[str, float] = {"neutral_mean_avg": sum(arr) / len(arr)}
        for emo, xs in per_model_deltas.get(model, {}).items():
            if xs:
                stats[f"delta_{emo}"] = sum(xs) / len(xs)
        out[model] = stats
    return out


def aggregate_by_model_and_category(runs: Iterable[RunInfo]) -> Dict[str, Dict[str, Dict[str, float]]]:
    neutrals: Dict[Tuple[str, str], List[float]] = {}
    deltas: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    for r in runs:
        rows = read_summary_overall(r.path)
        cat = parse_bfcl_category(r.task_type)
        key = (r.model, cat)
        neutrals.setdefault(key, []).append(rows["neutral"].mean_of_means)
        ds = compute_emotion_deltas(r.path)
        dmap = deltas.setdefault(key, {})
        for emo, d in ds.items():
            dmap.setdefault(emo, []).append(d)
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for (model, cat), arr in neutrals.items():
        stats: Dict[str, float] = {"neutral_mean_avg": sum(arr) / len(arr)}
        for emo, xs in deltas.get((model, cat), {}).items():
            if xs:
                stats[f"delta_{emo}"] = sum(xs) / len(xs)
        out.setdefault(model, {})[cat] = stats
    return out


def _fmt_pp(x: float) -> str:
    return f"{x*100:.2f} pp" if abs(x) >= 0.0005 else "0.00 pp"


def _references_section(runs: List[RunInfo]) -> str:
    lines: List[str] = []
    lines.append("References (run directories)")
    lines.append(f"- Base: {BFCL_DIR}")
    for r in runs:
        lines.append(f"- {r.model} | {r.task_type} | {r.path}")
    return "\n".join(lines)


def build_model_markdown(per_model: Dict[str, Dict[str, float]], runs: List[RunInfo]) -> str:
    lines: List[str] = []
    lines.append("# BFCL (Live) Emotion Impact by Model")
    lines.append("")
    lines.append("Last Updated: 2025-10-01")
    lines.append("")
    lines.append("Scope")
    lines.append(f"- Inputs: all `summary_overall.csv` under `{BFCL_DIR}`.")
    lines.append("- Metric: mean_of_means per emotion; deltas vs neutral (pp); averages across BFCL live categories per model.")
    lines.append("")
    for model in sorted(per_model.keys()):
        agg = per_model[model]
        lines.append(f"## {model}")
        lines.append(f"- Neutral baseline (avg across categories): {agg.get('neutral_mean_avg', 0.0)*100:.2f}%")
        for emo in ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]:
            k = f"delta_{emo}"
            if k in agg:
                lines.append(f"- {emo.title()}: {_fmt_pp(agg[k])}")
        lines.append("")
    lines.append("")
    lines.append(_references_section(runs))
    return "\n".join(lines)


def build_category_markdown(per_model_cat: Dict[str, Dict[str, Dict[str, float]]], runs: List[RunInfo]) -> str:
    lines: List[str] = []
    lines.append("# BFCL (Live) Emotion Impact by Category")
    lines.append("")
    lines.append("Last Updated: 2025-10-01")
    lines.append("")
    lines.append("Scope")
    lines.append("- Grouping: per model, per BFCL live category: simple, multiple, parallel, parallel_multiple.")
    lines.append("- Metric: mean_of_means per emotion; deltas vs neutral (pp); averages across runs per model+category.")
    lines.append("")
    # Index run paths per model+category for inline references
    paths_idx: Dict[str, Dict[str, List[str]]] = {}
    for r in runs:
        cat = parse_bfcl_category(r.task_type)
        paths_idx.setdefault(r.model, {}).setdefault(cat, []).append(str(r.path))

    for model in sorted(per_model_cat.keys()):
        lines.append(f"## {model}")
        fams = per_model_cat[model]
        for cat in ["simple", "multiple", "parallel", "parallel_multiple"]:
            if cat not in fams:
                continue
            stats = fams[cat]
            ref_paths = ", ".join(paths_idx.get(model, {}).get(cat, []))
            if ref_paths:
                lines.append(f"- {cat}: {ref_paths}")
            else:
                lines.append(f"- {cat}:")
            lines.append(f"  - Neutral baseline: {stats.get('neutral_mean_avg', 0.0)*100:.2f}%")
            for emo in ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]:
                k = f"delta_{emo}"
                if k in stats:
                    lines.append(f"  - {emo.title()}: {_fmt_pp(stats[k])}")
        lines.append("")
    lines.append("")
    lines.append(_references_section(runs))
    return "\n".join(lines)


def main() -> None:
    runs = discover_runs(BFCL_DIR)
    if not runs:
        raise SystemExit(f"No runs found under {BFCL_DIR}")
    per_model = aggregate_by_model(runs)
    per_model_cat = aggregate_by_model_and_category(runs)
    OUT_MODEL_MD.write_text(build_model_markdown(per_model, runs))
    OUT_CATEGORY_MD.write_text(build_category_markdown(per_model_cat, runs))
    print(f"Wrote {OUT_MODEL_MD}")
    print(f"Wrote {OUT_CATEGORY_MD}")


if __name__ == "__main__":
    main()
