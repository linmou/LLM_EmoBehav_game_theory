"""
Generate a concise markdown summary of emotion impacts for all runs under
`results/fantom/`, mirroring the style of qwen3_emotion_game_summary.md.

Method
- Uses `summary_overall.csv` per run to get mean_of_means per emotion.
- Computes delta(emotion) = mean(emotion) - mean(neutral) for that run.
- Aggregates per model (prefix before `_fantom_`) by simple average across runs.
- Also reports accessible vs inaccessible splits when available.

Output
- Writes `result_analysis/fantom_emotion_summary.md`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

from .fantom_emotion_impacts import read_summary_overall, compute_emotion_deltas


ROOT = Path(__file__).resolve().parents[1]
FANTOM_DIR = ROOT / "results" / "fantom"
OUT_MD = ROOT / "result_analysis" / "fantom_emotion_summary.md"


@dataclass
class RunInfo:
    path: Path
    model: str
    task_type: str


def discover_runs(base: Path) -> List[RunInfo]:
    runs: List[RunInfo] = []
    if not base.exists():
        return runs
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        if not (p / "summary_overall.csv").exists():
            continue
        # model prefix before `_fantom_`
        name = p.name
        if "_fantom_" not in name:
            # Skip non-standard entries
            continue
        model = name.split("_fantom_", 1)[0]
        # task_type from config for reliability
        cfg_path = p / "experiment_config.json"
        task_type = "unknown"
        try:
            cfg = json.loads(cfg_path.read_text())
            task_type = cfg.get("benchmark", {}).get("task_type", "unknown")
        except Exception:
            # fallback to the name suffix after `_fantom_`
            task_type = name.split("_fantom_", 1)[1]
        runs.append(RunInfo(p, model, task_type))
    return runs


def parse_task_family(task_type: str) -> str:
    """Extract the coarse task family from a task_type string.

    Examples:
      - full_answerability_binary_accessible -> answerability
      - short_fact -> fact
      - full_infoaccessibility_list_inaccessible -> infoaccessibility
      - short_belief_choice_inaccessible -> belief_choice
    """
    if not task_type:
        return "unknown"
    first = task_type
    # drop the "full_" / "short_" prefix if present
    for pref in ("full_", "short_"):
        if first.startswith(pref):
            first = first[len(pref):]
            break
    # family is the first token, except for two-word 'belief_choice'
    parts = first.split("_")
    if parts[:2] == ["belief", "choice"]:
        return "belief_choice"
    return parts[0]


def aggregate_by_model(runs: Iterable[RunInfo]) -> Dict[str, Dict[str, float]]:
    """Return per-model aggregates: neutral_mean_avg and avg delta per emotion.

    Aggregation is a simple arithmetic mean across runs for the same model.
    """
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
    for model, neutrals in per_model_neutral.items():
        agg: Dict[str, float] = {}
        agg["neutral_mean_avg"] = sum(neutrals) / len(neutrals)
        for emo, arr in per_model_deltas.get(model, {}).items():
            if not arr:
                continue
            agg[f"delta_{emo}"] = sum(arr) / len(arr)
        out[model] = agg
    return out


def aggregate_by_model_and_task(runs: Iterable[RunInfo]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Aggregate per model and per task family.

    Returns: model -> task_family -> metrics dict like aggregate_by_model.
    """
    # Collect neutral means and deltas per (model, family)
    neutrals: Dict[Tuple[str, str], List[float]] = {}
    deltas: Dict[Tuple[str, str], Dict[str, List[float]]] = {}

    for r in runs:
        rows = read_summary_overall(r.path)
        fam = parse_task_family(r.task_type)
        key = (r.model, fam)
        neutrals.setdefault(key, []).append(rows["neutral"].mean_of_means)
        ds = compute_emotion_deltas(r.path)
        dd = deltas.setdefault(key, {})
        for emo, d in ds.items():
            dd.setdefault(emo, []).append(d)

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for (model, fam), arr in neutrals.items():
        fam_stats: Dict[str, float] = {"neutral_mean_avg": sum(arr) / len(arr)}
        for emo, xs in deltas.get((model, fam), {}).items():
            if xs:
                fam_stats[f"delta_{emo}"] = sum(xs) / len(xs)
        out.setdefault(model, {})[fam] = fam_stats
    return out


def aggregate_accessibility_splits(runs: Iterable[RunInfo]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Per-model aggregates split by accessible vs inaccessible.

    Returns: model -> { 'accessible'| 'inaccessible' : { 'neutral_mean_avg', 'delta_emo': avg } }
    """
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    buckets = {"accessible": [], "inaccessible": []}
    # collect buckets per model
    per_model_runs: Dict[str, Dict[str, List[RunInfo]]] = {}
    for r in runs:
        key = "inaccessible" if "inaccessible" in r.task_type else ("accessible" if "accessible" in r.task_type else None)
        if key is None:
            continue
        per_model_runs.setdefault(r.model, {}).setdefault(key, []).append(r)

    for model, splits in per_model_runs.items():
        out[model] = {}
        for key, rs in splits.items():
            agg = aggregate_by_model(rs).get(model, {})
            out[model][key] = agg
    return out


def fmt_pp(x: float) -> str:
    return f"{x*100:.2f} pp" if abs(x) >= 0.0005 else "0.00 pp"


def _references_section(runs: List[RunInfo]) -> str:
    lines: List[str] = []
    lines.append("References (run directories)")
    lines.append(f"- Base: {FANTOM_DIR}")
    for r in runs:
        lines.append(f"- {r.model} | {r.task_type} | {r.path}")
    return "\n".join(lines)


def build_markdown(per_model: Dict[str, Dict[str, float]], splits: Dict[str, Dict[str, Dict[str, float]]], runs: List[RunInfo]) -> str:
    lines: List[str] = []
    lines.append("# Fantom Emotion Impact by Model")
    lines.append("")
    lines.append("Scope")
    lines.append("- Inputs: all `summary_overall.csv` under `results/fantom/`.")
    lines.append("- Metric: mean_of_means score per emotion; deltas vs neutral in percentage points (pp).")
    lines.append("- Aggregation: simple averages across all fantom task variants per model.")
    lines.append("")
    for model in sorted(per_model.keys()):
        agg = per_model[model]
        lines.append(f"## {model}")
        neutral = agg.get("neutral_mean_avg", 0.0)
        lines.append(f"- Neutral baseline (avg across tasks): {neutral*100:.2f}%")
        # ordered emotions for readability
        for emo in ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]:
            key = f"delta_{emo}"
            if key in agg:
                lines.append(f"- {emo.title()}: {fmt_pp(agg[key])}")
        if model in splits and any(splits[model].values()):
            lines.append("- Accessibility split (avg deltas):")
            for key in ["accessible", "inaccessible"]:
                if key not in splits[model]:
                    continue
                sagg = splits[model][key]
                parts = []
                for emo in ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]:
                    k = f"delta_{emo}"
                    if k in sagg:
                        parts.append(f"{emo}:{fmt_pp(sagg[k])}")
                if parts:
                    lines.append(f"  - {key}: " + ", ".join(parts))
        lines.append("")
    lines.append("")
    lines.append(_references_section(runs))
    return "\n".join(lines)


def main() -> None:
    runs = discover_runs(FANTOM_DIR)
    if not runs:
        raise SystemExit(f"No runs found under {FANTOM_DIR}")
    per_model = aggregate_by_model(runs)
    splits = aggregate_accessibility_splits(runs)
    md = build_markdown(per_model, splits, runs)
    OUT_MD.write_text(md)
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
