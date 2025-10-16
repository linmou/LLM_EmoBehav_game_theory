"""
Generate per-task-family emotion impact summaries for Fantom runs.

Uses: discover_runs, aggregate_by_model_and_task from generate_fantom_emotion_summary.
Outputs: result_analysis/fantom_emotion_by_task_summary.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .generate_fantom_emotion_summary import (
    discover_runs,
    aggregate_by_model_and_task,
    FANTOM_DIR,
)


OUT_MD = Path(__file__).resolve().parents[1] / "result_analysis" / "fantom_emotion_by_task_summary.md"


def fmt_pp(x: float) -> str:
    return f"{x*100:.2f} pp" if abs(x) >= 0.0005 else "0.00 pp"


def _references_section(runs):
    lines = ["References (run directories)", f"- Base: {FANTOM_DIR}"]
    for r in runs:
        lines.append(f"- {r.model} | {r.task_type} | {r.path}")
    return "\n".join(lines)


def build_markdown(per_model_task: Dict[str, Dict[str, Dict[str, float]]], runs) -> str:
    lines: List[str] = []
    lines.append("# Fantom Emotion Impact by Task Family")
    lines.append("")
    lines.append("Last Updated: 2025-10-01")
    lines.append("")
    lines.append("Scope")
    lines.append("- Inputs: all `summary_overall.csv` under `results/fantom/`.")
    lines.append("- Grouping: per model, per task family (answerability, fact, infoaccessibility, belief_choice).")
    lines.append("- Metric: mean_of_means per emotion; deltas vs neutral (pp); averages across runs per model+family.")
    lines.append("")

    for model in sorted(per_model_task.keys()):
        lines.append(f"## {model}")
        fams = per_model_task[model]
        for fam in ["answerability", "fact", "infoaccessibility", "belief_choice"]:
            if fam not in fams:
                continue
            stats = fams[fam]
            lines.append(f"- {fam}:")
            neutral = stats.get("neutral_mean_avg", 0.0)
            lines.append(f"  - Neutral baseline: {neutral*100:.2f}%")
            for emo in ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]:
                k = f"delta_{emo}"
                if k in stats:
                    lines.append(f"  - {emo.title()}: {fmt_pp(stats[k])}")
        lines.append("")
    lines.append("")
    lines.append(_references_section(runs))
    return "\n".join(lines)


def main() -> None:
    runs = discover_runs(FANTOM_DIR)
    if not runs:
        raise SystemExit(f"No runs found under {FANTOM_DIR}")
    per_model_task = aggregate_by_model_and_task(runs)
    md = build_markdown(per_model_task, runs)
    OUT_MD.write_text(md)
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
