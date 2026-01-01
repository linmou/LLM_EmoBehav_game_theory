"""Heatmap generation for game-theory impact reports.

Reads `summary_choice_ratio.csv` / `summary_behavior_ratio.csv` and renders Δ vs neutral heatmaps.
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple


def _load_collapsed_choice_ratios(path: Path) -> Dict[str, Dict[int, float]]:
    acc: Dict[str, Dict[int, List[float]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            emotion = str(row["emotion"])
            option_id = int(float(row["option_id"]))
            ratio = float(row["ratio"])
            acc.setdefault(emotion, {}).setdefault(option_id, []).append(ratio)
    return {emo: {opt: mean(vals) for opt, vals in m.items()} for emo, m in acc.items()}


def _load_collapsed_behavior_ratios(path: Path) -> Dict[str, Dict[str, float]]:
    acc: Dict[str, Dict[str, List[float]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            emotion = str(row["emotion"])
            behavior = row.get("behavior")
            if behavior in (None, "", "nan", "NaN"):
                behavior = row.get("behavior_label")
            if behavior in (None, ""):
                raise KeyError("Expected CSV column 'behavior' or 'behavior_label'")
            ratio = float(row["ratio"])
            acc.setdefault(emotion, {}).setdefault(str(behavior), []).append(ratio)
    return {emo: {b: mean(vals) for b, vals in m.items()} for emo, m in acc.items()}


def _safe_filename(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "unnamed"


def _render_heatmap(
    *,
    out_path: Path,
    title: str,
    row_labels: List[str],
    col_labels: List[str],
    values: List[List[float]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = max(1, len(row_labels))
    n_cols = max(1, len(col_labels))
    fig_w = min(22.0, max(6.0, 0.55 * n_cols))
    fig_h = min(14.0, max(3.5, 0.35 * n_rows))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)

    vmax = 0.0
    for row in values:
        for v in row:
            vmax = max(vmax, abs(float(v)))
    vmax = vmax or 1.0

    im = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(list(range(len(col_labels))))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(list(range(len(row_labels))))
    ax.set_yticklabels(row_labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Δ vs neutral")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def write_delta_heatmaps(
    *,
    model: str,
    task: str,
    choice_csv: Path | None,
    behavior_csv: Path | None,
    out_dir: Path,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    prefix = f"{_safe_filename(model)}__{_safe_filename(task)}"

    if choice_csv is not None and choice_csv.exists():
        collapsed = _load_collapsed_choice_ratios(choice_csv)
        if "neutral" in collapsed:
            options = sorted({o for emo in collapsed for o in collapsed[emo]})
            emotions = sorted(e for e in collapsed if e != "neutral")
            base = collapsed["neutral"]
            for o in options:
                base.setdefault(o, 0.0)
            for emo in emotions:
                for o in options:
                    collapsed[emo].setdefault(o, 0.0)
            values = [[collapsed[emo][o] - base[o] for o in options] for emo in emotions]
            out_path = out_dir / f"{prefix}__option_delta_heatmap.png"
            _render_heatmap(
                out_path=out_path,
                title=f"{model} / {task} (Option Δ vs neutral)",
                row_labels=emotions,
                col_labels=[str(o) for o in options],
                values=values,
            )
            written.append(out_path)

    if behavior_csv is not None and behavior_csv.exists():
        collapsed = _load_collapsed_behavior_ratios(behavior_csv)
        if "neutral" in collapsed:
            behaviors = sorted({b for emo in collapsed for b in collapsed[emo]})
            emotions = sorted(e for e in collapsed if e != "neutral")
            base = collapsed["neutral"]
            for b in behaviors:
                base.setdefault(b, 0.0)
            for emo in emotions:
                for b in behaviors:
                    collapsed[emo].setdefault(b, 0.0)
            values = [[collapsed[emo][b] - base[b] for b in behaviors] for emo in emotions]
            out_path = out_dir / f"{prefix}__behavior_delta_heatmap.png"
            _render_heatmap(
                out_path=out_path,
                title=f"{model} / {task} (Behavior Δ vs neutral)",
                row_labels=emotions,
                col_labels=behaviors,
                values=values,
            )
            written.append(out_path)

    return written
