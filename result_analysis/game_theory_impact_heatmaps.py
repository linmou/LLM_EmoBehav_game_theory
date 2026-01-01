"""Heatmap generation for game-theory impact reports.

Generates one behavior-direction heatmap PDF per `task` (game_setting), where each cell is
the change in target-behavior share vs neutral:

    Δ = P(target_behavior | emotion) - P(target_behavior | neutral)

Target behavior is chosen by convention:
- If the game is binary (two non-unknown behaviors), prefer `defect` / `escalation` (else fallback to 2nd label).
- Otherwise, prefer `offer_none` or `reject` (else fallback to last label).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from result_analysis.game_theory_ratio_loading import load_behavior_by_intensity


def _safe_filename(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "unnamed"


def _choose_target_behaviors(behaviors: Sequence[str]) -> List[str]:
    norm = {b.strip().lower() for b in behaviors if isinstance(b, str)}
    non_unknown = sorted(b for b in norm if b and b != "unknown")

    is_binary = len(non_unknown) == 2
    if is_binary:
        for cand in ("defect", "escalation", "escalate"):
            if cand in norm:
                return [cand]
        return [non_unknown[-1]]

    for cand in ("offer_none", "reject"):
        if cand in norm:
            return [cand]
    return [non_unknown[-1]] if non_unknown else []


def _render_heatmap_pdf(
    *,
    out_path: Path,
    title: str,
    row_labels: List[str],
    col_labels: List[str],
    values: List[List[float]],
    norm_mode: str,
    symlog_linthresh: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    n_rows = max(1, len(row_labels))
    n_cols = max(1, len(col_labels))
    fig_w = min(22.0, max(6.0, 0.55 * n_cols))
    fig_h = min(14.0, max(3.5, 0.35 * n_rows))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)

    norm, vmin, vmax = _build_heatmap_norm(values, mode=norm_mode, linthresh=symlog_linthresh)
    if norm is None:
        im = ax.imshow(values, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    else:
        im = ax.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(list(range(len(col_labels))))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(list(range(len(row_labels))))
    ax.set_yticklabels(row_labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Δ vs neutral (target share)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

def _build_heatmap_norm(values: List[List[float]], *, mode: str, linthresh: float) -> tuple[object, float, float]:
    vmax = 0.0
    for row in values:
        for v in row:
            vmax = max(vmax, abs(float(v)))
    vmax = vmax or 1.0
    vmin = -vmax
    if mode == "linear":
        return None, vmin, vmax
    if mode == "symlog":
        from matplotlib.colors import SymLogNorm

        lt = float(linthresh)
        if lt <= 0:
            raise ValueError("symlog linthresh must be > 0")
        return SymLogNorm(linthresh=lt, vmin=vmin, vmax=vmax, base=10), vmin, vmax
    raise ValueError(f"Unknown heatmap norm mode: {mode}")


def write_behavior_change_heatmap_pdf(
    *,
    task: str,
    model_to_behavior_csv: Dict[str, Path],
    out_dir: Path,
    unknown_threshold: Optional[float],
    heatmap_norm: str,
    symlog_linthresh: float,
) -> Optional[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    models, emotions, values, _ = compute_peak_behavior_change_matrix(
        task=task, model_to_behavior_csv=model_to_behavior_csv, unknown_threshold=unknown_threshold
    )
    if not models or not emotions:
        return None

    out_path = out_dir / f"behavior_change_heatmap__{_safe_filename(task)}.pdf"
    _render_heatmap_pdf(
        out_path=out_path,
        title=f"{task} (behavior-direction peak |Δ| vs neutral)",
        row_labels=models,
        col_labels=emotions,
        values=values,
        norm_mode=heatmap_norm,
        symlog_linthresh=symlog_linthresh,
    )
    return out_path


def compute_peak_behavior_change_matrix(
    *,
    task: str,
    model_to_behavior_csv: Dict[str, Path],
    unknown_threshold: Optional[float],
) -> tuple[list[str], list[str], list[list[float]], dict[tuple[str, str], float]]:
    """Return (models, emotions, values, chosen_intensity) for peak-|delta| per (model, emotion)."""

    models = sorted(model_to_behavior_csv)
    if not models:
        return [], [], [], {}

    by_model: Dict[str, Dict[str, Dict[float, Dict[str, float]]]] = {}
    all_behaviors: set[str] = set()
    all_emotions: set[str] = set()
    for model in models:
        by_intensity, _ = load_behavior_by_intensity(model_to_behavior_csv[model], unknown_threshold=unknown_threshold)
        by_model[model] = by_intensity
        all_emotions |= set(by_intensity)
        for emo_map in by_intensity.values():
            for per_int in emo_map.values():
                all_behaviors |= {str(b) for b in per_int}

    emotions = sorted(e for e in all_emotions if e != "neutral")
    targets = _choose_target_behaviors(sorted(all_behaviors))
    if not emotions or not targets:
        return models, emotions, [[0.0 for _ in emotions] for _ in models], {}

    chosen: Dict[tuple[str, str], float] = {}
    values: List[List[float]] = []
    for model in models:
        by_intensity = by_model[model]
        neutral_per_int = by_intensity.get("neutral", {})
        neutral_ints = sorted(neutral_per_int)
        if neutral_ints:
            neutral_base = sum(
                sum(float(neutral_per_int[i].get(t, 0.0)) for t in targets) for i in neutral_ints
            ) / len(neutral_ints)
        else:
            neutral_base = 0.0

        row: List[float] = []
        for emo in emotions:
            emo_per_int = by_intensity.get(emo, {})
            best_delta: Optional[float] = None
            best_intensity: Optional[float] = None
            for intensity, m in emo_per_int.items():
                emo_target = sum(float(m.get(t, 0.0)) for t in targets)
                delta = emo_target - neutral_base
                if best_delta is None or abs(delta) > abs(best_delta) or (abs(delta) == abs(best_delta) and intensity < best_intensity):  # type: ignore[operator]
                    best_delta = delta
                    best_intensity = float(intensity)
            row.append(float(best_delta) if best_delta is not None else 0.0)
            if best_intensity is not None:
                chosen[(model, emo)] = best_intensity
        values.append(row)

    return models, emotions, values, chosen
