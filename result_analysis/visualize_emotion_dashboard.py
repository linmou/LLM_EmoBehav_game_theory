#!/usr/bin/env python3
"""
Emotion impact dashboard generator for game-theory runs.

Use-case:
- Point this at an experiment directory containing many run subdirectories
  (each with raw_results.json + summary_behavior_ratio.csv, like
  results/new_game_theory_decision/shuffle_choices/)
- It writes per-run plots (debug view) and a cross-run summary (comparison view).

Design: keep it dumb and file-driven. No refactors, no new dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


NEUTRAL_EMOTION = "neutral"
NEUTRAL_INTENSITY = 0.0


def discover_run_dirs(input_path: str | Path) -> List[Path]:
    root = Path(input_path)
    if (root / "raw_results.json").exists() or (root / "summary_behavior_ratio.csv").exists():
        return [root]

    run_dirs: List[Path] = []
    for candidate in root.rglob("*"):
        if not candidate.is_dir():
            continue
        if (candidate / "raw_results.json").exists() and (candidate / "experiment_config.json").exists():
            run_dirs.append(candidate)
            continue
        if (candidate / "summary_behavior_ratio.csv").exists() and (candidate / "experiment_config.json").exists():
            run_dirs.append(candidate)
    run_dirs.sort()
    return run_dirs


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _kl_div(p: np.ndarray, q: np.ndarray) -> float:
    # p, q are probability vectors; handle zeros safely.
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    # Jensen-Shannon divergence (base e). Well-defined with zeros if we avoid log(0).
    m = 0.5 * (p + q)
    return 0.5 * _kl_div(p, m) + 0.5 * _kl_div(q, m)


def compute_js_divergence(
    df_behavior_ratio: pd.DataFrame,
    *,
    neutral_emotion: str = NEUTRAL_EMOTION,
    neutral_intensity: float = NEUTRAL_INTENSITY,
) -> pd.DataFrame:
    df = df_behavior_ratio.copy()
    required = {"emotion", "intensity", "behavior_label", "ratio"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"summary_behavior_ratio missing columns: {sorted(missing)}")

    df["emotion"] = df["emotion"].astype(str)
    df["intensity"] = df["intensity"].astype(float)
    df["behavior_label"] = df["behavior_label"].astype(str)
    df["ratio"] = df["ratio"].astype(float)

    behaviors = sorted(df["behavior_label"].unique().tolist())

    neutral = df[(df["emotion"] == neutral_emotion) & (df["intensity"] == float(neutral_intensity))]
    if neutral.empty:
        raise ValueError("No neutral baseline found in summary_behavior_ratio.csv")

    neutral_vec = (
        neutral.set_index("behavior_label")["ratio"]
        .reindex(behaviors)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    neutral_vec = neutral_vec / max(neutral_vec.sum(), 1e-12)

    rows: List[Dict[str, Any]] = []
    for (emotion, intensity), sub in df.groupby(["emotion", "intensity"], sort=True):
        vec = (
            sub.set_index("behavior_label")["ratio"]
            .reindex(behaviors)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        vec = vec / max(vec.sum(), 1e-12)
        js = 0.0 if (emotion == neutral_emotion and float(intensity) == float(neutral_intensity)) else _js_divergence(vec, neutral_vec)
        rows.append({"emotion": emotion, "intensity": float(intensity), "js_divergence": float(js)})

    return pd.DataFrame(rows)


def compute_majority_behavior_effects(
    df_behavior_ratio: pd.DataFrame,
    *,
    neutral_emotion: str = NEUTRAL_EMOTION,
    neutral_intensity: float = NEUTRAL_INTENSITY,
) -> pd.DataFrame:
    df = df_behavior_ratio.copy()
    df["emotion"] = df["emotion"].astype(str)
    df["intensity"] = df["intensity"].astype(float)
    df["behavior_label"] = df["behavior_label"].astype(str)
    df["ratio"] = df["ratio"].astype(float)

    neutral = df[(df["emotion"] == neutral_emotion) & (df["intensity"] == float(neutral_intensity))]
    if neutral.empty:
        raise ValueError("No neutral baseline found in summary_behavior_ratio.csv")

    neutral_majority_behavior = str(neutral.sort_values("ratio", ascending=False).iloc[0]["behavior_label"])
    neutral_majority_ratio = float(neutral[neutral["behavior_label"] == neutral_majority_behavior]["ratio"].iloc[0])

    rows: List[Dict[str, Any]] = []
    for (emotion, intensity), sub in df.groupby(["emotion", "intensity"], sort=True):
        ratio = sub[sub["behavior_label"] == neutral_majority_behavior]["ratio"]
        cond_ratio = float(ratio.iloc[0]) if not ratio.empty else 0.0
        rows.append(
            {
                "emotion": emotion,
                "intensity": float(intensity),
                "neutral_majority_behavior": neutral_majority_behavior,
                "neutral_majority_ratio": neutral_majority_ratio,
                "majority_ratio": cond_ratio,
                "delta_majority_ratio": cond_ratio - neutral_majority_ratio,
            }
        )

    return pd.DataFrame(rows)


def compute_invalid_rates_from_raw_results(raw_path: str | Path) -> pd.DataFrame:
    rows = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    payload = []
    for r in rows:
        emotion = str(r.get("emotion", ""))
        intensity = float(r.get("intensity", 0.0) or 0.0)
        score = r.get("score")
        error = r.get("error")
        invalid = False
        if error:
            invalid = True
        elif score is None:
            invalid = True
        else:
            try:
                invalid = math.isnan(float(score))
            except Exception:
                invalid = True
        payload.append({"emotion": emotion, "intensity": intensity, "invalid": int(invalid)})

    df = pd.DataFrame(payload)
    if df.empty:
        return pd.DataFrame(columns=["emotion", "intensity", "invalid_count", "total_count", "invalid_rate"])

    out = (
        df.groupby(["emotion", "intensity"], as_index=False)
        .agg(invalid_count=("invalid", "sum"), total_count=("invalid", "size"))
    )
    out["invalid_rate"] = out["invalid_count"] / out["total_count"].clip(lower=1)
    return out


def _plot_heatmap(
    data: pd.DataFrame,
    *,
    index: str,
    columns: str,
    values: str,
    title: str,
    out_path: Path,
    fmt: str = ".3f",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    pivot = data.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
    plt.figure(figsize=(max(6, 0.6 * (pivot.shape[1] + 1)), max(4, 0.4 * (pivot.shape[0] + 3))))
    ax = sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_intensity_curves(
    df_behavior_ratio: pd.DataFrame,
    *,
    title: str,
    out_path: Path,
    max_behaviors: int = 6,
) -> None:
    df = df_behavior_ratio.copy()
    df["emotion"] = df["emotion"].astype(str)
    df["intensity"] = df["intensity"].astype(float)
    df["behavior_label"] = df["behavior_label"].astype(str)
    df["ratio"] = df["ratio"].astype(float)

    behaviors = (
        df[df["emotion"] != NEUTRAL_EMOTION]
        .groupby("behavior_label")["ratio"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    behaviors = behaviors[:max_behaviors]
    df = df[df["behavior_label"].isin(behaviors)]

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="intensity", y="ratio", hue="emotion", style="behavior_label", markers=True, dashes=False)
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


@dataclass(frozen=True)
class RunMeta:
    run_dir: Path
    model_name: str
    benchmark: str
    task: str
    intensities: List[float]
    emotions: List[str]


def _load_run_meta(run_dir: Path) -> RunMeta:
    cfg = json.loads((run_dir / "experiment_config.json").read_text(encoding="utf-8"))
    model_path = str(cfg.get("model_path", ""))
    model_name = Path(model_path).name if model_path else run_dir.name.split("_")[0]
    bench = cfg.get("benchmark") or {}
    benchmark = str(bench.get("name", ""))
    task = str(bench.get("task_type", ""))
    intensities = [float(x) for x in (cfg.get("intensities") or [])]
    emotions = [str(x) for x in (cfg.get("emotions") or [])]
    return RunMeta(run_dir=run_dir, model_name=model_name, benchmark=benchmark, task=task, intensities=intensities, emotions=emotions)


def _load_behavior_ratio(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "summary_behavior_ratio.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


def render_single_run(run: RunMeta, *, out_dir: Path, skip_raw: bool) -> Dict[str, Any]:
    _safe_mkdir(out_dir)
    df_ratio = _load_behavior_ratio(run.run_dir)
    df_ratio.to_csv(out_dir / "behavior_ratio.csv", index=False)

    js = compute_js_divergence(df_ratio)
    maj = compute_majority_behavior_effects(df_ratio)
    js.to_csv(out_dir / "js_divergence.csv", index=False)
    maj.to_csv(out_dir / "majority_behavior_effects.csv", index=False)

    # Heatmaps: ratios per intensity (use all, but keep filenames stable)
    for intensity in sorted(df_ratio["intensity"].unique().tolist()):
        sub = df_ratio[df_ratio["intensity"] == float(intensity)]
        _plot_heatmap(
            sub,
            index="emotion",
            columns="behavior_label",
            values="ratio",
            title=f"{run.model_name} | {run.task} | behavior ratios @ intensity={intensity}",
            out_path=out_dir / f"behavior_ratio_heatmap_i{intensity}.png",
            fmt=".3f",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )

    # Delta vs neutral for each intensity
    neutral = df_ratio[(df_ratio["emotion"] == NEUTRAL_EMOTION) & (df_ratio["intensity"] == NEUTRAL_INTENSITY)]
    if not neutral.empty:
        neutral_map = neutral.set_index("behavior_label")["ratio"].to_dict()
        delta_rows = []
        for _, r in df_ratio.iterrows():
            delta_rows.append(
                {
                    "emotion": r["emotion"],
                    "intensity": float(r["intensity"]),
                    "behavior_label": r["behavior_label"],
                    "delta_vs_neutral": float(r["ratio"]) - float(neutral_map.get(r["behavior_label"], 0.0)),
                }
            )
        df_delta = pd.DataFrame(delta_rows)
        df_delta.to_csv(out_dir / "delta_vs_neutral.csv", index=False)
        for intensity in sorted(df_delta["intensity"].unique().tolist()):
            sub = df_delta[df_delta["intensity"] == float(intensity)]
            _plot_heatmap(
                sub,
                index="emotion",
                columns="behavior_label",
                values="delta_vs_neutral",
                title=f"{run.model_name} | {run.task} | Δ vs neutral @ intensity={intensity}",
                out_path=out_dir / f"delta_vs_neutral_heatmap_i{intensity}.png",
                fmt=".3f",
                cmap="coolwarm",
                vmin=-0.5,
                vmax=0.5,
            )

    # JS divergence heatmap (emotion x intensity)
    _plot_heatmap(
        js,
        index="emotion",
        columns="intensity",
        values="js_divergence",
        title=f"{run.model_name} | {run.task} | JS divergence vs neutral",
        out_path=out_dir / "js_divergence_heatmap.png",
        fmt=".3f",
        cmap="magma",
        vmin=0.0,
        vmax=float(js["js_divergence"].max()) if not js.empty else None,
    )

    # Majority-behavior delta heatmap
    _plot_heatmap(
        maj,
        index="emotion",
        columns="intensity",
        values="delta_majority_ratio",
        title=f"{run.model_name} | {run.task} | Δ neutral-majority behavior",
        out_path=out_dir / "delta_majority_heatmap.png",
        fmt=".3f",
        cmap="coolwarm",
        vmin=-0.5,
        vmax=0.5,
    )

    # Intensity curves (when multiple intensities exist)
    if df_ratio["intensity"].nunique() > 1:
        _plot_intensity_curves(
            df_ratio,
            title=f"{run.model_name} | {run.task} | behavior ratios across intensity",
            out_path=out_dir / "intensity_curves.png",
        )

    invalid_df = None
    if not skip_raw:
        raw_path = run.run_dir / "raw_results.json"
        if raw_path.exists():
            invalid_df = compute_invalid_rates_from_raw_results(raw_path)
            invalid_df.to_csv(out_dir / "invalid_rates.csv", index=False)
            _plot_heatmap(
                invalid_df,
                index="emotion",
                columns="intensity",
                values="invalid_rate",
                title=f"{run.model_name} | {run.task} | invalid decision rate",
                out_path=out_dir / "invalid_rate_heatmap.png",
                fmt=".3f",
                cmap="Reds",
                vmin=0.0,
                vmax=float(invalid_df["invalid_rate"].max()) if not invalid_df.empty else None,
            )

    summary = {
        "run_dir": str(run.run_dir),
        "model_name": run.model_name,
        "benchmark": run.benchmark,
        "task": run.task,
        "neutral_majority_behavior": maj["neutral_majority_behavior"].iloc[0] if not maj.empty else None,
        "neutral_majority_ratio": float(maj["neutral_majority_ratio"].iloc[0]) if not maj.empty else None,
        "max_js_divergence": float(js["js_divergence"].max()) if not js.empty else None,
    }
    if invalid_df is not None and not invalid_df.empty:
        summary["max_invalid_rate"] = float(invalid_df["invalid_rate"].max())
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _max_intensity_per_run(df: pd.DataFrame) -> float:
    ints = sorted(set(float(x) for x in df["intensity"].tolist()))
    # Prefer non-neutral intensities if present.
    non_neutral = [x for x in ints if x != NEUTRAL_INTENSITY]
    return float(non_neutral[-1] if non_neutral else ints[-1])


def render_cross_run_summary(run_summaries: List[Dict[str, Any]], *, out_dir: Path) -> None:
    _safe_mkdir(out_dir)
    if not run_summaries:
        return
    pd.DataFrame(run_summaries).to_csv(out_dir / "runs_index.csv", index=False)


def _collect_effect_rows(
    *,
    run: RunMeta,
    df_ratio: pd.DataFrame,
    js: pd.DataFrame,
    maj: pd.DataFrame,
) -> List[Dict[str, Any]]:
    df_ratio = df_ratio.copy()
    df_ratio["emotion"] = df_ratio["emotion"].astype(str)
    df_ratio["intensity"] = df_ratio["intensity"].astype(float)
    df_ratio["behavior_label"] = df_ratio["behavior_label"].astype(str)
    df_ratio["ratio"] = df_ratio["ratio"].astype(float)

    max_intensity = _max_intensity_per_run(df_ratio)
    neutral = df_ratio[(df_ratio["emotion"] == NEUTRAL_EMOTION) & (df_ratio["intensity"] == NEUTRAL_INTENSITY)]
    neutral_majority_behavior = None
    neutral_majority_ratio = None
    if not neutral.empty:
        neutral_majority_behavior = str(neutral.sort_values("ratio", ascending=False).iloc[0]["behavior_label"])
        neutral_majority_ratio = float(neutral[neutral["behavior_label"] == neutral_majority_behavior]["ratio"].iloc[0])

    rows: List[Dict[str, Any]] = []
    # Only summarize at max intensity (plus neutral baseline for reference)
    for intensity in sorted(set([NEUTRAL_INTENSITY, max_intensity])):
        for emotion in sorted(df_ratio["emotion"].unique().tolist()):
            sub = df_ratio[(df_ratio["emotion"] == emotion) & (df_ratio["intensity"] == float(intensity))]
            if sub.empty:
                continue
            js_val = js[(js["emotion"] == emotion) & (js["intensity"] == float(intensity))]
            maj_val = maj[(maj["emotion"] == emotion) & (maj["intensity"] == float(intensity))]
            rows.append(
                {
                    "run_dir": str(run.run_dir),
                    "run_name": run.run_dir.name,
                    "model_name": run.model_name,
                    "benchmark": run.benchmark,
                    "task": run.task,
                    "emotion": emotion,
                    "intensity": float(intensity),
                    "js_divergence": float(js_val["js_divergence"].iloc[0]) if not js_val.empty else None,
                    "neutral_majority_behavior": neutral_majority_behavior,
                    "neutral_majority_ratio": neutral_majority_ratio,
                    "delta_majority_ratio": float(maj_val["delta_majority_ratio"].iloc[0]) if not maj_val.empty else None,
                }
            )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate emotion-impact dashboards from experiment directories.")
    parser.add_argument("--input", required=True, help="Experiment directory (single run dir or a root containing many runs)")
    parser.add_argument("--output", default=None, help="Output directory (default: <input>/viz_emotion_dashboard)")
    parser.add_argument("--max-runs", type=int, default=0, help="Limit number of runs processed (0 = no limit)")
    parser.add_argument("--skip-raw", action="store_true", help="Skip reading raw_results.json (faster; no invalid-rate plots)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    input_path = Path(args.input)
    out_root = Path(args.output) if args.output else (input_path / "viz_emotion_dashboard")
    _safe_mkdir(out_root)

    run_dirs = discover_run_dirs(input_path)
    if args.max_runs and args.max_runs > 0:
        run_dirs = run_dirs[: int(args.max_runs)]

    summaries: List[Dict[str, Any]] = []
    effect_rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        meta = _load_run_meta(run_dir)
        per_run_out = out_root / "runs" / run_dir.name
        summary = render_single_run(meta, out_dir=per_run_out, skip_raw=bool(args.skip_raw))
        summaries.append(summary)
        try:
            df_ratio = pd.read_csv(per_run_out / "behavior_ratio.csv")
            js = pd.read_csv(per_run_out / "js_divergence.csv")
            maj = pd.read_csv(per_run_out / "majority_behavior_effects.csv")
            effect_rows.extend(_collect_effect_rows(run=meta, df_ratio=df_ratio, js=js, maj=maj))
        except Exception:
            # Keep going; per-run dashboards are still useful even if aggregation fails.
            pass

    render_cross_run_summary(summaries, out_dir=out_root / "summary")
    if effect_rows:
        effects = pd.DataFrame(effect_rows)
        effects.to_csv(out_root / "summary" / "effects_long.csv", index=False)

        # Cross-run heatmaps per game: models x emotions (at max intensity only)
        effects_max = effects[effects["intensity"] != NEUTRAL_INTENSITY].copy()
        if not effects_max.empty:
            for task, sub in effects_max.groupby("task", sort=True):
                js_sub = sub.dropna(subset=["js_divergence"])
                if not js_sub.empty:
                    _plot_heatmap(
                        js_sub,
                        index="model_name",
                        columns="emotion",
                        values="js_divergence",
                        title=f"{task} | JS divergence vs neutral (max intensity)",
                        out_path=out_root / "summary" / f"js_heatmap_{task}.png",
                        fmt=".3f",
                        cmap="magma",
                        vmin=0.0,
                        vmax=float(js_sub["js_divergence"].max()),
                    )

                dm_sub = sub.dropna(subset=["delta_majority_ratio"])
                if not dm_sub.empty:
                    _plot_heatmap(
                        dm_sub,
                        index="model_name",
                        columns="emotion",
                        values="delta_majority_ratio",
                        title=f"{task} | Δ neutral-majority behavior (max intensity)",
                        out_path=out_root / "summary" / f"delta_majority_{task}.png",
                        fmt=".3f",
                        cmap="coolwarm",
                        vmin=-0.5,
                        vmax=0.5,
                    )

            # Model-level summary (mean JS over games+emotions)
            js_only = effects_max.dropna(subset=["js_divergence"])
            if not js_only.empty:
                model_mean = js_only.groupby("model_name", as_index=False)["js_divergence"].mean().sort_values("js_divergence", ascending=False)
                model_mean.to_csv(out_root / "summary" / "mean_js_by_model.csv", index=False)
                plt.figure(figsize=(max(8, 0.35 * len(model_mean)), 4))
                sns.barplot(data=model_mean, x="model_name", y="js_divergence")
                plt.xticks(rotation=60, ha="right")
                plt.title("Mean JS divergence vs neutral (higher = more emotion-sensitive)")
                plt.tight_layout()
                plt.savefig(out_root / "summary" / "mean_js_by_model.png", dpi=200)
                plt.close()

    print(f"Wrote dashboards to: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
