"""Generate option- and (when available) behavior-level emotion impact reports (vs neutral).

Designed to work with both:
- `results/new_game_theory_decision/shuffle_choices/` (has choice + behavior ratios)
- `results/new_game_theory/` (typically has choice ratios only)

Inputs (per run directory, when present):
- summary_choice_ratio.csv: emotion,intensity,option_id,ratio
- summary_behavior_ratio.csv: emotion,intensity,behavior_label,ratio

Method:
1) For each (model, game_setting), keep the latest timestamped run directory.
2) Ignore intensity by collapsing with mean(ratio) over intensities.
3) Compute per emotion delta vs neutral for each option_id / behavior_label:
     delta = ratio(emotion, x) - ratio(neutral, x)
4) Summarize per (model, game_setting, x) the best/worst deltas and range.

Outputs (written under root):
- option_impacted_by_emo_vs_neutral_latest.csv
- behavior_impacted_emo_vs_neutral_latest.csv (only if behavior inputs exist)
- game_theory_impact_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple


_EXPECTED_SCORE_SPECS: Dict[str, Dict[str, float]] = {
    "Trust_Game_Trustor": {"trust_none": 0.0, "trust_low": 1.0, "trust_high": 2.0},
    "Trust_Game_Trustee": {"return_none": 0.0, "return_medium": 1.0, "return_high": 2.0},
    "Ultimatum_Game_Proposer": {"offer_low": 0.0, "offer_medium": 1.0, "offer_high": 2.0},
    "Ultimatum_Game_Responder": {"reject": 0.0, "accept": 1.0},
}


_RUN_DIR_RE = re.compile(
    r"^(?P<model>.+)_game_theory(_decision)?_(?P<task>.+)_(?P<ts>\d{8}_\d{6})$"
)


@dataclass(frozen=True)
class RunRef:
    model: str
    task: str
    timestamp: str
    dir_path: Path


@dataclass(frozen=True)
class ReportOutputs:
    option_csv_path: Path
    behavior_csv_path: Optional[Path]
    report_path: Path


def _parse_run_dir(name: str) -> Optional[Tuple[str, str, str]]:
    match = _RUN_DIR_RE.match(name)
    if not match:
        return None
    return match.group("model"), match.group("task"), match.group("ts")


def _discover_latest_runs(root: Path) -> List[RunRef]:
    latest: Dict[Tuple[str, str], RunRef] = {}
    for run_dir in root.glob("**/"):
        if not run_dir.is_dir():
            continue
        parsed = _parse_run_dir(run_dir.name)
        if not parsed:
            continue
        model, task, ts = parsed
        key = (model, task)
        candidate = RunRef(model=model, task=task, timestamp=ts, dir_path=run_dir)
        if key not in latest or ts > latest[key].timestamp:
            latest[key] = candidate
    return [latest[k] for k in sorted(latest)]


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
            behavior = str(row["behavior_label"])
            ratio = float(row["ratio"])
            acc.setdefault(emotion, {}).setdefault(behavior, []).append(ratio)
    return {emo: {b: mean(vals) for b, vals in m.items()} for emo, m in acc.items()}


def _impact_rows_for_choice(run: RunRef, csv_path: Path) -> Tuple[List[Dict[str, object]], bool]:
    collapsed = _load_collapsed_choice_ratios(csv_path)
    if "neutral" not in collapsed:
        return [], True
    options = sorted({o for emo in collapsed for o in collapsed[emo]})
    emotions = sorted(collapsed)
    for emo in emotions:
        for o in options:
            collapsed[emo].setdefault(o, 0.0)
    base = collapsed["neutral"]
    rows: List[Dict[str, object]] = []
    for o in options:
        neutral_ratio = base[o]
        deltas = [(emo, collapsed[emo][o] - neutral_ratio, collapsed[emo][o]) for emo in emotions if emo != "neutral"]
        if not deltas:
            continue
        best = max(deltas, key=lambda x: x[1])
        worst = min(deltas, key=lambda x: x[1])
        rows.append(
            {
                "task": run.task,
                "model": run.model,
                "timestamp": run.timestamp,
                "option_id": o,
                "neutral_ratio": round(neutral_ratio, 6),
                "best_emotion": best[0],
                "best_delta_vs_neutral": round(best[1], 6),
                "best_ratio": round(best[2], 6),
                "worst_emotion": worst[0],
                "worst_delta_vs_neutral": round(worst[1], 6),
                "worst_ratio": round(worst[2], 6),
                "delta_range": round(best[1] - worst[1], 6),
            }
        )
    return rows, False


def _impact_rows_for_behavior(run: RunRef, csv_path: Path) -> Tuple[List[Dict[str, object]], bool]:
    collapsed = _load_collapsed_behavior_ratios(csv_path)
    if "neutral" not in collapsed:
        return [], True
    behaviors = sorted({b for emo in collapsed for b in collapsed[emo]})
    emotions = sorted(collapsed)
    for emo in emotions:
        for b in behaviors:
            collapsed[emo].setdefault(b, 0.0)
    base = collapsed["neutral"]
    rows: List[Dict[str, object]] = []
    for b in behaviors:
        neutral_ratio = base[b]
        deltas = [(emo, collapsed[emo][b] - neutral_ratio, collapsed[emo][b]) for emo in emotions if emo != "neutral"]
        if not deltas:
            continue
        best = max(deltas, key=lambda x: x[1])
        worst = min(deltas, key=lambda x: x[1])
        rows.append(
            {
                "task": run.task,
                "model": run.model,
                "timestamp": run.timestamp,
                "behavior_label": b,
                "neutral_ratio": round(neutral_ratio, 6),
                "best_emotion": best[0],
                "best_delta_vs_neutral": round(best[1], 6),
                "best_ratio": round(best[2], 6),
                "worst_emotion": worst[0],
                "worst_delta_vs_neutral": round(worst[1], 6),
                "worst_ratio": round(worst[2], 6),
                "delta_range": round(best[1] - worst[1], 6),
            }
        )
    return rows, False


def _expected_score_rows_for_run(run: RunRef, raw_path: Path) -> Tuple[List[Dict[str, object]], bool]:
    spec = _EXPECTED_SCORE_SPECS.get(run.task)
    if spec is None:
        return [], False
    data = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return [], False

    # Parse record -> (item_id, emotion, intensity, score)
    parsed: List[Tuple[object, str, float, float]] = []
    for rec in data:
        if not isinstance(rec, dict):
            continue
        if rec.get("error"):
            continue
        emotion = rec.get("emotion")
        intensity = rec.get("intensity")
        item_id = rec.get("item_id")
        if not isinstance(emotion, str) or intensity is None:
            continue
        try:
            intensity_f = float(intensity)
        except Exception:
            continue
        score_raw = rec.get("score")
        if score_raw is None:
            continue
        try:
            option_id = int(float(score_raw))
        except Exception:
            continue
        options = rec.get("metadata", {}).get("item_metadata", {}).get("options", [])
        if not isinstance(options, list):
            continue
        behavior = None
        for opt in options:
            if not isinstance(opt, dict):
                continue
            if int(opt.get("id", -1)) == option_id:
                behavior = opt.get("behavior")
                break
        if not isinstance(behavior, str):
            continue
        if behavior not in spec:
            continue
        parsed.append((item_id, emotion, intensity_f, float(spec[behavior])))

    # Neutral baselines per item.
    neutral_by_item: Dict[object, float] = {}
    neutrals: Dict[object, List[float]] = {}
    for item_id, emotion, _intensity, score in parsed:
        if emotion == "neutral":
            neutrals.setdefault(item_id, []).append(score)
    for item_id, vals in neutrals.items():
        neutral_by_item[item_id] = mean(vals)

    # Deltas per (emotion,intensity) for items that have neutral baseline.
    by_ei: Dict[Tuple[str, float], List[float]] = {}
    by_ei_scores: Dict[Tuple[str, float], List[float]] = {}
    by_ei_neutral: Dict[Tuple[str, float], List[float]] = {}
    for item_id, emotion, intensity, score in parsed:
        if emotion == "neutral":
            continue
        if item_id not in neutral_by_item:
            continue
        base = neutral_by_item[item_id]
        key = (emotion, intensity)
        by_ei.setdefault(key, []).append(score - base)
        by_ei_scores.setdefault(key, []).append(score)
        by_ei_neutral.setdefault(key, []).append(base)

    if not by_ei:
        return [], True

    rows: List[Dict[str, object]] = []
    for (emotion, intensity), deltas in sorted(by_ei.items(), key=lambda x: (x[0][0], x[0][1])):
        scores = by_ei_scores[(emotion, intensity)]
        bases = by_ei_neutral[(emotion, intensity)]
        rows.append(
            {
                "task": run.task,
                "model": run.model,
                "timestamp": run.timestamp,
                "emotion": emotion,
                "intensity": intensity,
                "delta_mean": round(mean(deltas), 6),
                "score_mean": round(mean(scores), 6),
                "neutral_score_mean": round(mean(bases), 6),
                "n_items": len(deltas),
            }
        )

    return rows, False


def _write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(
    *,
    root: Path,
    runs: List[RunRef],
    option_csv: Path,
    behavior_csv: Optional[Path],
    expected_score_csv: Optional[Path],
    option_rows: List[Dict[str, object]],
    behavior_rows: List[Dict[str, object]],
    expected_score_rows: List[Dict[str, object]],
    skipped_missing_neutral: List[Path],
    top_n: int,
    per_game_n: int,
) -> str:
    lines: List[str] = []
    lines.append("# Game-Theory Decision Impact Report (vs neutral)")
    lines.append("")
    lines.append("## Data Used")
    lines.append(f"- Root scanned: `{root}`")
    lines.append("- Input files searched: `**/summary_choice_ratio.csv`, `**/summary_behavior_ratio.csv`")
    lines.append(f"- Latest run per (model, game_setting): {len(runs)}")
    for run in runs:
        lines.append(f"  - `{run.dir_path}`")
    lines.append("")
    lines.append("## Method")
    lines.append("- For each `(model, game_setting)`, select the latest timestamped run directory.")
    lines.append("- Collapse intensity by averaging `ratio` over all intensities present.")
    lines.append("- Compute `delta_vs_neutral = ratio(emotion) - ratio(neutral)` for each option/behavior.")
    lines.append("- Summarize best/worst emotion deltas and `delta_range = best - worst`.")
    lines.append("")
    lines.append("## Outputs")
    lines.append(f"- Option CSV: `{option_csv}`")
    if behavior_csv is not None:
        lines.append(f"- Behavior CSV: `{behavior_csv}`")
    else:
        lines.append("- Behavior CSV: (not generated)")
    if expected_score_csv is not None:
        lines.append(f"- Expected-score CSV: [`{expected_score_csv}`]({expected_score_csv})")
    else:
        lines.append("- Expected-score CSV: (not generated)")
    lines.append("")

    def _table(rows: List[Dict[str, object]], *, key_col: str, label: str) -> None:
        lines.append(f"## Strongest {label} Effects (Top {top_n} by delta_range)")
        lines.append(f"| game_setting | model | {key_col} | neutral | best (Δ) | worst (Δ) | range |")
        lines.append("|---|---|---|---:|---|---|---:|")
        top = sorted(rows, key=lambda r: float(r["delta_range"]), reverse=True)[:top_n]
        for r in top:
            lines.append(
                "| {task} | {model} | {k} | {neutral:.3f} | {best} ({best_delta:+.3f}) | {worst} ({worst_delta:+.3f}) | {rng:.3f} |".format(
                    task=r["task"],
                    model=r["model"],
                    k=r[key_col],
                    neutral=float(r["neutral_ratio"]),
                    best=r["best_emotion"],
                    best_delta=float(r["best_delta_vs_neutral"]),
                    worst=r["worst_emotion"],
                    worst_delta=float(r["worst_delta_vs_neutral"]),
                    rng=float(r["delta_range"]),
                )
            )
        lines.append("")

        if per_game_n > 0:
            lines.append(f"## Per Game Setting {label} (Top {per_game_n} by delta_range)")
        else:
            lines.append(f"## Per Game Setting {label} (All models)")
        tasks = sorted({r["task"] for r in rows})
        for task in tasks:
            lines.append(f"### {task}")
            lines.append(f"| model | {key_col} | neutral | best (Δ) | worst (Δ) | range |")
            lines.append("|---|---|---:|---|---|---:|")
            task_rows = [r for r in rows if r["task"] == task]
            if per_game_n > 0:
                task_rows = sorted(task_rows, key=lambda r: float(r["delta_range"]), reverse=True)[:per_game_n]
            else:
                task_rows = sorted(task_rows, key=lambda r: (str(r["model"]), str(r[key_col])))
            for r in task_rows:
                lines.append(
                    "| {model} | {k} | {neutral:.3f} | {best} ({best_delta:+.3f}) | {worst} ({worst_delta:+.3f}) | {rng:.3f} |".format(
                        model=r["model"],
                        k=r[key_col],
                        neutral=float(r["neutral_ratio"]),
                        best=r["best_emotion"],
                        best_delta=float(r["best_delta_vs_neutral"]),
                        worst=r["worst_emotion"],
                        worst_delta=float(r["worst_delta_vs_neutral"]),
                        rng=float(r["delta_range"]),
                    )
                )
            lines.append("")

    _table(option_rows, key_col="option_id", label="Option")

    if behavior_rows:
        _table(behavior_rows, key_col="behavior_label", label="Behavior")
    else:
        lines.append("## Behavior Effects")
        lines.append("No behavior ratio inputs found (`summary_behavior_ratio.csv`).")
        lines.append("")

    if expected_score_rows:
        lines.append("## Expected-Score Effects (item-traced, vs neutral)")
        lines.append(
            "- Computed from `raw_results.json` by mapping chosen option -> behavior -> numeric score, then taking per-item delta vs neutral."
        )
        lines.append("- Reported per `(emotion, intensity)` (no collapsing across intensities).")
        lines.append("")
        lines.append("| game_setting | model | emotion | intensity | Δ mean | neutral mean | score mean | n_items |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|")
        top = sorted(expected_score_rows, key=lambda r: abs(float(r["delta_mean"])), reverse=True)[:top_n]
        for r in top:
            lines.append(
                "| {task} | {model} | {emo} | {intensity:.3f} | {delta:+.3f} | {neutral:.3f} | {score:.3f} | {n} |".format(
                    task=r["task"],
                    model=r["model"],
                    emo=r["emotion"],
                    intensity=float(r["intensity"]),
                    delta=float(r["delta_mean"]),
                    neutral=float(r["neutral_score_mean"]),
                    score=float(r["score_mean"]),
                    n=int(float(r["n_items"])),
                )
            )
        lines.append("")
    else:
        lines.append("## Expected-Score Effects")
        lines.append("No expected-score inputs found (`raw_results.json` for supported games).")
        lines.append("")

    if skipped_missing_neutral:
        lines.append("## Skipped runs (missing neutral)")
        for p in sorted({p.parent for p in skipped_missing_neutral}):
            lines.append(f"- `{p}`")
        lines.append("")

    return "\n".join(lines) + "\n"


def generate_game_theory_impact_report(
    *,
    root: Path,
    top_n: int = 20,
    per_game_n: int = 0,
) -> ReportOutputs:
    runs = _discover_latest_runs(root)
    if not runs:
        raise ValueError(f"No timestamped run directories found under {root}")

    option_rows: List[Dict[str, object]] = []
    behavior_rows: List[Dict[str, object]] = []
    expected_score_rows: List[Dict[str, object]] = []
    skipped_missing_neutral: List[Path] = []
    for run in runs:
        choice_csv = run.dir_path / "summary_choice_ratio.csv"
        behavior_csv = run.dir_path / "summary_behavior_ratio.csv"
        raw_json = run.dir_path / "raw_results.json"
        if choice_csv.exists():
            rows, missing_neutral = _impact_rows_for_choice(run, choice_csv)
            option_rows.extend(rows)
            if missing_neutral:
                skipped_missing_neutral.append(choice_csv)
        if behavior_csv.exists():
            rows, missing_neutral = _impact_rows_for_behavior(run, behavior_csv)
            behavior_rows.extend(rows)
            if missing_neutral:
                skipped_missing_neutral.append(behavior_csv)
        if raw_json.exists():
            rows, missing_neutral = _expected_score_rows_for_run(run, raw_json)
            expected_score_rows.extend(rows)
            if missing_neutral:
                skipped_missing_neutral.append(raw_json)

    if not option_rows:
        raise ValueError(f"No usable summary_choice_ratio.csv found under {root} (need neutral)")

    option_out = root / "option_impacted_by_emo_vs_neutral_latest.csv"
    behavior_out = root / "behavior_impacted_emo_vs_neutral_latest.csv" if behavior_rows else None
    expected_out = (
        root / "expected_score_delta_vs_neutral_by_emotion_intensity_latest.csv" if expected_score_rows else None
    )
    report_out = root / "game_theory_impact_report.md"

    _write_csv(option_rows, option_out)
    if behavior_out is not None:
        _write_csv(behavior_rows, behavior_out)
    if expected_out is not None:
        _write_csv(expected_score_rows, expected_out)

    report_out.write_text(
        _render_markdown(
            root=root,
            runs=runs,
            option_csv=option_out,
            behavior_csv=behavior_out,
            expected_score_csv=expected_out,
            option_rows=option_rows,
            behavior_rows=behavior_rows,
            expected_score_rows=expected_score_rows,
            skipped_missing_neutral=skipped_missing_neutral,
            top_n=top_n,
            per_game_n=per_game_n,
        ),
        encoding="utf-8",
    )

    return ReportOutputs(option_csv_path=option_out, behavior_csv_path=behavior_out, report_path=report_out)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/new_game_theory_decision/shuffle_choices"),
        help="Directory containing timestamped run subfolders.",
    )
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--per_game_n", type=int, default=0, help="0 means include all models")
    args = parser.parse_args(list(argv) if argv is not None else None)
    generate_game_theory_impact_report(root=args.root, top_n=args.top_n, per_game_n=args.per_game_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
