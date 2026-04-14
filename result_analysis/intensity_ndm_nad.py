#!/usr/bin/env python3
# Purpose: compute intensity-wise normalized decision magnitude/alignment from shuffle game-theory runs and render a LaTeX table.
"""Intensity-wise NDM/NAD analysis for shuffle game-theory decision runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from constants import Emotions, GameNames
from result_analysis.behavior_shift_alignment import DEFAULT_ALIGNMENT_SPECS


BEHAVIOR_TO_SCORE: dict[str, dict[str, float]] = {
    GameNames.PRISONERS_DILEMMA.value: {"defect": 0.0, "cooperate": 1.0},
    GameNames.STAG_HUNT.value: {"defect": 0.0, "cooperate": 1.0},
    GameNames.ESCALATION_GAME.value: {"withdraw": 0.0, "escalation": 1.0, "escalate": 1.0},
    GameNames.TRUST_GAME_TRUSTOR.value: {"trust_none": 0.0, "trust_low": 1.0, "trust_high": 2.0},
    GameNames.TRUST_GAME_TRUSTEE.value: {"return_none": 0.0, "return_medium": 1.0, "return_high": 2.0},
    GameNames.ULTIMATUM_GAME_PROPOSER.value: {"offer_low": 0.0, "offer_medium": 1.0, "offer_high": 2.0},
    GameNames.ULTIMATUM_GAME_RESPONDER.value: {"accept": 0.0, "reject": 1.0},
    GameNames.SEALED_AUCTION.value: {"devote_low": 0.0, "devote_medium": 1.0, "devote_high": 2.0},
    GameNames.BEAUTY_CONTEST.value: {"commit_0": 0.0, "commit_1": 1.0, "commit_2": 2.0, "commit_3": 3.0},
}

MODEL_LABELS: dict[str, str] = {
    "Llama-3.2-1B-Instruct": r"\shortstack{Llama-3.2\\1B}",
    "Llama-3.2-3B-Instruct": r"\shortstack{Llama-3.2\\3B}",
    "Phi-3.5-mini-instruct": r"\shortstack{Phi\\3.5-mini}",
    "Phi-4-mini-instruct": r"\shortstack{Phi\\4-mini}",
    "Qwen2.5-0.5B-Instruct": r"\shortstack{Qwen2.5\\0.5B}",
    "Qwen2.5-1.5B-Instruct": r"\shortstack{Qwen2.5\\1.5B}",
    "Qwen2.5-3B-Instruct": r"\shortstack{Qwen2.5\\3B}",
    "Qwen3-0.6B": r"\shortstack{Qwen3\\0.6B}",
    "Qwen3-1.7B": r"\shortstack{Qwen3\\1.7B}",
    "Qwen3-4B": r"\shortstack{Qwen3\\4B}",
    "gemma-3-270m-it": r"\shortstack{Gemma-3\\270M}",
    "gemma-3-1b-it": r"\shortstack{Gemma-3\\1B}",
    "gemma-3-4b-it": r"\shortstack{Gemma-3\\4B}",
    "Zamba2-1.2B-Instruct": r"\shortstack{Zamba2\\1.2B}",
    "Zamba2-2.7B-Instruct": r"\shortstack{Zamba2\\2.7B}",
    "mamba2-1.3b": r"\shortstack{mamba2\\1.3B}",
    "mamba2-2.7b": r"\shortstack{mamba2\\2.7B}",
    "Phi-3.5-vision-instruct": r"\shortstack{Phi-3.5\\vision}",
    "Phi-4-multimodal-instruct": r"\shortstack{Phi-4\\multi}",
    "Qwen3-VL-2B-Instruct": r"\shortstack{Qwen3-VL\\2B}",
    "Qwen3-VL-4B-Instruct": r"\shortstack{Qwen3-VL\\4B}",
    "Qwen3-VL-4B-Thinking": r"\shortstack{Qwen3-VL\\4B-Thk}",
    "InternVL3-1B": r"\shortstack{InternVL3\\1B}",
    "InternVL3-2B": r"\shortstack{InternVL3\\2B}",
    "InternVL3-8B-AWQ": r"\shortstack{InternVL3\\8B}",
}


def _parse_run_dir_name(name: str) -> tuple[str, str, str] | None:
    marker = "_game_theory_decision_"
    if marker not in name:
        return None
    model, rest = name.split(marker, 1)
    try:
        task, ts_date, ts_time = rest.rsplit("_", 2)
    except ValueError:
        return None
    return model, task, f"{ts_date}_{ts_time}"


def _discover_latest_runs(root: Path) -> dict[tuple[str, str], Path]:
    latest: dict[tuple[str, str], tuple[str, Path]] = {}
    for child in root.iterdir():
        if not child.is_dir():
            continue
        parsed = _parse_run_dir_name(child.name)
        if parsed is None:
            continue
        model, task, timestamp = parsed
        key = (model, task)
        if key not in latest or timestamp > latest[key][0]:
            latest[key] = (timestamp, child)
    return {key: path for key, (_, path) in latest.items()}


def _extract_behavior(record: dict[str, Any]) -> str | None:
    score = record.get("score")
    if score is None:
        return None
    try:
        option_id = int(float(score))
    except (TypeError, ValueError):
        return None
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return None
    item_metadata = metadata.get("item_metadata")
    if not isinstance(item_metadata, dict):
        return None
    options = item_metadata.get("options")
    if not isinstance(options, list):
        return None
    for option in options:
        if not isinstance(option, dict):
            continue
        try:
            if int(option.get("id", -1)) == option_id:
                behavior = option.get("behavior")
                if isinstance(behavior, str):
                    return behavior
                return None
        except (TypeError, ValueError):
            continue
    return None


def _load_rows_from_records(records: list[dict[str, Any]], model_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        if record.get("error"):
            continue
        task_name = record.get("task_name")
        emotion = record.get("emotion")
        intensity = record.get("intensity")
        item_id = record.get("item_id")
        repeat_id = record.get("repeat_id")
        if not isinstance(task_name, str) or task_name not in BEHAVIOR_TO_SCORE:
            continue
        if not isinstance(emotion, str):
            continue
        behavior = _extract_behavior(record)
        if behavior is None:
            continue
        behavior_scores = BEHAVIOR_TO_SCORE[task_name]
        if behavior not in behavior_scores:
            continue
        if intensity is None or item_id is None or repeat_id is None:
            continue
        try:
            intensity_value = float(intensity)
            item_key = int(item_id)
            repeat_key = int(repeat_id)
        except (TypeError, ValueError):
            continue
        score = behavior_scores[behavior]
        response_range = max(behavior_scores.values()) - min(behavior_scores.values())
        if response_range <= 0.0:
            continue
        rows.append(
            {
                "model": model_name,
                "task": task_name,
                "emotion": emotion,
                "intensity": intensity_value,
                "item_id": item_key,
                "repeat_id": repeat_key,
                "decision_score": score,
                "response_range": response_range,
            }
        )
    return rows


def load_rows_from_run_dir(run_dir: Path, model_name: str) -> list[dict[str, object]]:
    raw_path = run_dir / "raw_results.json"
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {raw_path}")
    records = [record for record in payload if isinstance(record, dict)]
    return _load_rows_from_records(records, model_name)


def _load_records_from_run_dir(run_dir: Path) -> list[dict[str, Any]]:
    raw_path = run_dir / "raw_results.json"
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {raw_path}")
    return [record for record in payload if isinstance(record, dict)]


def compute_metrics_by_intensity(records: list[dict[str, Any]], model_name: str) -> pd.DataFrame:
    rows = _load_rows_from_records(records, model_name)
    if not rows:
        raise ValueError("No valid records to score")
    df = pd.DataFrame(rows)

    neutral = (
        df[df["emotion"] == "neutral"][["model", "task", "item_id", "repeat_id", "decision_score"]]
        .drop_duplicates(subset=["model", "task", "item_id", "repeat_id"])
        .rename(columns={"decision_score": "neutral_score"})
    )

    shifted = df[df["emotion"] != "neutral"].copy()
    shifted = shifted.merge(neutral, on=["model", "task", "item_id", "repeat_id"], how="left")
    shifted = shifted.dropna(subset=["neutral_score"]).copy()
    if shifted.empty:
        raise ValueError("No non-neutral rows with neutral pairing")

    def expected_direction(task: str, emotion: str) -> int:
        spec = DEFAULT_ALIGNMENT_SPECS[GameNames.from_string(task)]
        return int(spec.expected_by_emotion[Emotions.from_string(emotion)])

    shifted["human_direction"] = [
        expected_direction(str(task), str(emotion))
        for task, emotion in zip(shifted["task"], shifted["emotion"], strict=True)
    ]
    shifted["normalized_delta"] = (shifted["decision_score"] - shifted["neutral_score"]) / shifted["response_range"]
    shifted["ndm_component"] = shifted["normalized_delta"].abs()
    shifted["nad_component"] = shifted["human_direction"] * shifted["normalized_delta"]

    aggregated = (
        shifted.groupby(["model", "intensity"], as_index=False)
        .agg(
            ndm=("ndm_component", "mean"),
            nad=("nad_component", "mean"),
            n_pairs=("normalized_delta", "size"),
        )
        .sort_values(["model", "intensity"], kind="mergesort")
        .reset_index(drop=True)
    )
    return aggregated


def compute_metrics_for_root(root: Path) -> pd.DataFrame:
    latest_runs = _discover_latest_runs(root)
    records_by_model: dict[str, list[dict[str, Any]]] = {}
    for (model_name, _task), run_dir in sorted(latest_runs.items()):
        records_by_model.setdefault(model_name, []).extend(_load_records_from_run_dir(run_dir))
    if not records_by_model:
        raise ValueError(f"No timestamped run directories found under {root}")
    frames = [
        compute_metrics_by_intensity(model_records, model_name)
        for model_name, model_records in sorted(records_by_model.items())
    ]
    return pd.concat(frames, ignore_index=True).sort_values(["model", "intensity"], kind="mergesort").reset_index(drop=True)


def _format_metric(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def render_latex_table(
    *,
    intensity_values: list[float],
    model_labels: dict[str, str],
    metrics_by_model: dict[str, dict[float, dict[str, float]]],
) -> str:
    model_names = [model for model in model_labels if model in metrics_by_model]
    header_cells = " &\n    ".join(model_labels[model] for model in model_names)
    rows: list[str] = []
    for intensity in intensity_values:
        ndm_values = [
            _format_metric(metrics_by_model.get(model, {}).get(intensity, {}).get("ndm"))
            for model in model_names
        ]
        nad_values = [
            _format_metric(metrics_by_model.get(model, {}).get(intensity, {}).get("nad"))
            for model in model_names
        ]
        rows.append(f"    NDM(${intensity:.1f}$) & " + " & ".join(ndm_values) + r" \\")
        rows.append(f"    NAD(${intensity:.1f}$) & " + " & ".join(nad_values) + r" \\")
        if intensity != intensity_values[-1]:
            rows.append("    \\midrule")

    row_block = "\n".join(rows)
    return (
        "% Intent: summarize how steering intensity changes normalized decision magnitude and human-direction alignment.\n\n"
        "\\subsection{Intensity effects on decision magnitude and human-direction alignment}\n"
        "\\label{subsec:intensity_impact}\n\n"
        "We replace the previous absolute-deviation summary with two normalized item-level metrics. "
        "$\\mathrm{NDM}$ measures the mean magnitude of the decision shift away from the neutral baseline, "
        "after scaling each item by its feasible response range. $\\mathrm{NAD}$ keeps the same normalization "
        "but signs each shift by the human-direction expectation used in the behavior-alignment analysis, so "
        "positive values mean the emotion-induced movement follows the expected human direction and negative values "
        "mean it moves against that direction. Table~\\ref{tab:intensity_impact_model} reports both metrics by "
        "steering intensity and model, using the latest run for each model-task pair under "
        "\\texttt{results/new\\_game\\_theory\\_decision/shuffle\\_300\\_samples}. "
        "The intensity grid is not uniform across model families, so `N/A' means that model was not evaluated at that steering strength.\n"
        "\\begin{table*}[h]\n"
        "    \\centering\n"
        "    \\scriptsize\n"
        "    \\setlength{\\tabcolsep}{3pt}\n"
        "    \\renewcommand{\\arraystretch}{1.12}\n"
        "    \\resizebox{\\textwidth}{!}{%\n"
        "    \\begin{tabular}{l"
        + ("c" * len(model_names))
        + "}\n"
        "    \\toprule\n"
        "    Metric &\n"
        f"    {header_cells} \\\\\n"
        "    \\midrule\n"
        f"{row_block}\n"
        "    \\bottomrule\n"
        "    \\end{tabular}%\n"
        "    }\n"
        "    \\caption{Normalized decision magnitude (NDM) and normalized alignment deviation (NAD) across steering intensities. "
        "Higher NDM means larger emotion-induced movement away from neutral decisions; higher NAD means that movement is more aligned with the human-direction prior. `N/A' indicates that the corresponding intensity was not part of that model's evaluated grid.}\n"
        "    \\label{tab:intensity_impact_model}\n"
        "    \\end{table*}\n"
    )


def write_latex_table(out_path: Path, table: str) -> None:
    out_path.write_text(table, encoding="utf-8")


def build_metrics_by_model(df: pd.DataFrame) -> dict[str, dict[float, dict[str, float]]]:
    metrics_by_model: dict[str, dict[float, dict[str, float]]] = {}
    for row in df.to_dict("records"):
        metrics_by_model.setdefault(str(row["model"]), {})[float(row["intensity"])] = {
            "ndm": float(row["ndm"]),
            "nad": float(row["nad"]),
        }
    return metrics_by_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/new_game_theory_decision/shuffle_300_samples"),
        help="Root directory containing timestamped game-theory shuffle run directories.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/jjl7137/COLM26-emotionalSLM/latex/intensity_impact.tex"),
        help="Output LaTeX file path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = compute_metrics_for_root(args.root)
    intensity_values = sorted({float(value) for value in metrics["intensity"].tolist()})
    model_order = [model for model in MODEL_LABELS if model in set(metrics["model"])]
    model_labels = {model: MODEL_LABELS[model] for model in model_order}
    latex = render_latex_table(
        intensity_values=intensity_values,
        model_labels=model_labels,
        metrics_by_model=build_metrics_by_model(metrics),
    )
    write_latex_table(args.out, latex)
    print(metrics.to_string(index=False))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
