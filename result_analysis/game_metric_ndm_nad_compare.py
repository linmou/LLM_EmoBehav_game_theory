#!/usr/bin/env python3
# Purpose: compute per-(model, game) NDM/NAD from game-theory shuffle runs and render a GT/Dec comparison LaTeX table.
"""Per-game NDM/NAD comparison across shuffle game-theory benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd  # type: ignore[import-untyped]

from constants import Emotions, GameNames
from result_analysis.behavior_shift_alignment import DEFAULT_ALIGNMENT_SPECS
from result_analysis.intensity_ndm_nad import BEHAVIOR_TO_SCORE, MODEL_LABELS


GT_ROOT = Path("results/new_game_theory/shuffle_300_samples")
DECISION_ROOT = Path("results/new_game_theory_decision/shuffle_300_samples")
DEFAULT_OUT = Path("result_analysis/_tmp_reports/game_metric_ndm_nad_compare_shuffle_300_samples.tex")
DEFAULT_CSV_OUT = Path("result_analysis/_tmp_reports/game_metric_ndm_nad_compare_shuffle_300_samples.csv")
GAME_ORDER = [
    GameNames.PRISONERS_DILEMMA.value,
    GameNames.STAG_HUNT.value,
    GameNames.ESCALATION_GAME.value,
    GameNames.TRUST_GAME_TRUSTOR.value,
    GameNames.TRUST_GAME_TRUSTEE.value,
    GameNames.ULTIMATUM_GAME_PROPOSER.value,
    GameNames.ULTIMATUM_GAME_RESPONDER.value,
    GameNames.BEAUTY_CONTEST.value,
    GameNames.SEALED_AUCTION.value,
]
GAME_LABELS = {
    GameNames.PRISONERS_DILEMMA.value: r"\shortstack{Prisoners'\\Dilemma\\GT / Dec}",
    GameNames.STAG_HUNT.value: r"\shortstack{Stag\\Hunt\\GT / Dec}",
    GameNames.ESCALATION_GAME.value: r"\shortstack{Escalation\\Game\\GT / Dec}",
    GameNames.TRUST_GAME_TRUSTOR.value: r"\shortstack{Trust\\Trustor\\GT / Dec}",
    GameNames.TRUST_GAME_TRUSTEE.value: r"\shortstack{Trust\\Trustee\\GT / Dec}",
    GameNames.ULTIMATUM_GAME_PROPOSER.value: r"\shortstack{Ultimatum\\Proposer\\GT / Dec}",
    GameNames.ULTIMATUM_GAME_RESPONDER.value: r"\shortstack{Ultimatum\\Responder\\GT / Dec}",
    GameNames.BEAUTY_CONTEST.value: r"\shortstack{Beauty\\Contest\\GT / Dec}",
    GameNames.SEALED_AUCTION.value: r"\shortstack{Sealed\\Auction\\GT / Dec}",
}


class MetricByBenchmark(TypedDict):
    gt: float | None
    decision: float | None


class ComparisonRow(TypedDict):
    model: str
    model_family: str
    params: str
    ndm_by_game: dict[str, dict[str, float]]
    nad_by_game: dict[str, dict[str, float]]
    mean_ndm: MetricByBenchmark
    mean_nad: MetricByBenchmark


def _parse_run_dir_name(name: str, marker: str) -> tuple[str, str, str] | None:
    if marker not in name:
        return None
    model, rest = name.split(marker, 1)
    try:
        task, ts_date, ts_time = rest.rsplit("_", 2)
    except ValueError:
        return None
    return model, task, f"{ts_date}_{ts_time}"


def _discover_latest_runs(root: Path, marker: str) -> tuple[dict[tuple[str, str], list[Path]], list[Path]]:
    candidates: dict[tuple[str, str], list[tuple[str, Path]]] = {}
    for child in root.iterdir():
        if not child.is_dir():
            continue
        parsed = _parse_run_dir_name(child.name, marker)
        if parsed is None:
            continue
        model, task, timestamp = parsed
        candidates.setdefault((model, task), []).append((timestamp, child))

    latest: dict[tuple[str, str], list[Path]] = {}
    skipped_incomplete: list[Path] = []
    for key, runs in candidates.items():
        usable_candidates: list[Path] = []
        for _timestamp, run_dir in sorted(runs, key=lambda item: item[0], reverse=True):
            if (run_dir / "raw_results.json").is_file():
                usable_candidates.append(run_dir)
            else:
                skipped_incomplete.append(run_dir)
        if usable_candidates:
            latest[key] = usable_candidates
    return latest, skipped_incomplete


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
            if int(option.get("id", -1)) != option_id:
                continue
        except (TypeError, ValueError):
            continue
        behavior = option.get("behavior")
        if isinstance(behavior, str):
            return behavior
        return None
    return None


def _load_rows_from_records(records: list[dict[str, Any]], model_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        if record.get("error"):
            continue
        task_name = record.get("task_name")
        emotion = record.get("emotion")
        item_id = record.get("item_id")
        repeat_id = record.get("repeat_id")
        if not isinstance(task_name, str) or task_name not in BEHAVIOR_TO_SCORE:
            continue
        if not isinstance(emotion, str):
            continue
        if item_id is None or repeat_id is None:
            continue
        behavior = _extract_behavior(record)
        if behavior is None:
            continue
        behavior_scores = BEHAVIOR_TO_SCORE[task_name]
        if behavior not in behavior_scores:
            continue
        try:
            item_key = int(item_id)
            repeat_key = int(repeat_id)
        except (TypeError, ValueError):
            continue
        response_range = max(behavior_scores.values()) - min(behavior_scores.values())
        if response_range <= 0.0:
            continue
        rows.append(
            {
                "model": model_name,
                "task": task_name,
                "emotion": emotion,
                "item_id": item_key,
                "repeat_id": repeat_key,
                "decision_score": behavior_scores[behavior],
                "response_range": response_range,
            }
        )
    return rows


def _load_records_from_run_dir(run_dir: Path) -> list[dict[str, Any]]:
    raw_path = run_dir / "raw_results.json"
    text = raw_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        decoder = json.JSONDecoder()
        try:
            payload, end = decoder.raw_decode(text)
        except json.JSONDecodeError as recover_exc:
            raise ValueError(f"Invalid JSON in {raw_path}: {recover_exc}") from exc
        trailing = text[end:].strip()
        if trailing:
            print(
                f"Recovered first JSON value from {raw_path} and ignored trailing data after char {end}",
                file=sys.stderr,
            )
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {raw_path}")
    return [record for record in payload if isinstance(record, dict)]


def compute_metrics_by_game(records: list[dict[str, Any]], model_name: str) -> pd.DataFrame:
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

    shifted["human_direction"] = [
        int(DEFAULT_ALIGNMENT_SPECS[GameNames.from_string(str(task))].expected_by_emotion[Emotions.from_string(str(emotion))])
        for task, emotion in zip(shifted["task"], shifted["emotion"], strict=True)
    ]
    shifted["normalized_delta"] = (shifted["decision_score"] - shifted["neutral_score"]) / shifted["response_range"]
    shifted["ndm_component"] = shifted["normalized_delta"].abs()
    shifted["nad_component"] = shifted["human_direction"] * shifted["normalized_delta"]

    return (
        shifted.groupby(["model", "task"], as_index=False)
        .agg(
            ndm=("ndm_component", "mean"),
            nad=("nad_component", "mean"),
            n_pairs=("normalized_delta", "size"),
        )
        .sort_values(["model", "task"], kind="mergesort")
        .reset_index(drop=True)
    )


def compute_metrics_for_root(root: Path, marker: str) -> pd.DataFrame:
    latest_runs, skipped_incomplete = _discover_latest_runs(root, marker)
    for run_dir in skipped_incomplete:
        print(f"Skipping incomplete run without raw_results.json: {run_dir}", file=sys.stderr)
    frames: list[pd.DataFrame] = []
    for (model_name, _task), run_dirs in sorted(latest_runs.items()):
        for run_dir in run_dirs:
            try:
                frames.append(compute_metrics_by_game(_load_records_from_run_dir(run_dir), model_name))
                break
            except ValueError as exc:
                print(f"Skipping unusable run {run_dir}: {exc}", file=sys.stderr)
    if not frames:
        raise ValueError(f"No timestamped run directories found under {root}")
    return pd.concat(frames, ignore_index=True).sort_values(["model", "task"], kind="mergesort").reset_index(drop=True)


def _format_metric(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def _metric_mean(metric_by_game: dict[str, dict[str, float]], benchmark: str) -> float | None:
    values = [value[benchmark] for value in metric_by_game.values() if benchmark in value]
    if not values:
        return None
    return sum(values) / len(values)


def _split_model_label(model_name: str) -> tuple[str, str]:
    family_map = {
        "Llama-3.2-1B-Instruct": ("Llama-3.2-Instruct", "1B"),
        "Llama-3.2-3B-Instruct": ("Llama-3.2-Instruct", "3B"),
        "Phi-3.5-mini-instruct": ("Phi-3.5-instruct", "4B"),
        "Phi-4-mini-instruct": ("Phi-4-instruct", "4B"),
        "Qwen2.5-0.5B-Instruct": ("Qwen2.5-Instruct", "0.5B"),
        "Qwen2.5-1.5B-Instruct": ("Qwen2.5-Instruct", "1.5B"),
        "Qwen2.5-3B-Instruct": ("Qwen2.5-Instruct", "3B"),
        "Qwen3-0.6B": ("Qwen3", "0.6B"),
        "Qwen3-1.7B": ("Qwen3", "1.7B"),
        "Qwen3-4B": ("Qwen3", "4B"),
        "gemma-3-270m-it": ("Gemma-3-it", "270M"),
        "gemma-3-1b-it": ("Gemma-3-it", "1B"),
        "gemma-3-4b-it": ("Gemma-3-it", "4B"),
        "Zamba2-1.2B-Instruct": ("Zamba2-Instruct", "1.2B"),
        "Zamba2-2.7B-Instruct": ("Zamba2-Instruct", "2.7B"),
        "mamba2-1.3b": ("Mamba2", "1.3B"),
        "mamba2-2.7b": ("Mamba2", "2.7B"),
        "InternVL3-1B": ("InternVL3", "1B"),
        "InternVL3-2B": ("InternVL3", "2B"),
        "InternVL3-8B-AWQ": ("InternVL3", "8B"),
        "Phi-3.5-vision-instruct": ("Phi-3.5-vision", "4B"),
        "Phi-4-multimodal-instruct": ("Phi-4-multimodal", "4B"),
        "Qwen2.5-VL-Instruct": ("Qwen2.5-VL-Instruct", "3B"),
        "Qwen3-VL-2B-Instruct": ("Qwen3-VL-Instruct", "2B"),
        "Qwen3-VL-4B-Instruct": ("Qwen3-VL-Instruct", "4B"),
        "Qwen3-VL-4B-Thinking": ("Qwen3-VL-Thinking", "4B"),
    }
    return family_map.get(model_name, (model_name, ""))


def build_comparison_rows(gt_df: pd.DataFrame, decision_df: pd.DataFrame) -> list[ComparisonRow]:
    metrics: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for benchmark, df in (("gt", gt_df), ("decision", decision_df)):
        for row in df.to_dict("records"):
            model = str(row["model"])
            task = str(row["task"])
            metric_slot = metrics.setdefault(model, {"ndm_by_game": {}, "nad_by_game": {}})
            metric_slot["ndm_by_game"].setdefault(task, {})[benchmark] = float(row["ndm"])
            metric_slot["nad_by_game"].setdefault(task, {})[benchmark] = float(row["nad"])

    rows: list[ComparisonRow] = []
    for model in MODEL_LABELS:
        if model not in metrics:
            continue
        ndm_by_game = metrics[model]["ndm_by_game"]
        nad_by_game = metrics[model]["nad_by_game"]
        model_family, params = _split_model_label(model)
        rows.append(
            {
                "model": model,
                "model_family": model_family,
                "params": params,
                "ndm_by_game": ndm_by_game,
                "nad_by_game": nad_by_game,
                "mean_ndm": {
                    "gt": _metric_mean(ndm_by_game, "gt"),
                    "decision": _metric_mean(ndm_by_game, "decision"),
                },
                "mean_nad": {
                    "gt": _metric_mean(nad_by_game, "gt"),
                    "decision": _metric_mean(nad_by_game, "decision"),
                },
            }
        )
    return rows


def render_comparison_latex_table(*, games: list[str], rows: list[ComparisonRow]) -> str:
    column_spec = "ll" + ("c" * (len(games) + 1))
    header_cells = " & \n    ".join(
        GAME_LABELS.get(game, game).replace("\\GT / Dec}", "\\ \\tiny NDM (GT / Dec) \\\\ \\tiny NAD (GT / Dec)}")
        for game in games
    )
    lines = [
        r"\begin{table*}[t]",
        r"    \centering",
        r"    \scriptsize",
        r"    \setlength{\tabcolsep}{2.1pt}",
        r"    \renewcommand{\arraystretch}{1.2}",
        r"    \resizebox{\textwidth}{!}{%",
        rf"    \begin{{tabular}}{{{column_spec}}}",
        r"    \toprule",
        "    Model & Params & \n    "
        + header_cells
        + r" & \shortstack{Mean\\ \tiny NDM (GT / Dec) \\ \tiny NAD (GT / Dec)} \\",
        r"    \midrule",
    ]
    for index, row in enumerate(rows):
        model_family = row["model_family"]
        params = row["params"]
        ndm_by_game = row["ndm_by_game"]
        nad_by_game = row["nad_by_game"]
        mean_ndm = row["mean_ndm"]
        mean_nad = row["mean_nad"]
        cells = []
        for game in games:
            cells.append(
                r"\shortstack{"
                + f"{_format_metric(ndm_by_game.get(game, {}).get('gt'))} / {_format_metric(ndm_by_game.get(game, {}).get('decision'))}"
                + r" \\ "
                + f"{_format_metric(nad_by_game.get(game, {}).get('gt'))} / {_format_metric(nad_by_game.get(game, {}).get('decision'))}"
                + r"}"
            )
        mean_cell = (
            r"\shortstack{"
            + f"{_format_metric(mean_ndm.get('gt'))} / {_format_metric(mean_ndm.get('decision'))}"
            + r" \\ "
            + f"{_format_metric(mean_nad.get('gt'))} / {_format_metric(mean_nad.get('decision'))}"
            + r"}"
        )
        if index != len(rows) - 1:
            separator = r""
        else:
            separator = r""
        lines.append(
            f"    {model_family} & {params} & "
            + " & ".join(cells)
            + f" & {mean_cell} \\\\"
        )
        if model_family in {"Gemma-3-it", "Mamba2"} or index == len(rows) - 1:
            if index != len(rows) - 1:
                lines.append(r"    \midrule")
        elif index + 1 < len(rows) and rows[index + 1]["model_family"] != model_family:
            lines.append("")
    lines.extend(
        [
            r"    \bottomrule",
            r"    \end{tabular}%",
            r"    }",
            r"    \caption{Per-game normalized decision magnitude (NDM) and normalized alignment deviation (NAD) for the latest runs under \texttt{results/new\_game\_theory/shuffle\_300\_samples} and \texttt{results/new\_game\_theory\_decision/shuffle\_300\_samples}. Each cell shows NDM on the first line and NAD on the second line, both reported as `GT / Dec'.}",
            r"    \label{tab:game_metric_ndm_nad_compare_shuffle_300_samples}",
            r"    \end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_latex_table(out_path: Path, table: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table, encoding="utf-8")


def write_metrics_csv(out_path: Path, gt_df: pd.DataFrame, decision_df: pd.DataFrame) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(
        [
            gt_df.assign(benchmark="GT"),
            decision_df.assign(benchmark="Dec"),
        ],
        ignore_index=True,
    )
    combined.to_csv(out_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt_root", type=Path, default=GT_ROOT, help="Root directory for shuffle game-theory runs.")
    parser.add_argument(
        "--decision_root",
        type=Path,
        default=DECISION_ROOT,
        help="Root directory for shuffle decision-only game-theory runs.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output LaTeX file path.")
    parser.add_argument("--csv_out", type=Path, default=DEFAULT_CSV_OUT, help="Output CSV file path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gt_df = compute_metrics_for_root(args.gt_root, "_game_theory_")
    decision_df = compute_metrics_for_root(args.decision_root, "_game_theory_decision_")
    rows = build_comparison_rows(gt_df, decision_df)
    games = [game for game in GAME_ORDER if any(game in row["ndm_by_game"] or game in row["nad_by_game"] for row in rows)]
    latex = render_comparison_latex_table(games=games, rows=rows)
    write_latex_table(args.out, latex)
    write_metrics_csv(args.csv_out, gt_df, decision_df)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
