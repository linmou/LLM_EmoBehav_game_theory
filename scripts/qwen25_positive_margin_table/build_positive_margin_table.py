#!/usr/bin/env python3
# Purpose: build the aggregated Qwen2.5 7-way positive-margin table from saved self-report sweep summaries in one command.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MARGIN_COLUMN = "delta_p_target_vs_top_p_non_target_mean"
DEFAULT_INTENSITIES = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 40.0, 80.0]
DEFAULT_EMOTIONS = ["anger", "happiness", "sadness", "fear", "disgust", "surprise"]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "auto_experiments" / "pd_selfreport_pd_coupling_multimodel"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "scripts" / "qwen25_positive_margin_table" / "outputs"


def _format_intensity(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def qwen25_model_roots(results_root: Path) -> dict[str, Path]:
    multimodel_root = results_root / "self_report_logprob_multimodel"
    roots: dict[str, Path] = {}

    legacy_root = results_root / "self_report_logprob"
    if legacy_root.exists():
        roots["qwen2p5-0p5b-instruct"] = legacy_root

    for model_slug in ("qwen2p5-1p5b-instruct", "qwen2p5-3b-instruct"):
        model_root = multimodel_root / model_slug
        if model_root.exists():
            roots[model_slug] = model_root

    return roots


def _collect_model_summary_rows(model_slug: str, model_root: Path, margin_column: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for summary_path in sorted(model_root.glob("*/target_option_softmax_by_steer.csv")):
        metadata_path = summary_path.parent / "run_metadata.json"
        if not metadata_path.exists():
            continue
        summary = pd.read_csv(summary_path)
        if summary.empty or margin_column not in summary.columns:
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        margin_value = pd.to_numeric(summary.iloc[0][margin_column], errors="coerce")
        if pd.isna(margin_value):
            continue
        rows.append(
            {
                "model": model_slug,
                "emotion": str(metadata["steer_emotion"]),
                "layer_1based": int(metadata["layer_1based"]),
                "intensity": float(metadata["intensity"]),
                "margin": float(margin_value),
            }
        )
    return rows


def collect_qwen25_margin_rows(
    results_root: Path,
    margin_column: str = DEFAULT_MARGIN_COLUMN,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_slug, model_root in qwen25_model_roots(results_root).items():
        rows.extend(_collect_model_summary_rows(model_slug, model_root, margin_column))
    if not rows:
        return pd.DataFrame(columns=["model", "emotion", "layer_1based", "intensity", "margin"])
    return pd.DataFrame(rows).sort_values(
        ["emotion", "intensity", "model", "layer_1based"]
    ).reset_index(drop=True)


def aggregate_positive_margin_table(
    results_root: Path,
    margin_column: str = DEFAULT_MARGIN_COLUMN,
    emotions: list[str] | None = None,
    intensities: list[float] | None = None,
) -> pd.DataFrame:
    rows = collect_qwen25_margin_rows(results_root, margin_column=margin_column)
    use_emotions = DEFAULT_EMOTIONS if emotions is None else [str(item) for item in emotions]
    use_intensities = DEFAULT_INTENSITIES if intensities is None else [float(item) for item in intensities]

    if rows.empty:
        return pd.DataFrame(
            [
                {
                    "emotion": emotion,
                    "intensity": intensity,
                    "positive_layers": 0,
                    "total_layers": 0,
                    "positive_fraction": 0.0,
                    "display": "0/0 (0.00)",
                }
                for emotion in use_emotions
                for intensity in use_intensities
            ]
        )

    rows = rows[rows["emotion"].isin(use_emotions) & rows["intensity"].isin(use_intensities)].copy()
    rows["is_positive"] = pd.to_numeric(rows["margin"], errors="coerce") > 0.0

    grouped = (
        rows.groupby(["emotion", "intensity"], as_index=False)
        .agg(
            positive_layers=("is_positive", "sum"),
            total_layers=("is_positive", "size"),
        )
        .sort_values(["emotion", "intensity"])
        .reset_index(drop=True)
    )

    full_index = pd.MultiIndex.from_product(
        [use_emotions, use_intensities],
        names=["emotion", "intensity"],
    )
    grouped = grouped.set_index(["emotion", "intensity"]).reindex(full_index, fill_value=0).reset_index()
    grouped["positive_layers"] = grouped["positive_layers"].astype(int)
    grouped["total_layers"] = grouped["total_layers"].astype(int)
    grouped["positive_fraction"] = grouped.apply(
        lambda row: 0.0 if int(row["total_layers"]) == 0 else float(row["positive_layers"]) / float(row["total_layers"]),
        axis=1,
    )
    grouped["display"] = grouped.apply(
        lambda row: f"{int(row['positive_layers'])}/{int(row['total_layers'])} ({float(row['positive_fraction']):.2f})",
        axis=1,
    )
    return grouped


def format_positive_margin_markdown(table: pd.DataFrame) -> str:
    if table.empty:
        return "| emotion |\n|---|\n"

    work = table.copy()
    intensity_order = sorted(work["intensity"].unique().tolist())
    pivot = (
        work.pivot(index="emotion", columns="intensity", values="display")
        .reindex(index=DEFAULT_EMOTIONS, columns=intensity_order)
        .fillna("0/0 (0.00)")
    )
    header = ["emotion"] + [_format_intensity(value) for value in intensity_order]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for emotion, row in pivot.iterrows():
        values = [emotion] + [str(row[intensity]) for intensity in intensity_order]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--margin-column", type=str, default=DEFAULT_MARGIN_COLUMN)
    args = parser.parse_args()

    table = aggregate_positive_margin_table(args.results_root, margin_column=args.margin_column)
    markdown = format_positive_margin_markdown(table)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "qwen25_positive_margin_table.csv"
    md_path = args.output_dir / "qwen25_positive_margin_table.md"
    table.to_csv(csv_path, index=False)
    md_path.write_text(markdown + "\n", encoding="utf-8")

    print(markdown)
    print(f"\nwrote {csv_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
