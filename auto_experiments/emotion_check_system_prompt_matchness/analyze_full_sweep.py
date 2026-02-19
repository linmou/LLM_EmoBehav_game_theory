#!/usr/bin/env python3
"""
Analyze full-sweep results for the best system prompt variant and compare with baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")
    df["is_match"] = df["predicted_emotion"] == df["ground_truth"]
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    return df


def _overall(df: pd.DataFrame) -> Dict[str, float]:
    steered = df[df["emotion"] != "neutral"].copy()
    neutral = df[df["emotion"] == "neutral"].copy()
    return {
        "steered_plain_accuracy": float(steered["is_match"].mean()),
        "steered_match_score": float(steered["score"].mean()),
        "neutral_plain_accuracy": float(neutral["is_match"].mean()),
        "num_steered_rows": int(len(steered)),
        "num_neutral_rows": int(len(neutral)),
    }


def _best_intensity(df: pd.DataFrame) -> pd.DataFrame:
    steered = df[df["emotion"] != "neutral"].copy()
    grouped = (
        steered.groupby(["emotion", "intensity"], as_index=False)
        .agg(plain_accuracy=("is_match", "mean"), match_score=("score", "mean"))
    )
    best = (
        grouped.sort_values(["emotion", "plain_accuracy", "match_score"], ascending=[True, False, False])
        .groupby("emotion", as_index=False)
        .first()
    )
    return best


def _confusion(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    steered = df[df["emotion"] != "neutral"].copy()
    count = pd.crosstab(steered["ground_truth"], steered["predicted_emotion"], dropna=False)
    count = count.sort_index().sort_index(axis=1)
    row_norm = count.div(count.sum(axis=1).replace(0, 1), axis=0)
    return count, row_norm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", type=Path, required=True, help="New detailed_results.csv")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline detailed_results.csv")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("auto_experiments/emotion_check_system_prompt_matchness/analysis_iteration2"),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    new_df = _load(args.new)
    base_df = _load(args.baseline)

    new_overall = _overall(new_df)
    base_overall = _overall(base_df)
    comparison = pd.DataFrame(
        [
            {"run": "new_full_sweep_best_prompt", **new_overall},
            {"run": "baseline_full_sweep", **base_overall},
            {
                "run": "delta_new_minus_baseline",
                "steered_plain_accuracy": new_overall["steered_plain_accuracy"] - base_overall["steered_plain_accuracy"],
                "steered_match_score": new_overall["steered_match_score"] - base_overall["steered_match_score"],
                "neutral_plain_accuracy": new_overall["neutral_plain_accuracy"] - base_overall["neutral_plain_accuracy"],
                "num_steered_rows": new_overall["num_steered_rows"] - base_overall["num_steered_rows"],
                "num_neutral_rows": new_overall["num_neutral_rows"] - base_overall["num_neutral_rows"],
            },
        ]
    )
    comparison.to_csv(args.out_dir / "overall_comparison.csv", index=False)

    new_best = _best_intensity(new_df).rename(
        columns={
            "intensity": "best_intensity",
            "plain_accuracy": "best_plain_accuracy",
            "match_score": "best_match_score",
        }
    )
    new_best.to_csv(args.out_dir / "best_intensity_per_emotion.csv", index=False)

    count, row_norm = _confusion(new_df)
    count.to_csv(args.out_dir / "confusion_matrix_counts_steered_only.csv")
    row_norm.to_csv(args.out_dir / "confusion_matrix_row_normalized_steered_only.csv")

    overlap_intensities = sorted(set(base_df["intensity"].dropna()) & set(new_df["intensity"].dropna()))
    base_overlap = base_df[base_df["intensity"].isin(overlap_intensities)].copy()
    new_overlap = new_df[new_df["intensity"].isin(overlap_intensities)].copy()
    overlap_comp = pd.DataFrame(
        [
            {"run": "new_overlap", **_overall(new_overlap)},
            {"run": "baseline_overlap", **_overall(base_overlap)},
            {
                "run": "delta_overlap",
                "steered_plain_accuracy": _overall(new_overlap)["steered_plain_accuracy"] - _overall(base_overlap)["steered_plain_accuracy"],
                "steered_match_score": _overall(new_overlap)["steered_match_score"] - _overall(base_overlap)["steered_match_score"],
                "neutral_plain_accuracy": _overall(new_overlap)["neutral_plain_accuracy"] - _overall(base_overlap)["neutral_plain_accuracy"],
                "num_steered_rows": _overall(new_overlap)["num_steered_rows"] - _overall(base_overlap)["num_steered_rows"],
                "num_neutral_rows": _overall(new_overlap)["num_neutral_rows"] - _overall(base_overlap)["num_neutral_rows"],
            },
        ]
    )
    overlap_comp.to_csv(args.out_dir / "overlap_intensity_comparison.csv", index=False)

    summary = {
        "new_run_path": str(args.new),
        "baseline_run_path": str(args.baseline),
        "overlap_intensities": overlap_intensities,
        "new_overall": new_overall,
        "baseline_overall": base_overall,
    }
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved analysis to: {args.out_dir}")


if __name__ == "__main__":
    main()

