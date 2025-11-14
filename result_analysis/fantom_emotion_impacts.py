"""
result_analysis/fantom_emotion_impacts.py
Minimal utilities to parse Fantom run summaries and compute emotion deltas.

We read `summary_overall.csv` produced per run and compute
delta(emotion) = mean_of_means(emotion) - mean_of_means(neutral).

This stays deliberately simple to avoid overdesign; significance,
stratification, and cross-run aggregation can layer on top.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple


@dataclass
class SummaryRow:
    emotion: str
    intensity: float
    repeats: int
    total_count: int
    mean_of_means: float
    between_run_var: float
    pooled_var: float


def read_summary_overall(run_dir: Path) -> Dict[str, SummaryRow]:
    """Read summary_overall.csv into a dict keyed by emotion.

    Expects columns: emotion,intensity,repeats,total_count,mean_of_means,between_run_var,pooled_var
    """
    csv_path = Path(run_dir) / "summary_overall.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing summary_overall.csv at {csv_path}")

    out: Dict[str, SummaryRow] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                emotion = row["emotion"].strip()
                r = SummaryRow(
                    emotion=emotion,
                    intensity=float(row["intensity"]),
                    repeats=int(row["repeats"]),
                    total_count=int(row["total_count"]),
                    mean_of_means=float(row["mean_of_means"]),
                    between_run_var=float(row["between_run_var"]),
                    pooled_var=float(row["pooled_var"]),
                )
            except Exception as e:
                raise ValueError(f"Bad row in {csv_path}: {row}") from e
            out[emotion] = r
    if "neutral" not in out:
        raise ValueError(f"No neutral row found in {csv_path}")
    return out


def compute_emotion_deltas(run_dir: Path, intensity: float | None = None) -> Dict[str, float]:
    """Compute per-emotion delta vs neutral for a single run directory.

    If `intensity` is provided, only compare emotions with that intensity.
    Neutral baseline is taken as-is (typically 0.0).
    Returns mapping emotion -> (mean_of_means(emotion) - mean_of_means(neutral)).
    """
    rows = read_summary_overall(run_dir)
    neutral_mean = rows["neutral"].mean_of_means
    deltas: Dict[str, float] = {}
    for emo, r in rows.items():
        if emo == "neutral":
            continue
        if intensity is not None and r.intensity != intensity:
            continue
        deltas[emo] = r.mean_of_means - neutral_mean
    return deltas


