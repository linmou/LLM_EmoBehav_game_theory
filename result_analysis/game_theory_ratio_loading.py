"""Load and (optionally) filter game-theory ratio CSVs.

Filtering rule:
- For behavior ratios: drop all rows for a given (emotion, intensity) if behavior 'unknown' ratio > threshold.
- For choice ratios: drop all rows for a given (emotion, intensity) if option_id == -1 ratio > threshold.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class DroppedSlice:
    emotion: str
    intensity: float
    unknown_ratio: float


def _mean_map_int(values: Dict[int, List[float]]) -> Dict[int, float]:
    return {k: mean(v) for k, v in values.items()}


def _mean_map_str(values: Dict[str, List[float]]) -> Dict[str, float]:
    return {k: mean(v) for k, v in values.items()}


def load_choice_by_intensity(
    path: Path, *, unknown_threshold: Optional[float]
) -> Tuple[Dict[str, Dict[float, Dict[int, float]]], List[DroppedSlice]]:
    acc: Dict[str, Dict[float, Dict[int, List[float]]]] = {}
    unknown_by_slice: Dict[Tuple[str, float], List[float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            emotion = str(row["emotion"])
            intensity = float(row["intensity"])
            option_id = int(float(row["option_id"]))
            ratio = float(row["ratio"])
            acc.setdefault(emotion, {}).setdefault(intensity, {}).setdefault(option_id, []).append(ratio)
            if option_id == -1:
                unknown_by_slice.setdefault((emotion, intensity), []).append(ratio)

    dropped: List[DroppedSlice] = []
    if unknown_threshold is not None:
        for (emotion, intensity), vals in unknown_by_slice.items():
            unk = mean(vals)
            if unk > unknown_threshold:
                dropped.append(DroppedSlice(emotion=emotion, intensity=float(intensity), unknown_ratio=float(unk)))
                acc.get(emotion, {}).pop(float(intensity), None)
        for emotion, per_int in list(acc.items()):
            if not per_int:
                acc.pop(emotion, None)

    out: Dict[str, Dict[float, Dict[int, float]]] = {}
    for emo, per_int in acc.items():
        out[emo] = {i: _mean_map_int(m) for i, m in per_int.items()}
    return out, dropped


def load_behavior_by_intensity(
    path: Path, *, unknown_threshold: Optional[float]
) -> Tuple[Dict[str, Dict[float, Dict[str, float]]], List[DroppedSlice]]:
    acc: Dict[str, Dict[float, Dict[str, List[float]]]] = {}
    unknown_by_slice: Dict[Tuple[str, float], List[float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            emotion = str(row["emotion"])
            intensity = float(row["intensity"])
            behavior = row.get("behavior")
            if behavior in (None, "", "nan", "NaN"):
                behavior = row.get("behavior_label")
            if behavior in (None, ""):
                raise KeyError("Expected CSV column 'behavior' or 'behavior_label'")
            behavior = str(behavior)
            ratio = float(row["ratio"])
            acc.setdefault(emotion, {}).setdefault(intensity, {}).setdefault(behavior, []).append(ratio)
            if behavior.strip().lower() == "unknown":
                unknown_by_slice.setdefault((emotion, intensity), []).append(ratio)

    dropped: List[DroppedSlice] = []
    if unknown_threshold is not None:
        for (emotion, intensity), vals in unknown_by_slice.items():
            unk = mean(vals)
            if unk > unknown_threshold:
                dropped.append(DroppedSlice(emotion=emotion, intensity=float(intensity), unknown_ratio=float(unk)))
                acc.get(emotion, {}).pop(float(intensity), None)
        for emotion, per_int in list(acc.items()):
            if not per_int:
                acc.pop(emotion, None)

    out: Dict[str, Dict[float, Dict[str, float]]] = {}
    for emo, per_int in acc.items():
        out[emo] = {i: _mean_map_str(m) for i, m in per_int.items()}
    return out, dropped


def collapse_choice_over_intensity(
    path: Path, *, unknown_threshold: Optional[float]
) -> Tuple[Dict[str, Dict[int, float]], List[DroppedSlice]]:
    by_intensity, dropped = load_choice_by_intensity(path, unknown_threshold=unknown_threshold)
    acc: Dict[str, Dict[int, List[float]]] = {}
    for emo, per_int in by_intensity.items():
        for m in per_int.values():
            for opt, ratio in m.items():
                acc.setdefault(emo, {}).setdefault(opt, []).append(float(ratio))
    return {emo: _mean_map_int(m) for emo, m in acc.items()}, dropped


def collapse_behavior_over_intensity(
    path: Path, *, unknown_threshold: Optional[float]
) -> Tuple[Dict[str, Dict[str, float]], List[DroppedSlice]]:
    by_intensity, dropped = load_behavior_by_intensity(path, unknown_threshold=unknown_threshold)
    acc: Dict[str, Dict[str, List[float]]] = {}
    for emo, per_int in by_intensity.items():
        for m in per_int.values():
            for beh, ratio in m.items():
                acc.setdefault(emo, {}).setdefault(beh, []).append(float(ratio))
    return {emo: _mean_map_str(m) for emo, m in acc.items()}, dropped

