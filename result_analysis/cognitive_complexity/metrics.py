from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable


def _iter_rows(path: Path) -> Iterable[dict]:
    with open(path, newline="") as f:
        yield from csv.DictReader(f)


def neutral_avg_by_item(path_like) -> Dict[str, float]:
    """Return average neutral (intensity 0.0) score per item_id.

    - Aggregates over repeat_id within the given file.
    - Skips rows that fail to parse the score.
    """
    path = Path(path_like)
    buckets: Dict[str, list] = {}
    for row in _iter_rows(path):
        if row.get("emotion") != "neutral" or row.get("intensity") != "0.0":
            continue
        item_id = row.get("item_id")
        if not item_id:
            continue
        try:
            s = float(row.get("score", ""))
        except (TypeError, ValueError):
            continue
        buckets.setdefault(item_id, []).append(s)
    return {k: mean(v) for k, v in buckets.items() if v}


def cognitive_complexity_ratio(
    thinking_path, no_thinking_path, item_id: str
) -> float:
    """Compute avg_thinking_neutral(item)/avg_no_thinking_neutral(item).

    Raises ValueError if denominator is zero or missing.
    """
    a_t = neutral_avg_by_item(thinking_path).get(item_id)
    a_n = neutral_avg_by_item(no_thinking_path).get(item_id)
    if a_t is None or a_n is None:
        raise ValueError(f"Missing averages for {item_id}")
    if a_n == 0.0:
        raise ValueError(f"Denominator is zero for {item_id}")
    return a_t / a_n

