"""
Group aggregation for similarity records.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

from .layer_similarity import LayerSimilarityRecord


@dataclass
class GroupSummary:
    steering_condition_id: str
    layer_index: int
    group_label: str
    mean_similarity_delta: float
    std_similarity_delta: float
    n_samples: int


def aggregate_by_group(
    records: Iterable[LayerSimilarityRecord],
    group_labels: Dict[str, str],
) -> List[GroupSummary]:
    summaries: List[GroupSummary] = []
    # group -> layer -> list of deltas
    buckets: Dict[tuple, List[float]] = {}

    for rec in records:
        group_label = group_labels.get(rec.sample_id)
        if group_label is None:
            continue
        key = (rec.steering_condition_id, rec.layer_index, group_label)
        buckets.setdefault(key, []).append(rec.similarity_delta)

    for (steering_condition_id, layer_index, group_label), deltas in buckets.items():
        arr = np.array(deltas, dtype=np.float32)
        summaries.append(
            GroupSummary(
                steering_condition_id=steering_condition_id,
                layer_index=layer_index,
                group_label=group_label,
                mean_similarity_delta=float(arr.mean()),
                std_similarity_delta=float(arr.std(ddof=0)) if len(arr) > 1 else 0.0,
                n_samples=len(arr),
            )
        )

    return summaries
