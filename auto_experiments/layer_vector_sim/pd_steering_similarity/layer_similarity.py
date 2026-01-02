"""
Per-layer similarity computation utilities.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .similarity_utils import cosine_similarity, similarity_delta


@dataclass
class LayerSimilarityRecord:
    sample_id: str
    steering_condition_id: str
    layer_index: int
    similarity_baseline: float
    similarity_steered: float
    similarity_delta: float


def compute_similarity_records(
    sample_id: str,
    steering_condition_id: str,
    hidden_baseline: Dict[int, np.ndarray],
    hidden_steered: Dict[int, np.ndarray],
    pd_defection_vectors: Dict[int, np.ndarray],
) -> List[LayerSimilarityRecord]:
    records: List[LayerSimilarityRecord] = []
    for layer_idx, pd_vec in pd_defection_vectors.items():
        if layer_idx not in hidden_baseline or layer_idx not in hidden_steered:
            continue
        base_sim = cosine_similarity(hidden_baseline[layer_idx], pd_vec)
        steered_sim = cosine_similarity(hidden_steered[layer_idx], pd_vec)
        delta = similarity_delta(base_sim, steered_sim)
        records.append(
            LayerSimilarityRecord(
                sample_id=sample_id,
                steering_condition_id=steering_condition_id,
                layer_index=layer_idx,
                similarity_baseline=base_sim,
                similarity_steered=steered_sim,
                similarity_delta=delta,
            )
        )
    return records
