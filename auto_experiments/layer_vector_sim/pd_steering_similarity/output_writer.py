"""
Writers for PD steering similarity outputs.
"""

import json
from pathlib import Path
from typing import Iterable, List

from .layer_similarity import LayerSimilarityRecord
from .group_aggregation import GroupSummary
from .emotion_aggregation import EmotionRanking


def write_similarity_records(records: Iterable[LayerSimilarityRecord], output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "similarity_records.json"
    payload = [
        {
            "sample_id": r.sample_id,
            "steering_condition_id": r.steering_condition_id,
            "layer_index": r.layer_index,
            "similarity_baseline": r.similarity_baseline,
            "similarity_steered": r.similarity_steered,
            "similarity_delta": r.similarity_delta,
        }
        for r in records
    ]
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def write_group_summaries(summaries: Iterable[GroupSummary], output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "group_summaries.json"
    payload = [
        {
            "steering_condition_id": s.steering_condition_id,
            "layer_index": s.layer_index,
            "group_label": s.group_label,
            "mean_similarity_delta": s.mean_similarity_delta,
            "std_similarity_delta": s.std_similarity_delta,
            "n_samples": s.n_samples,
        }
        for s in summaries
    ]
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def write_emotion_rankings(rankings: Iterable[EmotionRanking], output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "emotion_rankings.json"
    payload = [
        {
            "steering_condition_id": r.steering_condition_id,
            "mean_similarity_delta": r.mean_similarity_delta,
        }
        for r in rankings
    ]
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path
