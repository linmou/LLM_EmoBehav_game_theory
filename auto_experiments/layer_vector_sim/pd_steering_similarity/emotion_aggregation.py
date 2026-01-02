"""
Emotion-level aggregation and ranking for PD steering similarity.
"""

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class GroupSummaryInput:
    steering_condition_id: str
    mean_similarity_delta: float


@dataclass
class EmotionRanking:
    steering_condition_id: str
    mean_similarity_delta: float


def rank_emotions(summaries: Iterable[GroupSummaryInput]) -> List[EmotionRanking]:
    rankings = [
        EmotionRanking(
            steering_condition_id=s.steering_condition_id,
            mean_similarity_delta=s.mean_similarity_delta,
        )
        for s in summaries
    ]
    rankings.sort(key=lambda r: r.mean_similarity_delta, reverse=True)
    return rankings
