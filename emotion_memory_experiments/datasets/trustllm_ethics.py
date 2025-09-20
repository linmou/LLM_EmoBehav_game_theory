from typing import Any, List
from .trustllm_base import _TrustLLMFamilyDataset
from ..data_models import BenchmarkItem


class TrustLLMEthicsDataset(_TrustLLMFamilyDataset):
    FAMILY = "ethics"

    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        # Will be implemented in Phase 2
        return []

    def compute_split_metrics(self, records):
        # Minimal first metric: overall mean of item scores
        scores = [r.score for r in records if getattr(r, "score", None) is not None]
        if not scores:
            return {"overall": 0.0}
        return {"overall": float(sum(scores) / len(scores))}
