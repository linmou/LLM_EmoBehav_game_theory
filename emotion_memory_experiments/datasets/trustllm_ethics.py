from typing import Any, List
from .trustllm_base import _TrustLLMFamilyDataset
from ..data_models import BenchmarkItem


class TrustLLMEthicsDataset(_TrustLLMFamilyDataset):
    FAMILY = "ethics"

    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        # Will be implemented in Phase 2
        return []

