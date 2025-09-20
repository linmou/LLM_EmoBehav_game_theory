from typing import Any, List
from .trustllm_base import _TrustLLMFamilyDataset
from ..data_models import BenchmarkItem


class TrustLLMSafetyDataset(_TrustLLMFamilyDataset):
    FAMILY = "safety"

    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        return []

