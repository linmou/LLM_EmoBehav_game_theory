from typing import Any, List
from .trustllm_base import _TrustLLMFamilyDataset
from ..data_models import BenchmarkItem


class TrustLLMFairnessDataset(_TrustLLMFamilyDataset):
    FAMILY = "fairness"

    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        return []

