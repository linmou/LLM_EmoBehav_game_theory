from typing import Any, List
from .trustllm_base import _TrustLLMFamilyDataset
from ..data_models import BenchmarkItem


class TrustLLMPrivacyDataset(_TrustLLMFamilyDataset):
    FAMILY = "privacy"

    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        return []

