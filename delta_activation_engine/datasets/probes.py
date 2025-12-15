"""
Responsible: delta_activation_engine/datasets/probes.py
Purpose: Dataset that adapts a list of probe strings to BenchmarkItem entries.
"""

from __future__ import annotations

from typing import Any, List, Optional

from emotion_experiment_engine.data_models import BenchmarkConfig, BenchmarkItem
from emotion_experiment_engine.datasets.base import BaseBenchmarkDataset


class DeltaProbesDataset(BaseBenchmarkDataset):
    def __init__(
        self,
        config: BenchmarkConfig,
        prompt_wrapper,
        max_context_length: Optional[int] = None,
        tokenizer: Any = None,
        truncation_strategy: str = "right",
        answer_wrapper=None,
        *,
        probes: List[str],
        **kwargs,
    ) -> None:
        self._probes = list(probes or [])
        super().__init__(
            config=config,
            prompt_wrapper=prompt_wrapper,
            max_context_length=max_context_length,
            tokenizer=tokenizer,
            truncation_strategy=truncation_strategy,
            answer_wrapper=answer_wrapper,
        )

    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        items: List[BenchmarkItem] = []
        for i, text in enumerate(self._probes):
            items.append(
                BenchmarkItem(
                    id=i,
                    input_text=str(text),
                    context=None,
                    ground_truth="",
                    metadata=None,
                )
            )
        return items

    def evaluate_response(self, response: str, ground_truth: Any, task_name: str, prompt: str) -> float:
        return 0.0

    def get_task_metrics(self, task_name: str) -> List[str]:
        return []

