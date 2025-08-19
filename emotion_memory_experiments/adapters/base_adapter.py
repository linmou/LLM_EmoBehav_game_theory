"""
Base adapter class for benchmark adapters with comprehensive evaluation capabilities.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from torch.utils.data import DataLoader, Dataset

try:
    from ..data_models import BenchmarkConfig
except ImportError:
    from emotion_memory_experiments.data_models import BenchmarkConfig


class BenchmarkAdapter(ABC):
    """Enhanced abstract base class for benchmark adapters with comprehensive evaluation capabilities"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self._dataset: Optional[Dataset] = None

    @abstractmethod
    def create_dataset(self, prompt_wrapper=None) -> Dataset:
        """Create simple PyTorch Dataset for this benchmark with optional prompt wrapper"""
        pass

    @abstractmethod
    def evaluate_response(
        self, response: str, ground_truth: Any, task_name: str
    ) -> float:
        """Evaluate a response against ground truth using benchmark-specific method"""
        pass

    def get_dataset(self, prompt_wrapper=None) -> Dataset:
        """Get dataset with caching and optional prompt wrapper"""
        # Note: We don't cache when prompt_wrapper is provided to allow different wrappers
        if prompt_wrapper is not None:
            return self.create_dataset(prompt_wrapper=prompt_wrapper)

        if self._dataset is None:
            self._dataset = self.create_dataset()
        return self._dataset

    def get_dataloader(
        self, batch_size: int, shuffle: bool = False, prompt_wrapper=None, **kwargs
    ) -> DataLoader:
        """Create DataLoader for efficient batching with optional prompt wrapper"""
        dataset = self.get_dataset(prompt_wrapper=prompt_wrapper)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

    def evaluate_batch(
        self, responses: List[str], ground_truths: List[Any], task_names: List[str]
    ) -> List[float]:
        """Batch evaluation for pipeline acceleration"""
        results = []
        for response, gt, task in zip(responses, ground_truths, task_names):
            score = self.evaluate_response(response, gt, task)
            results.append(score)
        return results

    def evaluate_by_length(
        self, 
        responses: List[str], 
        ground_truths: List[Any], 
        task_names: List[str],
        lengths: List[int]
    ) -> Dict[str, float]:
        """
        LongBench-E style evaluation by context length categories.
        Returns scores segmented by length: {"0-4k": score, "4-8k": score, "8k+": score}
        """
        scores = {"0-4k": [], "4-8k": [], "8k+": []}
        
        for response, gt, task, length in zip(responses, ground_truths, task_names, lengths):
            score = self.evaluate_response(response, gt, task)
            
            if length < 4000:
                scores["0-4k"].append(score)
            elif length < 8000:
                scores["4-8k"].append(score)
            else:
                scores["8k+"].append(score)
        
        # Compute average scores for each length category
        result = {}
        for category, score_list in scores.items():
            if score_list:
                result[category] = round(100 * sum(score_list) / len(score_list), 2)
            else:
                result[category] = 0.0
                
        return result

    def get_evaluation_complexity(self, task_name: str) -> str:
        """Determine evaluation complexity for pipeline optimization"""
        return "simple"  # Default implementation, override in subclasses

    def get_task_metrics(self, task_name: str) -> List[str]:
        """Return list of metrics available for this task"""
        return ["accuracy"]  # Default implementation, override in subclasses

    def evaluate_with_detailed_metrics(
        self, response: str, ground_truth: Any, task_name: str
    ) -> Dict[str, float]:
        """Return detailed metrics for comprehensive analysis"""
        base_score = self.evaluate_response(response, ground_truth, task_name)
        return {
            "overall_score": base_score,
            "accuracy": base_score
        }