"""
Abstract base class for all benchmark datasets.
Provides common functionality while enforcing specialized implementation requirements.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from torch.utils.data import Dataset

from ..data_models import BenchmarkConfig, BenchmarkItem


class BaseBenchmarkDataset(Dataset, ABC):
    """
    Abstract base class for all benchmark datasets.

    Provides common functionality:
    - Data loading infrastructure
    - Context truncation
    - PyTorch Dataset interface (__len__, __getitem__)
    - Batch collation

    Requires specialized implementation:
    - _load_and_parse_data(): Benchmark-specific data parsing
    - evaluate_response(): Benchmark-specific evaluation logic
    - get_task_metrics(): Available metrics for tasks
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        prompt_wrapper: Optional[Callable] = None,
        max_context_length: Optional[int] = None,
        tokenizer: Any = None,
        truncation_strategy: str = "right",
    ):
        self.config = config
        self.prompt_wrapper = prompt_wrapper
        self.max_context_length = max_context_length
        self.tokenizer = tokenizer
        self.truncation_strategy = truncation_strategy

        # Load data using child class implementation
        self.items: List[BenchmarkItem] = self._load_and_parse_data()

        # Apply sample limit with random shuffling if specified
        if self.config.sample_limit:
            self.items = self.items[: self.config.sample_limit]

        # Apply truncation if parameters provided
        if self.max_context_length and self.tokenizer:
            self.items = self._apply_truncation(self.items)

    @abstractmethod
    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        """
        Load and parse data for this specific benchmark type.
        Child classes must implement their own data loading logic.

        Returns:
            List of BenchmarkItem objects
        """
        pass

    @abstractmethod
    def evaluate_response(
        self, response: str, ground_truth: Any, task_name: str
    ) -> float:
        """
        Evaluate a response against ground truth for this benchmark.
        Child classes implement benchmark-specific evaluation logic.

        Args:
            response: Model's response text
            ground_truth: Expected answer(s)
            task_name: Specific task type within benchmark

        Returns:
            Score between 0.0 and 1.0
        """
        pass

    @abstractmethod
    def get_task_metrics(self, task_name: str) -> List[str]:
        """
        Get available metrics for a specific task type.

        Args:
            task_name: Task type within this benchmark

        Returns:
            List of available metric names (e.g., ["accuracy", "f1_score"])
        """
        pass

    # Common functionality implementation

    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index with prompt formatting"""
        item = self.items[idx]

        # Extract options from metadata for multiple choice questions
        options = None
        if item.metadata and 'options' in item.metadata:
            options = item.metadata['options']

        # Create prompt using wrapper or default format
        if self.prompt_wrapper:
            if item.context:
                prompt = self.prompt_wrapper(
                    item.context, item.input_text, answer=item.ground_truth, options=options
                )
            else:
                prompt = self.prompt_wrapper(
                    "", item.input_text, answer=item.ground_truth, options=options
                )
        else:
            # Default prompt format
            if item.context:
                prompt = (
                    f"Context: {item.context}\nQuestion: {item.input_text}\nAnswer:"
                )
            else:
                prompt = f"{item.input_text}\nAnswer:"

        return {"item": item, "prompt": prompt, "ground_truth": item.ground_truth}

    def collate_fn(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function for DataLoader"""
        return {
            "prompts": [item["prompt"] for item in batch_items],
            "items": [item["item"] for item in batch_items],
            "ground_truths": [item["ground_truth"] for item in batch_items],
        }

    def _apply_truncation(self, items: List[BenchmarkItem]) -> List[BenchmarkItem]:
        """Apply truncation to contexts that exceed max_context_length"""
        truncated_items = []

        for item in items:
            if not item.context:
                truncated_items.append(item)
                continue

            # Count tokens in current context
            context_tokens = self.tokenizer.encode(
                item.context, add_special_tokens=False
            )

            if len(context_tokens) <= self.max_context_length:
                # No truncation needed
                truncated_items.append(item)
            else:
                # Apply truncation based on strategy
                if self.truncation_strategy == "right":
                    # Keep leftmost tokens
                    truncated_tokens = context_tokens[: self.max_context_length]
                elif self.truncation_strategy == "left":
                    # Keep rightmost tokens
                    truncated_tokens = context_tokens[-self.max_context_length :]
                else:
                    # Default to right truncation
                    truncated_tokens = context_tokens[: self.max_context_length]

                # Decode back to text
                truncated_context = self.tokenizer.decode(
                    truncated_tokens, skip_special_tokens=True
                )

                # Create new item with truncated context
                truncated_item = BenchmarkItem(
                    id=item.id,
                    context=truncated_context,
                    input_text=item.input_text,
                    ground_truth=item.ground_truth,
                    metadata={
                        **(item.metadata or {}),
                        "truncation_info": {
                            "original_length": len(context_tokens),
                            "truncated_length": len(truncated_tokens),
                            "strategy": self.truncation_strategy,
                            "was_truncated": True,
                        },
                    },
                )
                truncated_items.append(truncated_item)

        return truncated_items

    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """
        Load raw JSON data from file.
        Common loading logic used by all benchmark types.
        """
        data_path = self.config.get_data_path()
        if not data_path.exists():
            raise FileNotFoundError(f"Benchmark data not found: {data_path}")

        # Load based on file extension
        if data_path.suffix == ".jsonl":
            with open(data_path, "r", encoding="utf-8") as f:
                raw_data = [json.loads(line) for line in f if line.strip()]
        else:
            with open(data_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

        return raw_data

    def evaluate_batch(
        self, responses: List[str], ground_truths: List[Any], task_names: List[str]
    ) -> List[float]:
        """
        Async batch evaluation using LLM evaluation.
        Handles async calls in synchronous context.
        """
        import asyncio
        from ..evaluation_utils import llm_evaluate_batch
        
        async def run_batch_evaluation():
            return await llm_evaluate_batch(responses, ground_truths, task_names)
        
        # Handle async execution in sync context
        try:
            # Try to get the current event loop
            try:
                loop = asyncio.get_running_loop()
                # Already in async context - create new event loop in thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(lambda: asyncio.run(run_batch_evaluation()))
                    return future.result()
            except RuntimeError:
                # No running event loop - can run directly
                return asyncio.run(run_batch_evaluation())
        except Exception as e:
            # Fallback to individual evaluation on async errors
            print(f"Batch evaluation failed: {e}, falling back to individual evaluation")
            results = []
            for response, gt, task in zip(responses, ground_truths, task_names):
                try:
                    # Use fallback evaluation for compatibility
                    score = 1.0 if str(response).strip().lower() == str(gt).strip().lower() else 0.0
                    results.append(score)
                except Exception:
                    results.append(0.0)
            return results

    def evaluate_with_detailed_metrics(
        self, response: str, ground_truth: Any, task_name: str
    ) -> Dict[str, float]:
        """
        Return detailed metrics for evaluation.
        Default implementation returns basic score - child classes can extend.
        """
        base_score = self.evaluate_response(response, ground_truth, task_name)
        return {"overall_score": base_score, "accuracy": base_score}
