"""
Abstract base class for all benchmark datasets.
Provides common functionality while enforcing specialized implementation requirements.
"""

import json
from abc import ABC, abstractmethod
from copy import deepcopy
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

    LLM_EVAL_CONFIG = {
        "model": "gpt-4o-mini",
        "temperature": 0.0,
    }

    def __init__(
        self,
        config: BenchmarkConfig,
        prompt_wrapper: Optional[Callable],
        max_context_length: Optional[int] = None,
        tokenizer: Any = None,
        truncation_strategy: str = "right",
    ):
        self.config = config
        self.prompt_wrapper = prompt_wrapper
        self.max_context_length = max_context_length
        self.tokenizer = tokenizer
        self.truncation_strategy = truncation_strategy
        if hasattr(self, "LLM_EVAL_CONFIG"):
            self.llm_eval_config = deepcopy(self.LLM_EVAL_CONFIG)
            if self.config.llm_eval_config:
                self.llm_eval_config.update(self.config.llm_eval_config)
        else:
            self.llm_eval_config = self.config.llm_eval_config

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
        self, response: str, ground_truth: Any, task_name: str, prompt: str
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
        if item.metadata and "options" in item.metadata:
            options = item.metadata["options"]

        # Create prompt using wrapper or default format
        if self.prompt_wrapper:
            prompt = self.prompt_wrapper(
                context=item.context if item.context else "",
                question=item.input_text,
                answer=item.ground_truth,
                options=options,
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
        """
        Apply truncation to contexts that exceed max_context_length, processing
        in batches to manage memory usage.
        """
        # Define a batch size. This should ideally be an attribute of your class,
        # e.g., self.batch_size, configured during initialization.
        batch_size = 32  # You can adjust this value based on your available memory.

        all_truncated_items = []
        self.tokenizer.truncation_side = self.truncation_strategy

        # Process the items in batches
        for i in range(0, len(items), batch_size):
            # Create a slice of the list for the current batch
            batch_items = items[i : i + batch_size]

            # Separate items with None contexts from those that need processing
            items_to_process = []
            contexts_to_tokenize = []
            original_indices = []

            for j, item in enumerate(batch_items):
                if item.context is not None:
                    items_to_process.append(item)
                    contexts_to_tokenize.append(item.context or "")
                    original_indices.append(j)

            # Process contexts that are not None
            if contexts_to_tokenize:
                # Tokenize the batch of contexts
                tokenized_result = self.tokenizer(
                    contexts_to_tokenize,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_context_length,
                    return_tensors=None,  # Return lists, not tensors
                    padding=False,  # No padding needed
                )

                # Decode the batch of tokenized inputs back to text
                truncated_contexts = self.tokenizer.batch_decode(
                    tokenized_result["input_ids"], skip_special_tokens=True
                )
            else:
                truncated_contexts = []

            # Create new BenchmarkItem objects for the processed batch
            processed_idx = 0
            for j, item in enumerate(batch_items):
                if item.context is not None:
                    # Process items that had contexts
                    original_context = item.context or ""
                    truncated_context = truncated_contexts[processed_idx]
                    processed_idx += 1

                    # Always add truncation_info for processed contexts
                    metadata = deepcopy(item.metadata) if item.metadata else {}
                    metadata["truncation_info"] = {
                        "original_length": len(original_context),
                        "truncated_length": len(truncated_context),
                        "strategy": self.truncation_strategy,
                        "was_truncated": len(truncated_context) < len(original_context),
                    }

                    truncated_item = BenchmarkItem(
                        id=item.id,
                        context=truncated_context,
                        input_text=item.input_text,
                        ground_truth=item.ground_truth,
                        metadata=metadata,
                    )
                else:
                    # Preserve None contexts as-is
                    truncated_item = BenchmarkItem(
                        id=item.id,
                        context=None,
                        input_text=item.input_text,
                        ground_truth=item.ground_truth,
                        metadata=deepcopy(item.metadata) if item.metadata else None,
                    )

                all_truncated_items.append(truncated_item)

        return all_truncated_items

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
        self,
        responses: List[str],
        ground_truths: List[Any],
        task_names: List[str],
        prompts: List[str],
    ) -> List[float]:
        """
        ThreadPoolExecutor-based batch evaluation using individual evaluate_response calls.
        """
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(8, len(responses))) as executor:
            futures = [
                executor.submit(self.evaluate_response, resp, gt, task, prompt)
                for resp, gt, task, prompt in zip(
                    responses, ground_truths, task_names, prompts
                )
            ]
            return [future.result() for future in futures]

    def evaluate_with_detailed_metrics(
        self, response: str, ground_truth: Any, task_name: str
    ) -> Dict[str, float]:
        """
        Return detailed metrics for evaluation.
        Default implementation returns basic score - child classes can extend.
        """
        base_score = self.evaluate_response(response, ground_truth, task_name)
        return {"overall_score": base_score, "accuracy": base_score}
