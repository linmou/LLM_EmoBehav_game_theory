#!/usr/bin/env python3
"""
Component Specification Registry for Benchmark Processing

This module provides a single source of truth for the relationships between
benchmark processing components: prompt_wrapper, answer_wrapper, and dataset classes.

The registry eliminates the need to manually assemble these components in
experiment code and ensures consistency across different benchmark configurations.

Key Features:
- BenchmarkSpec: Encapsulates the three component types needed for each benchmark
- BENCHMARK_SPECS: Registry mapping (benchmark_name, task_type) to specifications
- create_benchmark_components(): Factory function that assembles all components
- Type safety and clear error messages for unknown combinations
- Easy extension for new benchmarks

Example Usage:
    from .benchmark_component_registry import create_benchmark_components

    prompt_wrapper, answer_wrapper, dataset = create_benchmark_components(
        benchmark_name="mtbench101",
        task_type="CM",
        config=config,
        prompt_format=prompt_format,
        emotion="anger"
    )
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from neuro_manipulation.prompt_wrapper import PromptWrapper

from .answer_wrapper import AnswerWrapper, EmotionAnswerWrapper, IdentityAnswerWrapper
from .benchmark_prompt_wrapper import get_benchmark_prompt_wrapper
from .mtbench101_prompt_wrapper import MTBench101PromptWrapper
from .truthfulqa_prompt_wrapper import TruthfulQAPromptWrapper
from .memory_prompt_wrapper import (
    MemoryPromptWrapper,
    PasskeyPromptWrapper,
    ConversationalQAPromptWrapper,
    LongContextQAPromptWrapper,
    LongbenchRetrievalPromptWrapper,
    EmotionCheckPromptWrapper,
)
from .data_models import BenchmarkConfig
from .dataset_factory import create_dataset_from_config
from .datasets.base import BaseBenchmarkDataset
from .datasets.emotion_check import EmotionCheckDataset
from .datasets.infinitebench import InfiniteBenchDataset
from .datasets.locomo import LoCoMoDataset
from .datasets.longbench import LongBenchDataset
from .datasets.mtbench101 import MTBench101Dataset
from .datasets.truthfulqa import TruthfulQADataset


@dataclass
class BenchmarkSpec:
    """
    Specification of all components needed to process a specific benchmark task.

    This encapsulates the knowledge of which prompt_wrapper, answer_wrapper,
    and dataset class should be used together for a given benchmark and task.
    """

    dataset_class: Type[BaseBenchmarkDataset]
    answer_wrapper_class: Type[AnswerWrapper]

    prompt_wrapper_class: Optional[Type[PromptWrapper]] = None

    def create_components(
        self,
        config: BenchmarkConfig,
        prompt_format: Any,
        emotion: Optional[str] = None,
        enable_thinking: bool = False,
        augmentation_config: Optional[Dict] = None,
        user_messages: str = "Please provide your answer.",
        **dataset_kwargs,
    ) -> Tuple[Callable, Callable, BaseBenchmarkDataset]:
        """
        Create all three components using this specification.

        Args:
            config: BenchmarkConfig for the dataset
            prompt_format: PromptFormat instance for the model
            emotion: Emotion context for processing
            enable_thinking: Whether to enable thinking mode
            augmentation_config: Configuration for prompt augmentation
            user_messages: User message template
            **dataset_kwargs: Additional arguments for dataset creation

        Returns:
            Tuple of (prompt_wrapper_partial, answer_wrapper_partial, dataset)
        """
        # Create prompt wrapper using explicit class when provided,
        # otherwise fall back to the factory selector.
        if self.prompt_wrapper_class is not None:
            try:
                # Some wrappers accept (prompt_format, task_type)
                prompt_wrapper = self.prompt_wrapper_class(
                    prompt_format, config.task_type
                )
            except TypeError:
                # Others accept only (prompt_format)
                prompt_wrapper = self.prompt_wrapper_class(prompt_format)
        else:
            prompt_wrapper = get_benchmark_prompt_wrapper(
                config.name, config.task_type, prompt_format
            )

        # Create partial function with emotion-specific parameters
        prompt_wrapper_partial = partial(
            prompt_wrapper.__call__,
            user_messages=user_messages,
            enable_thinking=enable_thinking,
            augmentation_config=augmentation_config,
            emotion=emotion,
        )

        # Create answer wrapper instance
        answer_wrapper = self.answer_wrapper_class()

        # Create partial function with emotion context
        answer_wrapper_partial = partial(
            answer_wrapper.__call__,
            emotion=emotion,
            benchmark_name=config.name,
            task_type=config.task_type,
        )

        # Create dataset using existing factory (maintains current architecture)
        dataset = create_dataset_from_config(
            config,
            prompt_wrapper=prompt_wrapper_partial,
            answer_wrapper=answer_wrapper_partial,
            **dataset_kwargs,
        )

        return prompt_wrapper_partial, answer_wrapper_partial, dataset


# Registry mapping (benchmark_name, task_type) to component specifications
# This is the single source of truth for component relationships
BENCHMARK_SPECS: Dict[Tuple[str, str], BenchmarkSpec] = {
    # MTBench101 tasks - use uppercase task types as per the actual dataset
    ("mtbench101", "CM"): BenchmarkSpec(
        dataset_class=MTBench101Dataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=MTBench101PromptWrapper,
    ),
    ("mtbench101", "EX"): BenchmarkSpec(
        dataset_class=MTBench101Dataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=MTBench101PromptWrapper,
    ),
    ("mtbench101", "HU"): BenchmarkSpec(
        dataset_class=MTBench101Dataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=MTBench101PromptWrapper,
    ),
    ("mtbench101", "RO"): BenchmarkSpec(
        dataset_class=MTBench101Dataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=MTBench101PromptWrapper,
    ),
    ("mtbench101", "SI"): BenchmarkSpec(
        dataset_class=MTBench101Dataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=MTBench101PromptWrapper,
    ),
    ("mtbench101", "WR"): BenchmarkSpec(
        dataset_class=MTBench101Dataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=MTBench101PromptWrapper,
    ),
    # Memory benchmarks - InfiniteBench
    ("infinitebench", "passkey"): BenchmarkSpec(
        dataset_class=InfiniteBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=PasskeyPromptWrapper,
    ),
    ("infinitebench", "number_string"): BenchmarkSpec(
        dataset_class=InfiniteBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=MemoryPromptWrapper,
    ),
    ("infinitebench", "kv_retrieval"): BenchmarkSpec(
        dataset_class=InfiniteBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=MemoryPromptWrapper,
    ),
    ("infinitebench", "longbook_sum_eng"): BenchmarkSpec(
        dataset_class=InfiniteBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=MemoryPromptWrapper,
    ),
    ("infinitebench", "longbook_choice_eng"): BenchmarkSpec(
        dataset_class=InfiniteBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=MemoryPromptWrapper,
    ),
    ("infinitebench", "longbook_qa_eng"): BenchmarkSpec(
        dataset_class=InfiniteBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=LongContextQAPromptWrapper,
    ),
    ("infinitebench", "longbook_qa_chn"): BenchmarkSpec(
        dataset_class=InfiniteBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=LongContextQAPromptWrapper,
    ),
    ("infinitebench", "math_calc"): BenchmarkSpec(
        dataset_class=InfiniteBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=MemoryPromptWrapper,
    ),
    ("infinitebench", "math_find"): BenchmarkSpec(
        dataset_class=InfiniteBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=MemoryPromptWrapper,
    ),
    ("infinitebench", "code_run"): BenchmarkSpec(
        dataset_class=InfiniteBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=MemoryPromptWrapper,
    ),
    ("infinitebench", "code_debug"): BenchmarkSpec(
        dataset_class=InfiniteBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=MemoryPromptWrapper,
    ),
    # Memory benchmarks - LongBench
    ("longbench", "narrativeqa"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=LongContextQAPromptWrapper,
    ),
    ("longbench", "qasper"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=LongContextQAPromptWrapper,
    ),
    ("longbench", "multifieldqa_en"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=LongContextQAPromptWrapper,
    ),
    ("longbench", "multifieldqa_zh"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
    ),
    ("longbench", "hotpotqa"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=LongContextQAPromptWrapper,
    ),
    ("longbench", "2wikimqa"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
    ),
    ("longbench", "musique"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
    ),
    ("longbench", "dureader"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
    ),
    ("longbench", "gov_report"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
    ),
    ("longbench", "qmsum"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
    ),
    ("longbench", "multi_news"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
    ),
    ("longbench", "vcsum"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
    ),
    ("longbench", "trec"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
    ),
    ("longbench", "triviaqa"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
    ),
    ("longbench", "samsum"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
    ),
    ("longbench", "lsht"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
    ),
    ("longbench", "passage_count"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
    ),
    ("longbench", "passage_retrieval_en"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=LongbenchRetrievalPromptWrapper,
    ),
    ("longbench", "passage_retrieval_zh"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=LongbenchRetrievalPromptWrapper,
    ),
    ("longbench", "lcc"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
    ),
    ("longbench", "repobench-p"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
    ),
    # LoCoMo benchmark
    ("locomo", "locomo"): BenchmarkSpec(
        dataset_class=LoCoMoDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=ConversationalQAPromptWrapper,
    ),
    # TruthfulQA benchmark
    ("truthfulqa", "mc1"): BenchmarkSpec(
        dataset_class=TruthfulQADataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=TruthfulQAPromptWrapper,
    ),
    ("truthfulqa", "mc2"): BenchmarkSpec(
        dataset_class=TruthfulQADataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=TruthfulQAPromptWrapper,
    ),
    # Emotion Check benchmark - special case with EmotionAnswerWrapper
    ("emotion_check", "emotion_check"): BenchmarkSpec(
        dataset_class=EmotionCheckDataset,
        answer_wrapper_class=EmotionAnswerWrapper,
        prompt_wrapper_class=EmotionCheckPromptWrapper,
    ),
    # New academic scale task using the same dataset and wrapper
    ("emotion_check", "academic_scale"): BenchmarkSpec(
        dataset_class=EmotionCheckDataset,
        answer_wrapper_class=EmotionAnswerWrapper,
        prompt_wrapper_class=EmotionCheckPromptWrapper,
    ),
}


def create_benchmark_components(
    benchmark_name: str,
    task_type: str,
    config: BenchmarkConfig,
    prompt_format: Any,
    emotion: Optional[str] = None,
    enable_thinking: bool = False,
    augmentation_config: Optional[Dict] = None,
    user_messages: str = "Please provide your answer.",
    **dataset_kwargs,
) -> Tuple[Callable, Callable, BaseBenchmarkDataset]:
    """
    Factory function to create all benchmark processing components from registry.

    This is the main entry point that replaces manual component assembly
    in experiment code. It uses the BENCHMARK_SPECS registry to ensure
    consistent component relationships.

    Args:
        benchmark_name: Name of the benchmark (e.g., "mtbench101", "infinitebench")
        task_type: Specific task within the benchmark (e.g., "CM", "passkey")
        config: BenchmarkConfig for dataset creation
        prompt_format: PromptFormat instance for the model
        emotion: Optional emotion context for processing
        enable_thinking: Whether to enable thinking mode in prompts
        augmentation_config: Configuration for prompt augmentation
        user_messages: Template for user messages in prompts
        **dataset_kwargs: Additional arguments passed to dataset creation

    Returns:
        Tuple of (prompt_wrapper_partial, answer_wrapper_partial, dataset)

    Raises:
        KeyError: If the (benchmark_name, task_type) combination is not registered

    Example:
        prompt_wrapper, answer_wrapper, dataset = create_benchmark_components(
            benchmark_name="mtbench101",
            task_type="CM",
            config=config,
            prompt_format=prompt_format,
            emotion="anger"
        )
    """
    # Normalize benchmark name for consistent lookup, but preserve task_type case
    benchmark_key = (benchmark_name.lower(), task_type)

    if benchmark_key not in BENCHMARK_SPECS:
        # Provide helpful error message with available options
        available_combinations = list(BENCHMARK_SPECS.keys())
        raise KeyError(
            f"Unknown benchmark combination: ({benchmark_name}, {task_type}). "
            f"Available combinations: {available_combinations}"
        )

    spec = BENCHMARK_SPECS[benchmark_key]

    return spec.create_components(
        config=config,
        prompt_format=prompt_format,
        emotion=emotion,
        enable_thinking=enable_thinking,
        augmentation_config=augmentation_config,
        user_messages=user_messages,
        **dataset_kwargs,
    )


def list_available_benchmarks() -> Dict[str, List[str]]:
    """
    Get a summary of all available benchmark and task combinations.

    Returns:
        Dictionary mapping benchmark names to lists of available tasks
    """
    benchmarks = {}
    for (benchmark_name, task_type), spec in BENCHMARK_SPECS.items():
        if benchmark_name not in benchmarks:
            benchmarks[benchmark_name] = []
        benchmarks[benchmark_name].append(task_type)

    return benchmarks


def get_benchmark_description(benchmark_name: str, task_type: str) -> str:
    """
    Get description for a specific benchmark and task combination.

    Args:
        benchmark_name: Name of the benchmark
        task_type: Task type within the benchmark

    Returns:
        Description string, or "Unknown combination" if not found
    """
    benchmark_key = (benchmark_name.lower(), task_type)
    # Description field removed from BenchmarkSpec. Provide a simple fallback.
    if benchmark_key in BENCHMARK_SPECS:
        return f"{benchmark_name} - {task_type}"
    return "Unknown combination"
