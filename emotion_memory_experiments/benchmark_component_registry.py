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

from .answer_wrapper import (
    AnswerWrapper,
    EmotionAnswerWrapper,
    IdentityAnswerWrapper,
)
from .benchmark_prompt_wrapper import get_benchmark_prompt_wrapper
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
    description: str = ""
    
    def create_components(
        self,
        config: BenchmarkConfig,
        prompt_format: Any,
        emotion: Optional[str] = None,
        enable_thinking: bool = False,
        augmentation_config: Optional[Dict] = None,
        user_messages: str = "Please provide your answer.",
        **dataset_kwargs
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
        # Create prompt wrapper using existing factory
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
            **dataset_kwargs
        )
        
        return prompt_wrapper_partial, answer_wrapper_partial, dataset


# Registry mapping (benchmark_name, task_type) to component specifications
# This is the single source of truth for component relationships
BENCHMARK_SPECS: Dict[Tuple[str, str], BenchmarkSpec] = {
    # MTBench101 tasks - use uppercase task types as per the actual dataset
    ("mtbench101", "CM"): BenchmarkSpec(
        dataset_class=MTBench101Dataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="MTBench101 Coding & Math tasks"
    ),
    ("mtbench101", "EX"): BenchmarkSpec(
        dataset_class=MTBench101Dataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="MTBench101 Extraction tasks"
    ),
    ("mtbench101", "HU"): BenchmarkSpec(
        dataset_class=MTBench101Dataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="MTBench101 Humanities tasks"
    ),
    ("mtbench101", "RO"): BenchmarkSpec(
        dataset_class=MTBench101Dataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="MTBench101 Roleplay tasks"
    ),
    ("mtbench101", "SI"): BenchmarkSpec(
        dataset_class=MTBench101Dataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="MTBench101 STEM Inquiry tasks"
    ),
    ("mtbench101", "WR"): BenchmarkSpec(
        dataset_class=MTBench101Dataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="MTBench101 Writing tasks"
    ),
    
    # Memory benchmarks - InfiniteBench
    ("infinitebench", "passkey"): BenchmarkSpec(
        dataset_class=InfiniteBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="InfiniteBench Passkey Retrieval"
    ),
    ("infinitebench", "number_string"): BenchmarkSpec(
        dataset_class=InfiniteBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="InfiniteBench Number String"
    ),
    ("infinitebench", "kv_retrieval"): BenchmarkSpec(
        dataset_class=InfiniteBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="InfiniteBench Key-Value Retrieval"
    ),
    ("infinitebench", "longbook_sum_eng"): BenchmarkSpec(
        dataset_class=InfiniteBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="InfiniteBench Long Book Summarization English"
    ),
    ("infinitebench", "longbook_choice_eng"): BenchmarkSpec(
        dataset_class=InfiniteBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="InfiniteBench Long Book Choice English"
    ),
    ("infinitebench", "longbook_qa_eng"): BenchmarkSpec(
        dataset_class=InfiniteBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="InfiniteBench Long Book QA English"
    ),
    ("infinitebench", "longbook_qa_chn"): BenchmarkSpec(
        dataset_class=InfiniteBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="InfiniteBench Long Book QA Chinese"
    ),
    ("infinitebench", "math_calc"): BenchmarkSpec(
        dataset_class=InfiniteBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="InfiniteBench Math Calculation"
    ),
    ("infinitebench", "math_find"): BenchmarkSpec(
        dataset_class=InfiniteBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="InfiniteBench Math Find"
    ),
    ("infinitebench", "code_run"): BenchmarkSpec(
        dataset_class=InfiniteBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="InfiniteBench Code Run"
    ),
    ("infinitebench", "code_debug"): BenchmarkSpec(
        dataset_class=InfiniteBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="InfiniteBench Code Debug"
    ),
    
    # Memory benchmarks - LongBench
    ("longbench", "narrativeqa"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="LongBench NarrativeQA"
    ),
    ("longbench", "qasper"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="LongBench Qasper"
    ),
    ("longbench", "multifieldqa_en"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="LongBench MultiFieldQA English"
    ),
    ("longbench", "multifieldqa_zh"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="LongBench MultiFieldQA Chinese"
    ),
    ("longbench", "hotpotqa"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="LongBench HotpotQA"
    ),
    ("longbench", "2wikimqa"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="LongBench 2WikiMQA"
    ),
    ("longbench", "musique"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="LongBench MuSiQue"
    ),
    ("longbench", "dureader"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="LongBench DuReader"
    ),
    ("longbench", "gov_report"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="LongBench Gov Report"
    ),
    ("longbench", "qmsum"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="LongBench QMSum"
    ),
    ("longbench", "multi_news"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="LongBench MultiNews"
    ),
    ("longbench", "vcsum"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="LongBench VCSum"
    ),
    ("longbench", "trec"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="LongBench TREC"
    ),
    ("longbench", "triviaqa"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="LongBench TriviaQA"
    ),
    ("longbench", "samsum"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="LongBench SAMSum"
    ),
    ("longbench", "lsht"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="LongBench LSHT"
    ),
    ("longbench", "passage_count"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="LongBench Passage Count"
    ),
    ("longbench", "passage_retrieval_en"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="LongBench Passage Retrieval English"
    ),
    ("longbench", "passage_retrieval_zh"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="LongBench Passage Retrieval Chinese"
    ),
    ("longbench", "lcc"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="LongBench LCC"
    ),
    ("longbench", "repobench-p"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="LongBench RepoBench-P"
    ),
    
    # LoCoMo benchmark
    ("locomo", "locomo"): BenchmarkSpec(
        dataset_class=LoCoMoDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="LoCoMo Long Context Modeling"
    ),
    
    # TruthfulQA benchmark
    ("truthfulqa", "truthfulqa"): BenchmarkSpec(
        dataset_class=TruthfulQADataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        description="TruthfulQA Truthfulness Assessment"
    ),
    
    # Emotion Check benchmark - special case with EmotionAnswerWrapper
    ("emotion_check", "emotion_check"): BenchmarkSpec(
        dataset_class=EmotionCheckDataset,
        answer_wrapper_class=EmotionAnswerWrapper,
        description="Emotion Check - Emotion Recognition Task"
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
    **dataset_kwargs
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
        **dataset_kwargs
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
    spec = BENCHMARK_SPECS.get(benchmark_key)
    return spec.description if spec else "Unknown combination"