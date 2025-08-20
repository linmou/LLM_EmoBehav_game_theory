"""
Data models for emotion memory experiments.
Defines standard formats for results, configurations, and data structures.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class ResultRecord:
    """Standard format for individual experiment results"""

    emotion: str
    intensity: float
    item_id: Union[int, str]
    task_name: str
    prompt: str
    response: str
    ground_truth: Any
    score: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkConfig:
    """Configuration for a memory benchmark"""

    name: str
    data_path: Path
    task_type: str  # e.g., 'passkey', 'kv_retrieval', 'longbook_qa_eng'
    evaluation_method: str = None  # Can be removed
    sample_limit: Optional[int] = None


@dataclass
class LoadingConfig:
    """Configuration for vLLM model loading"""

    model_path: Optional[str] = None  # Model name or path to load
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: Optional[int] = None  # Auto-detect if None
    max_model_len: int = 32768
    enforce_eager: bool = True
    quantization: Optional[str] = None  # 'awq' for AWQ models
    trust_remote_code: bool = True
    dtype: str = "float16"  # Model dtype: 'float16', 'bfloat16', 'float32'
    seed: int = 42
    disable_custom_all_reduce: bool = False
    
    # Context truncation settings
    enable_auto_truncation: bool = True  # Enable automatic context truncation
    truncation_strategy: str = "right"  # "right" or "left" (via tokenizer)
    preserve_ratio: float = 0.95  # Ratio of max_model_len to use for context


@dataclass
class ExperimentConfig:
    """Configuration for the emotion memory experiment"""

    model_path: str
    emotions: List[str]
    intensities: List[float]
    benchmark: BenchmarkConfig
    output_dir: str
    batch_size: int = 4  # Number of items to process per batch for memory efficiency
    generation_config: Optional[Dict[str, Any]] = None
    loading_config: Optional[LoadingConfig] = None  # vLLM loading configuration

    repe_eng_config: Optional[Dict[str, Any]] = None

    # Pipeline settings (always enabled with DataLoader)
    max_evaluation_workers: int = 2  # Number of evaluation worker threads
    pipeline_queue_size: int = 2  # Max queued batches (controls memory usage)


@dataclass
class BenchmarkItem:
    """Standardized format for benchmark items after loading"""

    id: Union[int, str]
    input_text: str  # The prompt/question
    context: Optional[str] = None  # Long context if separate
    ground_truth: Any = None  # Expected answer
    metadata: Optional[Dict[str, Any]] = None  # Task-specific data


# Default generation config matching emotion_game_experiment.py
DEFAULT_GENERATION_CONFIG = {
    "temperature": 0.1,
    "max_new_tokens": 100,
    "do_sample": False,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "top_k": -1,  # -1 means no top_k filtering
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "enable_thinking": False,  # Qwen thinking mode support
}
