"""
Data models for emotion memory experiments.
Defines standard formats for results, configurations, and data structures.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from pathlib import Path


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
    evaluation_method: str  # Maps to function in compute_scores.py
    sample_limit: Optional[int] = None


@dataclass
class ExperimentConfig:
    """Configuration for the emotion memory experiment"""
    model_path: str
    emotions: List[str]
    intensities: List[float]
    benchmark: BenchmarkConfig
    output_dir: str
    batch_size: int = 4
    generation_config: Optional[Dict[str, Any]] = None


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
    "top_p": 0.9
}