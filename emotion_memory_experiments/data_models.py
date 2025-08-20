"""
Data models for emotion memory experiments.
Defines standard formats for results, configurations, and data structures.
"""

import glob
import re
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
    task_type: str  # e.g., 'passkey', 'kv_retrieval', 'longbook_qa_eng'
    data_path: Optional[Path] = None  # Auto-generated if None
    evaluation_method: Optional[str] = None  # Can be removed
    sample_limit: Optional[int] = None
    augmentation_config: Optional[Dict[str, str]] = None  # Custom prefix/suffix for context

    def discover_datasets_by_pattern(self, base_data_dir: str = "data/memory_benchmarks") -> List[str]:
        """
        Discover task types matching the regex pattern in task_type.
        
        Args:
            base_data_dir: Base directory for memory benchmark data
            
        Returns:
            List of task types matching the regex pattern
            
        Examples:
            - task_type='.*' -> ['narrativeqa', 'qasper'] (all tasks)
            - task_type='.*qa.*' -> ['narrativeqa'] (contains 'qa')
            - task_type='pass.*' -> ['passkey'] (starts with 'pass')
        """
        base_path = Path(base_data_dir)
        glob_pattern = str(base_path / f"{self.name}_*.jsonl")
        
        # Find all files for this benchmark
        all_files = glob.glob(glob_pattern)
        
        # Extract task types and filter by regex pattern
        task_types = []
        try:
            regex_pattern = re.compile(self.task_type)
            
            for file_path in all_files:
                filename = Path(file_path).stem  # Remove .jsonl extension
                prefix = f"{self.name}_"
                if filename.startswith(prefix):
                    task_type = filename[len(prefix):]
                    if regex_pattern.match(task_type):
                        task_types.append(task_type)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{self.task_type}': {str(e)}")
        
        return sorted(task_types)

    def get_data_path(self, base_data_dir: str = "data/memory_benchmarks") -> Path:
        """
        Get the data path for this benchmark. Auto-generates if not set.
        
        Args:
            base_data_dir: Base directory for memory benchmark data
            
        Returns:
            Path to the benchmark data file
            
        Examples:
            - name='longbench', task_type='narrativeqa' -> data/memory_benchmarks/longbench_narrativeqa.jsonl
            - name='infinitebench', task_type='passkey' -> data/memory_benchmarks/infinitebench_passkey.jsonl
        """
        if self.data_path is not None:
            return self.data_path
        
        # Auto-generate path based on naming convention
        filename = f"{self.name}_{self.task_type}.jsonl"
        return Path(base_data_dir) / filename


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
