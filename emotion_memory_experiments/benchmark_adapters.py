"""
Benchmark adapters for different memory benchmark suites.
This module provides backward compatibility by importing from the new modular structure.
"""

# Import everything from the new modular structure for backward compatibility
from .adapters import (
    BenchmarkAdapter,
    InfiniteBenchDataset,
    LongBenchDataset,
    LoCoMoDataset,
    collate_memory_benchmarks,
    InfiniteBenchAdapter,
    LongBenchAdapter,
    LoCoMoAdapter,
    get_adapter
)

# Import data models for backward compatibility
from .data_models import BenchmarkConfig, BenchmarkItem

# Re-export for backward compatibility
__all__ = [
    "BenchmarkAdapter",
    "InfiniteBenchDataset", 
    "LongBenchDataset",
    "LoCoMoDataset",
    "collate_memory_benchmarks",
    "InfiniteBenchAdapter",
    "LongBenchAdapter",
    "LoCoMoAdapter", 
    "get_adapter",
    "BenchmarkConfig",
    "BenchmarkItem"
]