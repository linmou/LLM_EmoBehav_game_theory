"""
Specialized dataset classes for benchmark evaluation.
Replaces the monolithic SmartMemoryBenchmarkDataset with specialized classes.
"""

from .base import BaseBenchmarkDataset
from .infinitebench import InfiniteBenchDataset
from .longbench import LongBenchDataset
from .locomo import LoCoMoDataset

__all__ = [
    "BaseBenchmarkDataset",
    "InfiniteBenchDataset", 
    "LongBenchDataset",
    "LoCoMoDataset"
]