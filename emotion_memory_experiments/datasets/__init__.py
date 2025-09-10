"""
Specialized dataset classes for benchmark evaluation.
Replaces the monolithic SmartMemoryBenchmarkDataset with specialized classes.
"""

from .base import BaseBenchmarkDataset
from .emotion_check import EmotionCheckDataset
from .infinitebench import InfiniteBenchDataset
from .longbench import LongBenchDataset
from .locomo import LoCoMoDataset

__all__ = [
    "BaseBenchmarkDataset",
    "EmotionCheckDataset",
    "InfiniteBenchDataset", 
    "LongBenchDataset",
    "LoCoMoDataset"
]