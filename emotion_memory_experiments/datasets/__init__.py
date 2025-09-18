"""
Specialized dataset classes namespace with lazy-loading to avoid heavy imports at package import time.
"""

from .base import BaseBenchmarkDataset

__all__ = [
    "BaseBenchmarkDataset",
    "EmotionCheckDataset",
    "InfiniteBenchDataset",
    "LongBenchDataset",
    "LoCoMoDataset",
    "FantomDataset",
    "BFCLDataset",
    "MTBench101Dataset",
    "TruthfulQADataset",
]

_LAZY_MAP = {
    "EmotionCheckDataset": (".emotion_check", "EmotionCheckDataset"),
    "InfiniteBenchDataset": (".infinitebench", "InfiniteBenchDataset"),
    "LongBenchDataset": (".longbench", "LongBenchDataset"),
    "LoCoMoDataset": (".locomo", "LoCoMoDataset"),
    "FantomDataset": (".fantom", "FantomDataset"),
    "BFCLDataset": (".bfcl", "BFCLDataset"),
    "MTBench101Dataset": (".mtbench101", "MTBench101Dataset"),
    "TruthfulQADataset": (".truthfulqa", "TruthfulQADataset"),
}


def __getattr__(name):  # pragma: no cover (thin shim)
    if name in _LAZY_MAP:
        mod_name, attr = _LAZY_MAP[name]
        from importlib import import_module

        mod = import_module(mod_name, package=__name__)
        return getattr(mod, attr)
    raise AttributeError(name)
