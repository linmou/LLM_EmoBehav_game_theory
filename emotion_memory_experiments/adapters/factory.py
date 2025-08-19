"""
Factory functions for creating benchmark adapters.
"""

try:
    from .base_adapter import BenchmarkAdapter
    from .infinitebench_adapter import InfiniteBenchAdapter
    from .longbench_adapter import LongBenchAdapter
    from .locomo_adapter import LoCoMoAdapter
    from ..data_models import BenchmarkConfig
except ImportError:
    from emotion_memory_experiments.adapters.base_adapter import BenchmarkAdapter
    from emotion_memory_experiments.adapters.infinitebench_adapter import InfiniteBenchAdapter
    from emotion_memory_experiments.adapters.longbench_adapter import LongBenchAdapter
    from emotion_memory_experiments.adapters.locomo_adapter import LoCoMoAdapter
    from emotion_memory_experiments.data_models import BenchmarkConfig


def get_adapter(config: BenchmarkConfig) -> BenchmarkAdapter:
    """Factory function to get appropriate adapter for benchmark"""
    if config.name.lower() in ["infinitebench", "infinite_bench"]:
        return InfiniteBenchAdapter(config)
    elif config.name.lower() in ["locomo", "loco_mo"]:
        return LoCoMoAdapter(config)
    elif config.name.lower() in ["longbench", "long_bench"]:
        return LongBenchAdapter(config)
    else:
        raise ValueError(f"Unknown benchmark: {config.name}")