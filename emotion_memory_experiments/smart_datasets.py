"""
DEPRECATED: smart_datasets.py has been refactored into specialized dataset classes.

This module provides backward compatibility by redirecting to the new architecture.

For new code, use:
    from emotion_memory_experiments.dataset_factory import create_dataset_from_config
    from emotion_memory_experiments.datasets.base import BaseBenchmarkDataset

For legacy compatibility:
    from emotion_memory_experiments.smart_datasets import SmartMemoryBenchmarkDataset, create_dataset_from_config
"""

import warnings

# New imports
from .dataset_factory import create_dataset_from_config as _new_create_dataset_from_config
from .datasets.base import BaseBenchmarkDataset

# Deprecation warning
warnings.warn(
    "smart_datasets.py is deprecated. Use dataset_factory and specialized dataset classes instead.",
    DeprecationWarning,
    stacklevel=2
)


def create_dataset_from_config(config, **kwargs):
    """Backward compatibility wrapper for create_dataset_from_config."""
    warnings.warn(
        "Import create_dataset_from_config from dataset_factory instead of smart_datasets",
        DeprecationWarning,
        stacklevel=2
    )
    return _new_create_dataset_from_config(config=config, **kwargs)


def get_dataset_from_config(config, **kwargs):
    """Legacy alias for create_dataset_from_config."""
    warnings.warn(
        "get_dataset_from_config is deprecated. Use create_dataset_from_config instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _new_create_dataset_from_config(config=config, **kwargs)


# Backward compatibility class alias
SmartMemoryBenchmarkDataset = BaseBenchmarkDataset

# Make sure the legacy class has the same interface
SmartMemoryBenchmarkDataset.__doc__ = """
DEPRECATED: SmartMemoryBenchmarkDataset has been refactored into specialized dataset classes.
This is now an alias to BaseBenchmarkDataset for backward compatibility.
Use the new specialized dataset classes from emotion_memory_experiments.datasets.
"""