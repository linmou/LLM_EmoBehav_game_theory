"""
Unit tests for neuro_manipulation package
"""

# Ensure PyTorch loads before HuggingFace/NumPy stacks.
# Some environments can abort when NumPy initializes OpenMP first.
import torch  # noqa: F401
