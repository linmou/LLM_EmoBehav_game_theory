"""
Responsible: delta_activation_engine/backends/base.py
Purpose: Backend interface for computing representations given prompts.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np


class BaseBackend:
    def get_repr(
        self,
        prompts: List[str],
        *,
        steered: bool,
        emotion: Optional[str] = None,
        intensity: Optional[float] = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def get_run_metadata(self) -> dict:
        """Optional backend-specific metadata to persist alongside results."""
        return {}

