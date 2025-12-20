"""
Threading-related environment guards.

This repo uses vLLM, which spawns a subprocess during model inspection. On some
conda/HPC environments, `MKL_THREADING_LAYER=INTEL` can crash with `libgomp`.
We set a safe threading layer proactively.
"""

from __future__ import annotations

import os


def ensure_mkl_threading_layer() -> None:
    """Ensure MKL threading layer won't crash with GNU OpenMP (`libgomp`)."""
    layer = os.environ.get("MKL_THREADING_LAYER")
    if layer is None or layer.strip() == "":
        os.environ["MKL_THREADING_LAYER"] = "GNU"
        return

    if layer.strip().upper() == "INTEL":
        os.environ["MKL_THREADING_LAYER"] = "GNU"

