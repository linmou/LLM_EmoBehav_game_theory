"""
Load Prisoner's Dilemma defection vectors saved per layer.
"""

from pathlib import Path
from typing import Dict

import numpy as np


def load_pd_defection_vectors(directory: Path) -> Dict[int, np.ndarray]:
    """
    Load defection direction vectors from layer_{k}.npy files.
    Returns normalized float32 vectors keyed by layer index.
    Raises ValueError if none are found.
    """
    dir_path = Path(directory)
    vectors: Dict[int, np.ndarray] = {}

    for npy_path in sorted(dir_path.glob("layer_*.npy")):
        try:
            layer_idx = int(npy_path.stem.split("_")[1])
        except Exception:
            continue
        vec = np.load(npy_path)
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            vectors[layer_idx] = vec.astype(np.float32)
        else:
            vectors[layer_idx] = (vec / norm).astype(np.float32)

    if not vectors:
        raise ValueError(f"No PD defection vectors found in {dir_path}")

    return vectors
