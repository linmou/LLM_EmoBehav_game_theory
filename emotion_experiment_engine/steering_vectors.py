"""
Responsible file: emotion_experiment_engine/steering_vectors.py
Purpose: Load per-layer steering vectors (e.g., PD defection directions) from
         `layer_vectors/` directories into a minimal RepReader-like object.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


class LayerVectorReader:
    def __init__(self, directions: Dict[int, np.ndarray]) -> None:
        self.directions = directions
        self.direction_signs = {layer: 1.0 for layer in directions}


def load_layer_vectors_dir(vectors_dir: Path) -> Dict[int, np.ndarray]:
    vectors_dir = Path(vectors_dir)
    if not vectors_dir.exists():
        raise FileNotFoundError(f"Steering vector directory not found: {vectors_dir}")
    if not vectors_dir.is_dir():
        raise NotADirectoryError(f"Expected directory: {vectors_dir}")

    out: Dict[int, np.ndarray] = {}
    for path in sorted(vectors_dir.glob("layer_*.npy")):
        name = path.stem  # layer_12
        try:
            layer = int(name.split("_", 1)[1])
        except Exception as exc:
            raise ValueError(f"Invalid layer vector filename: {path.name}") from exc

        vec = np.load(path).astype(np.float32, copy=False)
        if vec.ndim != 1:
            raise ValueError(f"Expected 1D vector in {path}, got shape {vec.shape}")
        out[layer] = vec

    if not out:
        raise FileNotFoundError(f"No layer_*.npy files found in {vectors_dir}")
    return out
