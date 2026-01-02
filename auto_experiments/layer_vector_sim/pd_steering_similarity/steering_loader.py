"""
Load per-layer emotion steering vectors (layer_{k}.npy files).
"""

from pathlib import Path
from typing import Dict

import numpy as np


def load_layer_vectors(directory: Path) -> Dict[int, np.ndarray]:
    """
    Load steering vectors saved as layer_{k}.npy in the given directory.

    Returns a dict mapping layer index to np.ndarray.
    Raises ValueError if no vectors are found.
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
        raise ValueError(f"No layer vectors found in {dir_path}")

    return vectors


def load_emotion_vectors(directory: Path) -> Dict[int, np.ndarray]:
    """
    Wrapper to load normalized emotion steering vectors.
    """
    target_dir = _resolve_layer_vector_dir(directory)
    return load_layer_vectors(target_dir)


def _resolve_layer_vector_dir(root: Path) -> Path:
    """
    Resolve the directory containing layer_{k}.npy vectors.
    Accepts either the layer_vectors directory itself or a parent that contains
    layer_vectors directly or under a seed_* subdirectory.
    """
    root = Path(root)
    if root.name == "layer_vectors":
        return root

    direct = root / "layer_vectors"
    if direct.exists():
        return direct

    for candidate in sorted(root.glob("seed_*")):
        lv = candidate / "layer_vectors"
        if lv.exists():
            return lv

    return root
