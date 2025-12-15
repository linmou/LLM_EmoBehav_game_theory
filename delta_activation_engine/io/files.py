"""
Responsible: delta_activation_engine/io/files.py
Purpose: Simple file utilities for saving arrays and metadata.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_npz_vector(path: str, vector: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    np.savez_compressed(path, vector=vector)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

