"""
Responsible: delta_activation_engine/config/job.py
Purpose: Strict YAML parsing for base (non-chat) delta activation jobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, TextIO

import yaml


@dataclass
class DeltaActivationJobConfig:
    model_path: str
    emotions: List[str]
    intensities: List[float]
    output_dir: str
    loading_config: Dict[str, Any]
    repe_eng_config: Dict[str, Any]


def _ensure_required(data: Dict[str, Any], keys: List[str]) -> None:
    missing = [k for k in keys if k not in data or data[k] is None]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")


def load_job_config_from_yaml(stream_or_path: TextIO | str) -> DeltaActivationJobConfig:
    if hasattr(stream_or_path, "read"):
        data = yaml.safe_load(stream_or_path)
    else:
        with open(stream_or_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Top-level YAML must be a mapping")

    _ensure_required(
        data,
        [
            "model_path",
            "emotions",
            "intensities",
            "output_dir",
            "loading_config",
            "repe_eng_config",
        ],
    )

    model_path = str(data["model_path"])  # enforce string
    emotions = list(data["emotions"])  # enforce list copy
    intensities = [float(x) for x in list(data["intensities"])]
    output_dir = str(data["output_dir"])  # enforce string
    loading_config = dict(data["loading_config"])  # enforce dict copy
    repe_eng_config = dict(data["repe_eng_config"])  # enforce dict copy

    # Basic type checks (fail fast)
    if not isinstance(emotions, list) or not all(isinstance(e, str) for e in emotions):
        raise TypeError("'emotions' must be List[str]")
    if not isinstance(intensities, list) or not all(isinstance(x, float) for x in intensities):
        raise TypeError("'intensities' must be List[float]")
    if not isinstance(loading_config, dict) or not isinstance(repe_eng_config, dict):
        raise TypeError("'loading_config' and 'repe_eng_config' must be mappings")

    return DeltaActivationJobConfig(
        model_path=model_path,
        emotions=emotions,
        intensities=intensities,
        output_dir=output_dir,
        loading_config=loading_config,
        repe_eng_config=repe_eng_config,
    )

