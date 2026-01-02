"""
Responsible: auto_experiments/task_similarity/pipeline_config_reader.py
Purpose: Minimal parser for EmotionExperiment `experiment_config.json`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class PipelineConfig:
    model_path: str
    emotions: List[str]
    intensities: List[float]


def read_pipeline_config(experiment_config_path: Path) -> PipelineConfig:
    raw = json.loads(experiment_config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("experiment_config.json must be a JSON object")

    model_path = raw.get("model_path")
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("experiment_config.json missing/invalid: model_path")

    emotions = raw.get("emotions")
    if not isinstance(emotions, list) or not emotions:
        raise ValueError("experiment_config.json missing/invalid: emotions")
    emotions_out = []
    for e in emotions:
        if not isinstance(e, str) or not e:
            raise ValueError("experiment_config.json invalid: emotions must be non-empty strings")
        emotions_out.append(e)

    intensities = raw.get("intensities")
    if not isinstance(intensities, list) or not intensities:
        raise ValueError("experiment_config.json missing/invalid: intensities")
    intensities_out = []
    for x in intensities:
        try:
            intensities_out.append(float(x))
        except Exception as exc:
            raise ValueError("experiment_config.json invalid: intensities must be numbers") from exc

    return PipelineConfig(model_path=model_path, emotions=emotions_out, intensities=intensities_out)

