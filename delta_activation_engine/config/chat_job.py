"""
Responsible: delta_activation_engine/config/chat_job.py
Purpose: Strict YAML parsing for chat-template-aware delta activation jobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TextIO

import yaml


@dataclass
class PromptingConfig:
    benchmark_name: str
    task_type: str
    probes: Optional[List[str]]
    probe_source: Optional[str]
    enable_thinking: Optional[bool]


@dataclass
class DeltaActivationChatJobConfig:
    model_path: str
    emotions: List[str]
    intensities: List[float]
    output_dir: str
    loading_config: Dict[str, Any]
    repe_eng_config: Dict[str, Any]
    prompt_config: PromptingConfig


def _ensure_required(data: Dict[str, Any], keys: List[str]) -> None:
    missing = [k for k in keys if k not in data or data[k] is None]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")


def load_chat_job_config_from_yaml(stream_or_path: TextIO | str) -> DeltaActivationChatJobConfig:
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
            "prompt_config",
        ],
    )

    model_path = str(data["model_path"])  # enforce string
    emotions = list(data["emotions"])  # enforce list copy
    intensities = [float(x) for x in list(data["intensities"])]
    output_dir = str(data["output_dir"])  # enforce string
    loading_config = dict(data["loading_config"])  # enforce dict copy
    repe_eng_config = dict(data["repe_eng_config"])  # enforce dict copy

    pc_raw = data["prompt_config"]
    if not isinstance(pc_raw, dict):
        raise TypeError("prompt_config must be a mapping")
    _ensure_required(pc_raw, ["benchmark_name", "task_type"])

    probes = list(pc_raw["probes"]) if pc_raw.get("probes") is not None else None
    probe_source = str(pc_raw.get("probe_source")) if pc_raw.get("probe_source") is not None else None
    enable_thinking = bool(pc_raw.get("enable_thinking")) if pc_raw.get("enable_thinking") is not None else None

    pc = PromptingConfig(
        benchmark_name=str(pc_raw["benchmark_name"]),
        task_type=str(pc_raw["task_type"]),
        probes=probes,
        probe_source=probe_source,
        enable_thinking=enable_thinking,
    )

    # Basic type checks (fail fast)
    if not isinstance(emotions, list) or not all(isinstance(e, str) for e in emotions):
        raise TypeError("'emotions' must be List[str]")
    if not isinstance(intensities, list) or not all(isinstance(x, float) for x in intensities):
        raise TypeError("'intensities' must be List[float]")
    if not isinstance(loading_config, dict) or not isinstance(repe_eng_config, dict):
        raise TypeError("'loading_config' and 'repe_eng_config' must be mappings")

    return DeltaActivationChatJobConfig(
        model_path=model_path,
        emotions=emotions,
        intensities=intensities,
        output_dir=output_dir,
        loading_config=loading_config,
        repe_eng_config=repe_eng_config,
        prompt_config=pc,
    )

