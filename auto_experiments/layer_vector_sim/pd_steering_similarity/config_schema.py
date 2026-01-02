"""
Config schema and validation for PD steering similarity analysis.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass
class ModelConfig:
    name: str
    path: Path


@dataclass
class BenchmarkConfig:
    name: str
    task: str
    raw_results_path: Path


@dataclass
class SteeringConfig:
    emotions: List[str]
    intensities: List[float]
    loader: str


@dataclass
class PDDefectionVectorsConfig:
    dir: Path


@dataclass
class OutputConfig:
    dir: Path


@dataclass
class PDSteeringConfig:
    model: ModelConfig
    benchmark: BenchmarkConfig
    steering: SteeringConfig
    pd_defection_vectors: PDDefectionVectorsConfig
    output: OutputConfig


def _require_section(data: dict, key: str) -> dict:
    try:
        section = data[key]
    except KeyError as exc:
        raise ValueError(f"Missing required section '{key}'") from exc
    if not isinstance(section, dict):
        raise ValueError(f"Section '{key}' must be a mapping")
    return section


def load_config(path: Path) -> PDSteeringConfig:
    """
    Load and validate PD steering similarity config from YAML.
    """
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")

    model_section = _require_section(raw, "model")
    benchmark_section = _require_section(raw, "benchmark")
    steering_section = _require_section(raw, "steering")
    pd_vectors_section = _require_section(raw, "pd_defection_vectors")
    output_section = _require_section(raw, "output")

    model = ModelConfig(
        name=_as_str(model_section, "name"),
        path=_as_path(model_section, "path"),
    )
    benchmark = BenchmarkConfig(
        name=_as_str(benchmark_section, "name"),
        task=_as_str(benchmark_section, "task"),
        raw_results_path=_as_path(benchmark_section, "raw_results_path"),
    )
    steering = SteeringConfig(
        emotions=_as_str_list(steering_section, "emotions"),
        intensities=_as_float_list(steering_section, "intensities"),
        loader=_as_str(steering_section, "loader"),
    )
    pd_vectors = PDDefectionVectorsConfig(
        dir=_as_path(pd_vectors_section, "dir"),
    )
    output = OutputConfig(
        dir=_as_path(output_section, "dir"),
    )

    return PDSteeringConfig(
        model=model,
        benchmark=benchmark,
        steering=steering,
        pd_defection_vectors=pd_vectors,
        output=output,
    )


def _as_str(section: dict, key: str) -> str:
    try:
        value = section[key]
    except KeyError as exc:
        raise ValueError(f"Missing required key '{key}' in section") from exc
    if not isinstance(value, str):
        raise ValueError(f"'{key}' must be a string")
    return value


def _as_path(section: dict, key: str) -> Path:
    return Path(_as_str(section, key))


def _as_str_list(section: dict, key: str) -> List[str]:
    try:
        values = section[key]
    except KeyError as exc:
        raise ValueError(f"Missing required key '{key}' in section") from exc
    if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
        raise ValueError(f"'{key}' must be a list of strings")
    return values


def _as_float_list(section: dict, key: str) -> List[float]:
    try:
        values = section[key]
    except KeyError as exc:
        raise ValueError(f"Missing required key '{key}' in section") from exc
    if not isinstance(values, list):
        raise ValueError(f"'{key}' must be a list of numbers")
    floats: List[float] = []
    for item in values:
        if not isinstance(item, (int, float)):
            raise ValueError(f"'{key}' must be a list of numbers")
        floats.append(float(item))
    return floats
