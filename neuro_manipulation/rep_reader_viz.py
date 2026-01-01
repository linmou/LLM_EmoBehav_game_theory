from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.decomposition import PCA


def collect_direction_points(
    emotion_rep_readers: Mapping[str, Any], *, model_id: str
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    vectors: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []

    for emotion, rep_reader in emotion_rep_readers.items():
        if emotion in {"layer_acc", "args"}:
            continue
        directions = getattr(rep_reader, "directions", None)
        if not isinstance(directions, dict):
            raise TypeError(f"emotion {emotion!r} rep_reader has no directions dict")

        for layer, layer_dirs in directions.items():
            layer_dirs = np.asarray(layer_dirs, dtype=np.float32)
            if layer_dirs.ndim != 2:
                raise ValueError(
                    f"emotion {emotion!r} layer {layer} directions must be 2D, got {layer_dirs.shape}"
                )
            for component in range(layer_dirs.shape[0]):
                vectors.append(layer_dirs[component].reshape(-1))
                meta.append(
                    {
                        "model_id": model_id,
                        "emotion": emotion,
                        "layer": int(layer),
                        "component": int(component),
                    }
                )

    if not vectors:
        raise ValueError("No direction vectors found (empty emotion_rep_readers?)")

    return np.vstack(vectors), meta


def reduce_vectors_to_2d(vectors: np.ndarray, *, method: str = "pca") -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2D, got {vectors.shape}")
    if vectors.shape[0] < 2:
        raise ValueError("Need at least 2 vectors to reduce to 2D")

    if method != "pca":
        raise ValueError(f"Unsupported reduction method: {method}")

    return PCA(n_components=2).fit_transform(vectors).astype(np.float32, copy=False)


def emotion_reader_cache_path(
    config: Mapping[str, Any], *, hidden_layers: list[int]
) -> Path:
    from neuro_manipulation.utils import dict_to_unique_code, validate_multimodal_experiment_feasibility

    feasibility = validate_multimodal_experiment_feasibility(config)
    if not feasibility["feasible"]:
        raise ValueError(f"Config not feasible for emotion readers: {feasibility['reasons']}")

    experiment_mode = feasibility["mode"]
    multimodal_intent = bool(config.get("multimodal_intent", False))
    emotion_data_seed = int(config.get("emotion_data_seed", 0))

    args = {
        "emotions": config["emotions"],
        "data_dir": config["data_dir"],
        "model_name_or_path": config["model_name_or_path"],
        "rep_token": config["rep_token"],
        "hidden_layers": hidden_layers,
        "n_difference": config["n_difference"],
        "direction_method": config["direction_method"],
        "experiment_mode": experiment_mode,
        "multimodal_intent": multimodal_intent,
        "emotion_data_seed": emotion_data_seed,
    }

    arg_codes = dict_to_unique_code(args)
    return Path(
        f"neuro_manipulation/representation_storage/emotion_rep_reader_{arg_codes[:10]}.pkl"
    )


def infer_repe_config_for_model(model_path: str, series_config: Mapping[str, Any]) -> dict[str, Any]:
    from neuro_manipulation.configs.experiment_config import get_repe_eng_config

    repe_overrides = series_config.get("repe_eng_config")
    if repe_overrides is not None and not isinstance(repe_overrides, dict):
        raise TypeError("series_config['repe_eng_config'] must be a dict when provided")
    return get_repe_eng_config(model_path, yaml_config=repe_overrides)
