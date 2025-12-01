"""
Responsible: auto_experiments/task_similarity/emotion_delta_similarity.py
Purpose: Compare PD delta activations with emotion delta activations via PCA-based global direction similarity.

The core workflow:
- Load per-seed PD delta vectors from a directory of runs.
- Load per-seed emotion delta vectors (per emotion) from chat delta runs.
- For a shared set of seeds, compute PCA on PD and each emotion to obtain
  their first principal components.
- Measure cosine similarity between PD PC1 and each emotion's PC1.

This module is intentionally file-format-focused and stateless. All paths and
model prefixes are provided via configuration/CLI, nothing is hard-coded.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


@dataclass
class EmotionPCASummary:
    pc1_cosine: float
    pd_variance_explained: float
    emotion_variance_explained: float


def compute_pca_first_component(matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute the first principal component and its explained-variance ratio.

    The returned PC1 is oriented so that its dot product with the mean vector
    of the input rows is non-negative and is normalized to unit L2 norm.
    """
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape {matrix.shape}")
    if matrix.shape[0] < 1:
        raise ValueError("matrix must contain at least one sample")

    mat = np.asarray(matrix, dtype=np.float64)
    mean_vec = mat.mean(axis=0, keepdims=True)
    centered = mat - mean_vec

    # SVD-based PCA.
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    total_var = float(np.sum(s ** 2))

    # Degenerate case: all rows identical (no variance across samples).
    # Fall back to the normalized mean direction and report zero variance ratio.
    if total_var <= 0.0 or s.size == 0 or s[0] <= 0.0:
        overall_mean = mat.mean(axis=0)
        norm = float(np.linalg.norm(overall_mean))
        if norm == 0.0:
            raise ValueError("cannot compute PCA: all-zero data matrix")
        pc1 = (overall_mean / norm).astype(np.float32)
        return pc1, 0.0

    pc1 = vt[0]  # first right-singular vector

    # Fix orientation so that PC1 roughly aligns with the (uncentered) mean direction.
    overall_mean = mat.mean(axis=0)
    if np.dot(pc1, overall_mean) < 0.0:
        pc1 = -pc1

    norm = np.linalg.norm(pc1)
    if norm == 0.0:
        raise ValueError("principal component has zero norm")

    pc1 = (pc1 / norm).astype(np.float32)
    var_ratio = float((s[0] ** 2) / total_var)
    return pc1, var_ratio


def compute_pca_similarity(
    pd_vectors: Mapping[int, np.ndarray],
    emotion_vectors: Mapping[int, Mapping[str, np.ndarray]],
    seeds: Sequence[int],
) -> Dict[str, EmotionPCASummary]:
    """
    Compute PCA-based global direction similarity between PD deltas and emotion deltas.

    Args:
        pd_vectors: mapping seed -> PD delta vector (1D).
        emotion_vectors: mapping seed -> {emotion -> delta vector (1D)}.
        seeds: list of seeds to include (must exist in both mappings).

    Returns:
        Dict mapping emotion -> EmotionPCASummary.
    """
    if not seeds:
        raise ValueError("seeds must be non-empty")

    # Ensure seed coverage.
    for seed in seeds:
        if seed not in pd_vectors:
            raise KeyError(f"Missing PD vector for seed {seed}")
        if seed not in emotion_vectors:
            raise KeyError(f"Missing emotion vectors for seed {seed}")

    # Build PD matrix.
    pd_mat = np.stack([np.asarray(pd_vectors[seed], dtype=np.float32) for seed in seeds], axis=0)
    pd_pc1, pd_var_ratio = compute_pca_first_component(pd_mat)

    # Emotions set: require consistency across seeds.
    first_seed = seeds[0]
    first_emotions = set(emotion_vectors[first_seed].keys())
    if not first_emotions:
        raise ValueError("No emotions found for first seed")

    for seed in seeds[1:]:
        emo_keys = set(emotion_vectors[seed].keys())
        if emo_keys != first_emotions:
            raise ValueError(f"Inconsistent emotion set for seed {seed}: {emo_keys} vs {first_emotions}")

    result: Dict[str, EmotionPCASummary] = {}
    for emo in sorted(first_emotions):
        emo_mat = np.stack(
            [np.asarray(emotion_vectors[seed][emo], dtype=np.float32) for seed in seeds],
            axis=0,
        )
        emo_pc1, emo_var_ratio = compute_pca_first_component(emo_mat)
        # Both PCs are unit vectors after compute_pca_first_component.
        cos = float(np.dot(pd_pc1, emo_pc1))
        result[emo] = EmotionPCASummary(
            pc1_cosine=cos,
            pd_variance_explained=pd_var_ratio,
            emotion_variance_explained=emo_var_ratio,
        )

    return result


def _iter_model_run_dirs(root: Path, model_prefix: str) -> Iterable[Path]:
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if not entry.name.startswith(model_prefix):
            continue
        yield entry


def load_pd_seed_vectors(
    pd_root: Path,
    model_prefix: str,
    seed_min: int,
    seed_max: int,
) -> Dict[int, np.ndarray]:
    """
    Load PD delta vectors from a directory of runs.

    Assumes each run dir contains:
      - metadata.json with a "seed" field
      - delta.npz containing exactly one 1D array (e.g., keyed by measurement_layer)
    """
    if seed_min > seed_max:
        raise ValueError(f"seed_min {seed_min} cannot be greater than seed_max {seed_max}")

    seed_to_dir: Dict[int, Path] = {}
    for run_dir in _iter_model_run_dirs(pd_root, model_prefix):
        meta_path = run_dir / "metadata.json"
        delta_path = run_dir / "delta.npz"
        if not meta_path.exists() or not delta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if "seed" not in meta:
            raise KeyError(f"metadata.json at {meta_path} missing 'seed'")
        seed = int(meta["seed"])
        if seed < seed_min or seed > seed_max:
            continue
        existing = seed_to_dir.get(seed)
        # Keep the lexicographically latest run for each seed.
        if existing is None or existing.name < run_dir.name:
            seed_to_dir[seed] = run_dir

    seed_to_vec: Dict[int, np.ndarray] = {}
    for seed, run_dir in seed_to_dir.items():
        arr = np.load(run_dir / "delta.npz")
        keys = list(arr.files)
        if len(keys) != 1:
            raise ValueError(f"Expected single key in delta.npz at {run_dir}, found {keys}")
        vec = arr[keys[0]]
        if vec.ndim != 1:
            raise ValueError(f"Expected 1D delta vector at {run_dir}, got shape {vec.shape}")
        seed_to_vec[seed] = np.asarray(vec, dtype=np.float32)

    return seed_to_vec


def load_chat_seed_emotion_vectors(
    chat_root: Path,
    model_prefix: str,
    intensity: float,
    seed_min: int,
    seed_max: int,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], List[str]]:
    """
    Load emotion delta vectors from chat delta_activations runs.

    For each seed, keeps only the lexicographically latest run directory.
    Assumes each selected run contains:
      - metadata.json with job_config.loading_config.seed, emotions, intensities
      - deltas/emotion=<emo>_int=<intensity>.npz with a 'vector' 1D array
    """
    if seed_min > seed_max:
        raise ValueError(f"seed_min {seed_min} cannot be greater than seed_max {seed_max}")

    seed_to_dir: Dict[int, Path] = {}
    for run_dir in _iter_model_run_dirs(chat_root, model_prefix):
        meta_path = run_dir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        job_cfg = meta.get("job_config")
        if not isinstance(job_cfg, dict):
            raise KeyError(f"metadata.json at {meta_path} missing 'job_config'")
        loading_cfg = job_cfg.get("loading_config")
        if not isinstance(loading_cfg, dict) or "seed" not in loading_cfg:
            raise KeyError(f"metadata.json at {meta_path} missing loading_config.seed")
        seed = int(loading_cfg["seed"])
        if seed < seed_min or seed > seed_max:
            continue
        existing = seed_to_dir.get(seed)
        if existing is None or existing.name < run_dir.name:
            seed_to_dir[seed] = run_dir

    if not seed_to_dir:
        raise ValueError("No matching chat runs found")

    # Use the first selected run to infer emotions and intensities.
    first_dir = next(iter(seed_to_dir.values()))
    first_meta = json.loads((first_dir / "metadata.json").read_text(encoding="utf-8"))
    emotions = list(first_meta.get("emotions") or [])
    if not emotions:
        raise ValueError(f"No 'emotions' field found in metadata.json at {first_dir}")
    intensities_raw = first_meta.get("intensities")
    if intensities_raw is None:
        raise ValueError(f"No 'intensities' field found in metadata.json at {first_dir}")
    intensities = [float(x) for x in intensities_raw]
    if float(intensity) not in intensities:
        raise ValueError(f"Requested intensity {intensity} not in available intensities {intensities}")

    seed_to_vectors: Dict[int, Dict[str, np.ndarray]] = {}
    for seed, run_dir in seed_to_dir.items():
        emo_map: Dict[str, np.ndarray] = {}
        deltas_dir = run_dir / "deltas"
        for emo in emotions:
            fname = f"emotion={emo}_int={float(intensity)}.npz"
            path = deltas_dir / fname
            if not path.exists():
                raise FileNotFoundError(f"Missing delta file for emotion {emo} at {path}")
            arr = np.load(path)
            if "vector" not in arr:
                raise ValueError(f"'vector' key missing in {path}")
            vec = arr["vector"]
            if vec.ndim != 1:
                raise ValueError(f"Expected 1D emotion delta vector at {path}, got shape {vec.shape}")
            emo_map[emo] = np.asarray(vec, dtype=np.float32)
        seed_to_vectors[seed] = emo_map

    return seed_to_vectors, emotions


def compute_pca_similarity_from_roots(
    pd_root: Path,
    chat_root: Path,
    pd_model_prefix: str,
    chat_model_prefix: str,
    intensity: float,
    seed_min: int,
    seed_max: int,
) -> Dict[str, EmotionPCASummary]:
    """
    Convenience wrapper: load PD and emotion vectors from roots and compute PCA similarity.
    """
    pd_vectors = load_pd_seed_vectors(pd_root, pd_model_prefix, seed_min=seed_min, seed_max=seed_max)
    chat_vectors, _ = load_chat_seed_emotion_vectors(
        chat_root,
        chat_model_prefix,
        intensity=intensity,
        seed_min=seed_min,
        seed_max=seed_max,
    )
    seeds = sorted(set(pd_vectors.keys()) & set(chat_vectors.keys()))
    if not seeds:
        raise ValueError("No overlapping seeds between PD and emotion runs")
    return compute_pca_similarity(pd_vectors, chat_vectors, seeds=seeds)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PCA-based cosine similarity between PD delta activations and emotion deltas."
    )
    parser.add_argument("--pd_root", required=True, help="Directory containing PD delta runs.")
    parser.add_argument("--chat_root", required=True, help="Directory containing chat delta_activations runs.")
    parser.add_argument("--pd_model_prefix", required=True, help="Prefix of PD run directories to include.")
    parser.add_argument("--chat_model_prefix", required=True, help="Prefix of chat run directories to include.")
    parser.add_argument("--intensity", type=float, default=1.5, help="Emotion intensity to compare (e.g., 1.5).")
    parser.add_argument(
        "--seed_min",
        type=int,
        default=0,
        help="Minimum seed (inclusive) for runs to include.",
    )
    parser.add_argument(
        "--seed_max",
        type=int,
        default=99,
        help="Maximum seed (inclusive) for runs to include.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    pd_root = Path(args.pd_root)
    chat_root = Path(args.chat_root)

    result = compute_pca_similarity_from_roots(
        pd_root=pd_root,
        chat_root=chat_root,
        pd_model_prefix=args.pd_model_prefix,
        chat_model_prefix=args.chat_model_prefix,
        intensity=args.intensity,
        seed_min=args.seed_min,
        seed_max=args.seed_max,
    )

    # Print a simple ranked table by PC1 cosine.
    items = sorted(result.items(), key=lambda kv: kv[1].pc1_cosine, reverse=True)
    print("emotion,pc1_cosine,pd_var_explained,emotion_var_explained")
    for emo, summary in items:
        print(
            f"{emo},{summary.pc1_cosine:.6f},{summary.pd_variance_explained:.6f},{summary.emotion_variance_explained:.6f}"
        )


if __name__ == "__main__":
    main()
