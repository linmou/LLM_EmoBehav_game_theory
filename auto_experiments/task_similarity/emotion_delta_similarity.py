"""
Responsible: auto_experiments/task_similarity/emotion_delta_similarity.py
Purpose: Compare PD delta activations with emotion delta activations via PCA-based global direction similarity.

The core workflow:
- Load all PD delta vectors from a directory of runs.
- Load all emotion delta vectors (per emotion) from chat delta runs.
- Compute PCA on pooled PD vectors and on each emotion to obtain their first
  principal components.
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
    projection_similarity: float
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
    pd_vectors: Sequence[np.ndarray],
    emotion_vectors: Mapping[str, Sequence[np.ndarray]],
) -> Dict[str, EmotionPCASummary]:
    """
    Compute PCA-based global direction similarity between PD deltas and emotion deltas.

    Args:
        pd_vectors: sequence of PD delta vectors (1D).
        emotion_vectors: mapping emotion -> sequence of delta vectors (1D).

    Returns:
        Dict mapping emotion -> EmotionPCASummary.
    """
    if not pd_vectors:
        raise ValueError("pd_vectors must be non-empty")
    if not emotion_vectors:
        raise ValueError("emotion_vectors must be non-empty")

    # Validate shapes and shared dimensionality.
    pd_dim = None
    for vec in pd_vectors:
        arr = np.asarray(vec, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"PD vector must be 1D, got shape {arr.shape}")
        if pd_dim is None:
            pd_dim = arr.shape[0]
        elif arr.shape[0] != pd_dim:
            raise ValueError("All PD vectors must have the same dimensionality")
    assert pd_dim is not None

    def _validate_emotion_vectors(vectors: Sequence[np.ndarray], emotion: str) -> np.ndarray:
        rows = []
        for vec in vectors:
            arr = np.asarray(vec, dtype=np.float32)
            if arr.ndim != 1:
                raise ValueError(f"Emotion '{emotion}' vector must be 1D, got shape {arr.shape}")
            if arr.shape[0] != pd_dim:
                raise ValueError(f"Dimension mismatch between PD vectors and emotion '{emotion}' vectors")
            rows.append(arr)
        if not rows:
            raise ValueError(f"No vectors provided for emotion '{emotion}'")
        return np.stack(rows, axis=0)

    pd_mat = np.stack([np.asarray(vec, dtype=np.float32) for vec in pd_vectors], axis=0)
    pd_pc1, pd_var_ratio = compute_pca_first_component(pd_mat)

    result: Dict[str, EmotionPCASummary] = {}
    for emo in sorted(emotion_vectors.keys()):
        emo_mat = _validate_emotion_vectors(emotion_vectors[emo], emo)
        emo_pc1, emo_var_ratio = compute_pca_first_component(emo_mat)
        cos = float(np.dot(pd_pc1, emo_pc1))
        result[emo] = EmotionPCASummary(
            pc1_cosine=cos,
            projection_similarity=abs(cos),
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


def load_pd_vectors(
    pd_root: Path,
    model_prefix: str,
) -> List[np.ndarray]:
    """
    Load all PD delta vectors from a directory of runs.

    Assumes each run dir contains:
      - metadata.json
      - delta.npz containing one or more 1D arrays:
          - legacy: a single measurement_layer key
          - current: one key per layer (e.g. "0","1",...,"N-1")
    """
    vectors: List[np.ndarray] = []
    expected_dim: int | None = None

    for run_dir in _iter_model_run_dirs(pd_root, model_prefix):
        meta_path = run_dir / "metadata.json"
        delta_path = run_dir / "delta.npz"
        if not meta_path.exists() or not delta_path.exists():
            continue
        _ = json.loads(meta_path.read_text(encoding="utf-8"))

        arr = np.load(delta_path)
        keys = list(arr.files)
        if not keys:
            continue

        def _key_sort(k: str) -> tuple[int, str]:
            try:
                return (0, f"{int(k):08d}")
            except Exception:
                return (1, str(k))

        for key in sorted(keys, key=_key_sort):
            vec = arr[key]
            if vec.ndim != 1:
                raise ValueError(f"Expected 1D delta vector at {run_dir} key={key}, got shape {vec.shape}")
            vec = np.asarray(vec, dtype=np.float32)
            if expected_dim is None:
                expected_dim = vec.shape[0]
            elif vec.shape[0] != expected_dim:
                raise ValueError(f"Inconsistent PD vector dimension at {run_dir}")
            vectors.append(vec)

    if not vectors:
        raise ValueError("No PD delta vectors found")

    return vectors


def load_chat_emotion_vectors(
    chat_root: Path,
    model_prefix: str,
    intensity: float,
) -> Tuple[Dict[str, List[np.ndarray]], List[str]]:
    """
    Load emotion delta vectors from chat delta_activations runs.

    Assumes each selected run contains:
      - metadata.json with emotions and intensities fields
      - deltas/emotion=<emo>_int=<intensity>.npz with a 'vector' 1D array
    """
    run_dirs = list(_iter_model_run_dirs(chat_root, model_prefix))
    if not run_dirs:
        raise ValueError("No matching chat runs found")

    emotions: List[str] | None = None
    vectors: Dict[str, List[np.ndarray]] = {}
    expected_dim: int | None = None

    for run_dir in run_dirs:
        meta_path = run_dir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        job_cfg = meta.get("job_config")
        if not isinstance(job_cfg, dict):
            # Older or incompatible formats are ignored.
            continue

        run_emotions = list(meta.get("emotions") or [])
        if not run_emotions:
            raise ValueError(f"No 'emotions' field found in metadata.json at {meta_path}")

        intensities_raw = meta.get("intensities")
        if intensities_raw is None:
            raise ValueError(f"No 'intensities' field found in metadata.json at {meta_path}")
        run_intensities = [float(x) for x in intensities_raw]
        if float(intensity) not in run_intensities:
            # Skip runs that do not contain the requested intensity.
            continue

        if emotions is None:
            emotions = run_emotions
            vectors = {emo: [] for emo in emotions}
        else:
            if set(run_emotions) != set(emotions):
                raise ValueError(f"Inconsistent emotions between runs: {run_emotions} vs {emotions}")

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
            vec = np.asarray(vec, dtype=np.float32)
            if expected_dim is None:
                expected_dim = vec.shape[0]
            elif vec.shape[0] != expected_dim:
                raise ValueError(f"Inconsistent emotion vector dimension at {path}")
            vectors[emo].append(vec)

    if not vectors:
        raise ValueError("No emotion delta vectors found for requested intensity")
    assert emotions is not None

    return vectors, emotions


def compute_pca_similarity_from_roots(
    pd_root: Path,
    chat_root: Path,
    pd_model_prefix: str,
    chat_model_prefix: str,
    intensity: float,
) -> Dict[str, EmotionPCASummary]:
    """
    Convenience wrapper: load PD and emotion vectors from roots and compute PCA similarity.
    """
    pd_vectors = load_pd_vectors(pd_root, pd_model_prefix)
    chat_vectors, _ = load_chat_emotion_vectors(
        chat_root,
        chat_model_prefix,
        intensity=intensity,
    )
    return compute_pca_similarity(pd_vectors, chat_vectors)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PCA-based cosine similarity between PD delta activations and emotion deltas."
    )
    parser.add_argument("--pd_root", required=True, help="Directory containing PD delta runs.")
    parser.add_argument("--chat_root", required=True, help="Directory containing chat delta_activations runs.")
    parser.add_argument("--pd_model_prefix", required=True, help="Prefix of PD run directories to include.")
    parser.add_argument("--chat_model_prefix", required=True, help="Prefix of chat run directories to include.")
    parser.add_argument("--intensity", type=float, default=1.5, help="Emotion intensity to compare (e.g., 1.5).")
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
    )

    # Print a simple ranked table by PC1 cosine.
    items = sorted(result.items(), key=lambda kv: kv[1].pc1_cosine, reverse=True)
    print("emotion,pc1_cosine,projection_similarity,pd_var_explained,emotion_var_explained")
    for emo, summary in items:
        print(
            f"{emo},{summary.pc1_cosine:.6f},{summary.projection_similarity:.6f},"
            f"{summary.pd_variance_explained:.6f},{summary.emotion_variance_explained:.6f}"
        )


if __name__ == "__main__":
    main()
