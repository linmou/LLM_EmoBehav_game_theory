from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


@dataclass
class DefectionVector:
    vec: np.ndarray
    layer: str
    source: str


@dataclass
class EmotionVector:
    vec: np.ndarray
    emotion: str
    intensity: float
    source: str
    timestamp: str


def _load_npz_vectors(path: Path) -> Iterable[DefectionVector]:
    with np.load(path) as data:
        for key in data.files:
            vec = data[key]
            if np.linalg.norm(vec) == 0:
                continue
            yield DefectionVector(vec=vec.astype(np.float32), layer=str(key), source=str(path))


def _load_npy_vector(path: Path, layer_hint: str) -> Iterable[DefectionVector]:
    vec = np.load(path)
    if np.linalg.norm(vec) == 0:
        return []
    return [DefectionVector(vec=vec.astype(np.float32), layer=layer_hint, source=str(path))]


def collect_defection_vectors(defection_root: Path) -> Dict[str, List[DefectionVector]]:
    vectors: Dict[str, List[DefectionVector]] = defaultdict(list)
    for scope in ("best_layer", "middle-third-layers"):
        scope_dir = defection_root / scope
        if not scope_dir.exists():
            continue
        for size_dir in scope_dir.iterdir():
            if not size_dir.is_dir():
                continue
            for run_dir in size_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                name_parts = run_dir.name.split("_delta_")
                if len(name_parts) < 2:
                    continue
                model_name = name_parts[0]
                delta_npz = run_dir / "delta.npz"
                if delta_npz.exists():
                    vectors[model_name].extend(_load_npz_vectors(delta_npz))
                    continue
                delta_npy = run_dir / "delta.npy"
                if delta_npy.exists():
                    layer_hint = "?"
                    meta_path = run_dir / "metadata.json"
                    if meta_path.exists():
                        try:
                            meta = json.loads(meta_path.read_text())
                            if "layer" in meta:
                                layer_hint = str(meta["layer"])
                        except Exception:
                            layer_hint = "?"
                    vectors[model_name].extend(_load_npy_vector(delta_npy, layer_hint))
    return vectors


def collect_emotion_vectors(emotion_root: Path) -> Dict[str, List[EmotionVector]]:
    vectors: Dict[str, List[EmotionVector]] = defaultdict(list)
    for run_dir in emotion_root.iterdir():
        if not run_dir.is_dir():
            continue
        parts = run_dir.name.split("_")
        if len(parts) < 3:
            continue
        model_name = "_".join(parts[:-2])
        timestamp = "_".join(parts[-2:])
        delta_dir = run_dir / "deltas"
        if not delta_dir.exists():
            continue
        for npz_path in delta_dir.glob("emotion=*int=*.npz"):
            stem = npz_path.stem
            try:
                emotion_part, intensity_part = stem.split("_")
                emotion = emotion_part.split("=")[1]
                intensity = float(intensity_part.split("=")[1])
            except Exception:
                continue
            with np.load(npz_path) as data:
                vec = data["vector"]
            if np.linalg.norm(vec) == 0:
                continue
            vectors[model_name].append(
                EmotionVector(
                    vec=vec.astype(np.float32),
                    emotion=emotion,
                    intensity=intensity,
                    source=str(npz_path),
                    timestamp=timestamp,
                )
            )
    return vectors


def _principal_component(vectors: List[DefectionVector]) -> np.ndarray:
    matrix = np.stack([v.vec for v in vectors], axis=0)
    # Use SVD without centering to find dominant direction; align to sum for stable sign.
    _, _, vh = np.linalg.svd(matrix, full_matrices=False)
    pc = vh[0]
    summed = matrix.sum(axis=0)
    if np.dot(pc, summed) < 0:
        pc = -pc
    return pc.astype(np.float32)


def compute_best_matrix(
    defection_vectors: Dict[str, List[DefectionVector]], emotion_vectors: Dict[str, List[EmotionVector]]
):
    results: Dict[str, defaultdict] = {}
    for model in sorted(set(defection_vectors.keys()) & set(emotion_vectors.keys())):
        defections = defection_vectors[model]
        emotions = emotion_vectors[model]
        if not emotions or not defections:
            continue
        numeric_layers = [int(d.layer) for d in defections if d.layer.isdigit()]
        if not numeric_layers:
            continue
        max_layer = max(numeric_layers)
        defections = [d for d in defections if d.layer.isdigit() and int(d.layer) == max_layer]
        if not defections:
            continue
        pc_vec = _principal_component(defections)
        matrix: defaultdict = defaultdict(dict)
        for emo_entry in emotions:
            if emo_entry.vec.shape != pc_vec.shape:
                continue
            denom = float(np.linalg.norm(emo_entry.vec) * np.linalg.norm(pc_vec))
            if denom == 0.0:
                continue
            cos = float(np.dot(emo_entry.vec, pc_vec) / denom)
            matrix[emo_entry.intensity][emo_entry.emotion] = {
                "cos": cos,
                "def_layer": str(max_layer),
                "def_source": defections[0].source,
                "emo_source": emo_entry.source,
                "timestamp": emo_entry.timestamp,
            }
        if matrix:
            results[model] = matrix
    return results


def _format_intensity(val: float) -> str:
    as_str = f"{val:.3f}"
    return as_str.rstrip("0").rstrip(".")


def format_markdown_tables(matrix):
    lines: List[str] = []
    for model, rows in matrix.items():
        lines.append(f"## {model}")
        all_emotions = sorted({emo for row in rows.values() for emo in row.keys()})
        header = "| intensity\\emotion | " + " | ".join(all_emotions) + " |"
        separator = "| --- | " + " | ".join("---" for _ in all_emotions) + " |"
        lines.append(header)
        lines.append(separator)
        for intensity in sorted(rows.keys()):
            cells = []
            for emo in all_emotions:
                cell = rows[intensity].get(emo)
                if cell is None:
                    cells.append("-")
                else:
                    cells.append(f"{cell['cos']:.4f} (L{cell['def_layer']})")
            lines.append("| " + _format_intensity(intensity) + " | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare emotion deltas to defection deltas and emit markdown tables.")
    parser.add_argument(
        "--emotion-root",
        type=Path,
        default=Path("results/delta_activations/chat"),
        help="Root directory of emotion delta runs.",
    )
    parser.add_argument(
        "--defection-root",
        type=Path,
        default=Path("auto_experiments/task-similarity/results/delta"),
        help="Root directory of defection delta runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("result_analysis/defection_emotion_similarity.md"),
        help="Where to write the markdown comparison.",
    )
    args = parser.parse_args(argv)

    defection_vectors = collect_defection_vectors(args.defection_root)
    emotion_vectors = collect_emotion_vectors(args.emotion_root)
    matrix = compute_best_matrix(defection_vectors, emotion_vectors)
    markdown = format_markdown_tables(matrix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown)
    print(f"Wrote comparison for {len(matrix)} models to {args.output}")


if __name__ == "__main__":
    main()
