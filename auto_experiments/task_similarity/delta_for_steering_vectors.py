"""
Responsible: auto_experiments/task_similarity/delta_for_steering_vectors.py
Purpose: Batch wrapper to run compute_pd_delta.run_delta over steering vector runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm.auto import tqdm

from . import compute_pd_delta


def _resolve_layer(explicit_layer: Optional[int], steering_root: Path) -> int:
    if explicit_layer is not None:
        return int(explicit_layer)
    metrics_path = steering_root.parent / "layer_metrics.json"
    if not metrics_path.exists():
        raise ValueError(
            "layer is None and no layer_metrics.json found next to steering_root; "
            f"expected at {metrics_path}"
        )
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    if "best_layer" not in data:
        raise ValueError(f"layer_metrics.json missing 'best_layer': {metrics_path}")
    return int(data["best_layer"])


def run_for_steering_vectors(
    model_path: str,
    steering_root: Path,
    delta_root: Path,
    layer: Optional[int],
    intensity: float,
    max_length: int,
    batch_size: int,
    use_middle_third: bool = False,
) -> List[Dict[str, Any]]:
    steering_root = Path(steering_root)
    delta_root = Path(delta_root)
    delta_root.mkdir(parents=True, exist_ok=True)

    if use_middle_third:
        effective_layer: Optional[int] = None
    else:
        effective_layer = _resolve_layer(layer, steering_root)

    results: List[Dict[str, Any]] = []
    seed_counter = 0

    # Flatten all seed directories first so we can show a single progress bar.
    seed_dirs = []
    for ts_dir in sorted(p for p in steering_root.iterdir() if p.is_dir()):
        for seed_dir in sorted(p for p in ts_dir.iterdir() if p.is_dir()):
            seed_dirs.append(seed_dir)

    for seed_dir in tqdm(seed_dirs, desc="delta runs", total=len(seed_dirs)):
        # Steering vectors are stored per seed under a layer_vectors subdirectory.
        vec_dir = seed_dir / "layer_vectors"
        vector_path = vec_dir if vec_dir.is_dir() else seed_dir

        ts_name = seed_dir.parent.name
        out_dir = delta_root / ts_name / seed_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        result = compute_pd_delta.run_delta(
            model_path=model_path,
            vector_path=vector_path,
            layer=effective_layer,
            use_middle_third=use_middle_third,
            intensity=intensity,
            output_dir=out_dir,
            max_length=max_length,
            batch_size=batch_size,
            seed=seed_counter,
        )
        results.append(result)
        seed_counter += 1

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute delta activations for all steering vector runs under a root."
    )
    parser.add_argument(
        "--model",
        default="/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct",
        help="HF model path. Default is Qwen2.5-0.5B-Instruct.",
    )
    parser.add_argument(
        "--steering_root",
        default="auto_experiments/task_similarity/results/steering_vectors/Qwen2.5-0.5B-Instruct",
        help="Root directory containing steering vectors (timestamp/seed_*/layer_vectors).",
    )
    parser.add_argument(
        "--delta_root",
        default="auto_experiments/task_similarity/results/delta/Qwen2.5-0.5B-steering_vectors_midthird",
        help="Output root for delta activation runs.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Single layer to steer. If omitted and --middle_third is not set, "
        "best_layer from layer_metrics.json is used.",
    )
    parser.add_argument("--intensity", type=float, default=1.5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    results = run_for_steering_vectors(
        model_path=args.model,
        steering_root=Path(args.steering_root),
        delta_root=Path(args.delta_root),
        layer=args.layer,
        intensity=args.intensity,
        max_length=args.max_length,
        batch_size=args.batch_size,
        use_middle_third=True,
    )

    summary = {"runs": len(results)}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
