"""
Responsible: auto_experiments/task-similarity/repeat_pd_delta_runs.py
Purpose: Run compute_pd_delta multiple times with sequential seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import compute_pd_delta


def _log(msg: str) -> None:
    print(msg, flush=True)


def run_batch(
    model_path: str,
    vector_path: Path,
    output_dir: Path,
    layer: Optional[int],
    middle_third: bool,
    intensity: float,
    max_length: int,
    batch_size: int,
    start_seed: int,
    num_runs: int,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for idx in range(num_runs):
        seed = start_seed + idx
        _log(f"[{idx + 1}/{num_runs}] seed={seed} start")
        result = compute_pd_delta.run_delta(
            model_path=model_path,
            vector_path=Path(vector_path),
            layer=layer,
            use_middle_third=middle_third,
            intensity=intensity,
            output_dir=Path(output_dir),
            max_length=max_length,
            batch_size=batch_size,
            seed=seed,
        )
        results.append(result)
        _log(f"[{idx + 1}/{num_runs}] seed={seed} done")
    _log(f"completed {num_runs} runs starting at seed {start_seed}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--vector_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layer", type=int, default=None, help="Target layer. If omitted with --middle_third, use middle third.")
    parser.add_argument("--middle_third", action="store_true", help="Apply vector to middle third of layers.")
    parser.add_argument("--intensity", type=float, default=1.5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--num_runs", type=int, default=100)
    args = parser.parse_args()

    if not args.middle_third and args.layer is None:
        raise ValueError("Specify --layer or use --middle_third")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_batch(
        model_path=args.model,
        vector_path=Path(args.vector_path),
        output_dir=output_dir,
        layer=args.layer,
        middle_third=args.middle_third,
        intensity=args.intensity,
        max_length=args.max_length,
        batch_size=args.batch_size,
        start_seed=args.start_seed,
        num_runs=args.num_runs,
    )
    summary = {
        "runs": len(results),
        "seeds": [r.get("seed") for r in results],
        "output_dir": str(output_dir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
