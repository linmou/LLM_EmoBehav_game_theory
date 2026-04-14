#!/usr/bin/env python3
# Purpose: reproduce the Qwen2.5 positive-margin table with one command, either from existing results or by rerunning the sweep pipeline first.

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STIMULUS_DATA_DIR = "data/stimulus/crowd-enVent_textlike"
DEFAULT_RESULTS_ROOT = (
    PROJECT_ROOT / "results" / "auto_experiments" / "pd_selfreport_pd_coupling_multimodel"
)
QWEN25_MODEL_PATHS = {
    "qwen2p5-0p5b-instruct": "/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2p5-1p5b-instruct": "/home/jjl7137/huggingface_models/Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2p5-3b-instruct": "/home/jjl7137/huggingface_models/Qwen/Qwen2.5-3B-Instruct",
}
FULL_QWEN25_INTENSITIES = "1,2,4,6,8,10,15,20,40,80"


class Step(NamedTuple):
    name: str
    argv: list[str]


def _python_script(script_relpath: str) -> list[str]:
    return [sys.executable, str(PROJECT_ROOT / script_relpath)]


def _model_output_root(results_root: Path, model_slug: str) -> Path:
    if model_slug == "qwen2p5-0p5b-instruct":
        return results_root / "self_report_logprob"
    return results_root / "self_report_logprob_multimodel" / model_slug


def build_steps(
    mode: str,
    *,
    stimulus_data_dir: str = DEFAULT_STIMULUS_DATA_DIR,
    results_root: Path = DEFAULT_RESULTS_ROOT,
) -> list[Step]:
    if mode == "table-only":
        return [
            Step(
                name="positive_margin_table",
                argv=[
                    *_python_script(
                        "scripts/qwen25_positive_margin_table/build_positive_margin_table.py"
                    ),
                    "--results-root",
                    str(results_root),
                ],
            )
        ]
    if mode == "rerun":
        steps: list[Step] = []
        for model_slug, model_path in QWEN25_MODEL_PATHS.items():
            steps.append(
                Step(
                    name=f"selfreport_sweep_{model_slug}",
                    argv=[
                        *_python_script(
                            "scripts/qwen25_positive_margin_table/run_selfreport_qwen25_sweep.py"
                        ),
                        "--model-path",
                        model_path,
                        "--skip-existing",
                        "--stimulus-data-dir",
                        str(stimulus_data_dir),
                        "--intensities",
                        FULL_QWEN25_INTENSITIES,
                        "--output-root",
                        str(_model_output_root(results_root, model_slug)),
                    ],
                )
            )
        steps.append(
            Step(
                name="positive_margin_table",
                argv=[
                    *_python_script(
                        "scripts/qwen25_positive_margin_table/build_positive_margin_table.py"
                    ),
                    "--results-root",
                    str(results_root),
                ],
            )
        )
        return steps
    raise ValueError(f"Unsupported mode: {mode}")


def run_step(step: Step) -> None:
    print(f"[run] {step.name}: {' '.join(step.argv)}", flush=True)
    subprocess.run(step.argv, cwd=PROJECT_ROOT, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["table-only", "rerun"],
        default="rerun",
        help="Rerun the full sweep pipeline by default; use table-only only when existing results are already available.",
    )
    parser.add_argument("--stimulus-data-dir", type=str, default=DEFAULT_STIMULUS_DATA_DIR)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    for step in build_steps(
        mode=str(args.mode),
        stimulus_data_dir=str(args.stimulus_data_dir),
        results_root=Path(args.results_root),
    ):
        run_step(step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
