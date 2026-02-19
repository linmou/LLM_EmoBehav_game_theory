#!/usr/bin/env python3
"""
Run system-prompt variant search for PsySET emotion steering.

This script is resume-safe:
- it skips variants that already have completed results;
- it appends metrics into one compact CSV summary.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml


DEFAULT_VARIANTS: List[Dict[str, str]] = [
    {"variant_id": "baseline", "system_prompt": ""},
    {
        "variant_id": "v1_concrete_state",
        "system_prompt": (
            "You are a participant writing one short first-person statement. "
            "Describe immediate internal state using concrete details and natural phrasing. "
            "No meta commentary."
        ),
    },
    {
        "variant_id": "v2_diary_micro",
        "system_prompt": (
            "You are writing a diary micro-entry in first person. "
            "Use one vivid moment with body cues, attention focus, and action tendency. "
            "Keep it under 30 words."
        ),
    },
    {
        "variant_id": "v3_appraisal_action",
        "system_prompt": (
            "You are responding in first person from inside the moment. "
            "State what stands out, how the situation is interpreted, and what you want to do next. "
            "Keep it concise and specific."
        ),
    },
    {
        "variant_id": "v4_instinctive_reaction",
        "system_prompt": (
            "You are giving an instinctive first-person reaction. "
            "Use concrete language, avoid generic statements, and avoid explanations. "
            "Write one short sentence."
        ),
    },
]


def _validate_no_explicit_emotion_wording(prompt_text: str) -> None:
    if not prompt_text:
        return
    banned_words = [
        "anger",
        "happiness",
        "sadness",
        "fear",
        "disgust",
        "surprise",
        "neutral",
        "emotion",
        "emotional",
        "feel",
        "feeling",
    ]
    low = prompt_text.lower()
    for word in banned_words:
        if re.search(rf"\b{re.escape(word)}\b", low):
            raise ValueError(
                f"Prompt contains banned explicit emotion wording '{word}': {prompt_text}"
            )


def _load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_config(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _find_latest_result_dir(variant_output_root: Path) -> Path:
    candidates = [p for p in variant_output_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run folder found under {variant_output_root}")
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def _compute_metrics(detailed_results_csv: Path) -> Dict[str, float]:
    df = pd.read_csv(detailed_results_csv)
    steered = df[df["emotion"] != "neutral"].copy()
    steered["is_match"] = steered["predicted_emotion"] == steered["ground_truth"]
    plain_accuracy = float(steered["is_match"].mean())
    match_score = float(pd.to_numeric(steered["score"], errors="coerce").fillna(0.0).mean())
    neutral = df[df["emotion"] == "neutral"].copy()
    neutral_acc = float((neutral["predicted_emotion"] == neutral["ground_truth"]).mean())
    return {
        "plain_accuracy": plain_accuracy,
        "match_score": match_score,
        "neutral_accuracy": neutral_acc,
        "num_rows": int(len(df)),
        "num_steered_rows": int(len(steered)),
    }


def _run_variant(
    variant: Dict[str, str],
    base_config: Dict,
    configs_dir: Path,
    results_root: Path,
    conda_bin: str,
    conda_env: str,
    cuda_devices: str,
    force: bool,
) -> Dict[str, str]:
    variant_id = variant["variant_id"]
    prompt_text = variant["system_prompt"].strip()
    _validate_no_explicit_emotion_wording(prompt_text)

    variant_output_root = results_root / variant_id
    variant_output_root.mkdir(parents=True, exist_ok=True)
    existing_dirs = [p for p in variant_output_root.iterdir() if p.is_dir()]
    if existing_dirs and not force:
        latest_dir = _find_latest_result_dir(variant_output_root)
        detailed = latest_dir / "detailed_results.csv"
        if detailed.exists():
            metrics = _compute_metrics(detailed)
            return {
                "variant_id": variant_id,
                "status": "skipped_existing",
                "system_prompt": prompt_text,
                "output_dir": str(latest_dir),
                **{k: str(v) for k, v in metrics.items()},
                "duration_seconds": "0.0",
            }

    cfg = dict(base_config)
    cfg["experiment_name"] = f"{base_config.get('experiment_name', 'prompt_search')}_{variant_id}"
    cfg["output_dir"] = str(variant_output_root)

    config_path = configs_dir / f"{variant_id}.yaml"
    _save_config(cfg, config_path)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    if prompt_text:
        env["EMOTION_CHECK_SYSTEM_PROMPT_OVERRIDE"] = prompt_text
    else:
        env.pop("EMOTION_CHECK_SYSTEM_PROMPT_OVERRIDE", None)

    cmd = [
        conda_bin,
        "run",
        "-n",
        conda_env,
        "python",
        "-m",
        "emotion_experiment_engine.emotion_experiment_series_runner",
        "--config",
        str(config_path),
    ]

    start = time.time()
    subprocess.run(cmd, check=True, env=env)
    duration = time.time() - start

    latest_dir = _find_latest_result_dir(variant_output_root)
    metrics = _compute_metrics(latest_dir / "detailed_results.csv")
    return {
        "variant_id": variant_id,
        "status": "completed",
        "system_prompt": prompt_text,
        "output_dir": str(latest_dir),
        **{k: str(v) for k, v in metrics.items()},
        "duration_seconds": f"{duration:.2f}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path(
            "auto_experiments/emotion_check_system_prompt_matchness/configs/quick_intensity4.yaml"
        ),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(
            "results/auto_experiments/emotion_check_system_prompt_matchness/quick_prompt_search"
        ),
    )
    parser.add_argument(
        "--generated-config-dir",
        type=Path,
        default=Path(
            "auto_experiments/emotion_check_system_prompt_matchness/generated_configs/quick_prompt_search"
        ),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path(
            "auto_experiments/emotion_check_system_prompt_matchness/quick_prompt_search_summary.csv"
        ),
    )
    parser.add_argument("--conda-bin", default="/home/jjl7137/anaconda3/bin/conda")
    parser.add_argument("--conda-env", default="llm")
    parser.add_argument("--cuda-devices", default="2,3")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--variants-json", type=Path, default=None)
    args = parser.parse_args()

    base_config = _load_config(args.base_config)
    variants = DEFAULT_VARIANTS
    if args.variants_json:
        with args.variants_json.open("r", encoding="utf-8") as f:
            variants = json.load(f)

    rows = []
    for variant in variants:
        row = _run_variant(
            variant=variant,
            base_config=base_config,
            configs_dir=args.generated_config_dir,
            results_root=args.results_root,
            conda_bin=args.conda_bin,
            conda_env=args.conda_env,
            cuda_devices=args.cuda_devices,
            force=args.force,
        )
        print(
            f"[{row['status']}] {row['variant_id']} "
            f"acc={row['plain_accuracy']} match={row['match_score']} dir={row['output_dir']}"
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.summary_csv, index=False)
    print(f"Saved summary: {args.summary_csv}")


if __name__ == "__main__":
    main()

