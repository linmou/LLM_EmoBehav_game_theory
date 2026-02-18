"""
Run chat-template-aware delta activation sweeps across Qwen2.5 Instruct models.

Usage:
  python scripts/run_delta_chat_seed_sweep.py --start 50 --end 149 --cuda 0

Defaults:
  start=50, end=149, cuda=0
"""

import argparse
import os
from pathlib import Path

from delta_activation_engine.config import load_chat_job_config_from_yaml
from delta_activation_engine.pipelines.chat_seed_plan import build_seeded_chat_jobs
from delta_activation_engine.pipelines.chat_runner import run_job_chat


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chat delta-activation sweeps with seeded configs.")
    parser.add_argument("--start", type=int, default=50, help="Start seed (inclusive)")
    parser.add_argument("--end", type=int, default=149, help="End seed (inclusive)")
    parser.add_argument("--cuda", type=str, default="0", help="CUDA_VISIBLE_DEVICES value")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    seeds = list(range(args.start, args.end + 1))
    base = load_chat_job_config_from_yaml("config/delta_activations/qwen_chat_template.yaml")
    models = [
        os.path.expandvars("${USER_HOME}/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct").replace("${USER_HOME}", "/home/jjl7137"),
        os.path.expandvars("${USER_HOME}/huggingface_models/Qwen/Qwen2.5-1.5B-Instruct").replace("${USER_HOME}", "/home/jjl7137"),
        os.path.expandvars("${USER_HOME}/huggingface_models/Qwen/Qwen2.5-3B-Instruct").replace("${USER_HOME}", "/home/jjl7137"),
    ]

    jobs = build_seeded_chat_jobs(base, models, seeds, output_root="results/delta_activations")
    print(f"Total jobs: {len(jobs)} | seeds {args.start}-{args.end} | CUDA {args.cuda}")

    for idx, job in enumerate(jobs, 1):
        seed = job.loading_config.get("seed")
        print(f"[{idx}/{len(jobs)}] Running {job.model_path} seed={seed}...")
        out = run_job_chat(job)
        print(f"[{idx}/{len(jobs)}] Saved -> {out}")


if __name__ == "__main__":
    main()
