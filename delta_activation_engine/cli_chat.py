"""
Responsible: delta_activation_engine/cli_chat.py
Purpose: Simple CLI for the chat-template-aware delta activation pipeline.

Usage:
  python -m delta_activation_engine.cli_chat --config path/to/chat_job.yaml
"""

from __future__ import annotations

import argparse

from .config import load_chat_job_config_from_yaml
from .pipelines.chat_runner import run_job_chat


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute delta activations (chat-template-aware)")
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()

    cfg = load_chat_job_config_from_yaml(args.config)
    out_dir = run_job_chat(cfg)
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
