"""
Simple CLI entrypoint for delta activation computation.

Usage:
  python -m delta_activation_engine.cli --config path/to/config.yaml
"""

from __future__ import annotations

import argparse

from .config import load_job_config_from_yaml
from .backends import HFBackend
from .pipelines.runner import run_job


def main():
    parser = argparse.ArgumentParser(description="Compute delta activations (HF backend)")
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()

    cfg = load_job_config_from_yaml(args.config)
    backend = HFBackend(cfg)
    out_dir = run_job(cfg, backend)
    print(out_dir)


if __name__ == "__main__":
    main()
