"""
CLI wrapper for PD steering similarity.
"""

import argparse
from pathlib import Path

from . import run_pd_steering_similarity


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PD steering similarity analysis.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--steering_root",
        required=False,
        help="Optional root dir containing layer_vectors (will auto-resolve).",
    )
    args = parser.parse_args()

    steering_root = Path(args.steering_root) if args.steering_root else None
    run_pd_steering_similarity.run_analysis(
        config_path=Path(args.config),
        steering_root=steering_root,
        hidden_state_fn=None,
    )


if __name__ == "__main__":
    main()
