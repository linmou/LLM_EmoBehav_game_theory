#!/usr/bin/env python3
"""
scripts/augment_game_choice_bins_gemini_smoke.py

Purpose:
- Smoke-test the Gemini choice-bin augmentation by generating a small subset
  (default: 3 records) for each configured game JSON file.

Writes NEW JSON files into an output directory.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_INPUTS = [
    ("Prisoners_Dilemma", "data_creation/scenario_creation/langgraph_creation/Prisoners_Dilemma_all_data_samples.json", 4),
    ("Trust_Game_Trustor", "data_creation/scenario_creation/langgraph_creation/Trust_Game_Trustor_all_data_samples.json", 5),
    ("Ultimatum_Game_Proposer", "data_creation/scenario_creation/langgraph_creation/Ultimatum_Game_Proposer_all_data_samples.json", 5),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a small augmented subset for each game (Gemini smoke test).")
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/game_aug_smoke"), help="Output directory for new JSONs")
    parser.add_argument("--per-game", type=int, default=3, help="How many records to augment per game")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers inside each augmentation run")
    parser.add_argument("--keep-pivot-in-meta", action="store_true", help="Keep hidden pivot M in metadata (2-anchor bins=4)")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "augment_game_choice_bins_gemini.py"

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for game_name, rel_in, bins in DEFAULT_INPUTS:
        inp = repo_root / rel_in
        if not inp.exists():
            raise SystemExit(f"Missing input: {inp}")

        out = args.out_dir / f"{game_name}_all_data_samples.aug_bins{bins}.{args.model}.smoke{args.per_game}.json"

        cmd = [
            sys.executable,
            str(script),
            "--input",
            str(inp),
            "--output",
            str(out),
            "--bins",
            str(bins),
            "--model",
            args.model,
            "--limit",
            str(args.per_game),
            "--workers",
            str(args.workers),
        ]
        if args.keep_pivot_in_meta and bins == 4:
            cmd.append("--keep-pivot-in-meta")

        print(f"[{game_name}] {inp.name} -> {out}", file=sys.stderr)
        subprocess.run(cmd, check=True)

    print(f"done: {args.out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

