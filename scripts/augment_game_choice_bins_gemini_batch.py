#!/usr/bin/env python3
"""
scripts/augment_game_choice_bins_gemini_batch.py

Purpose:
- Batch-generate partial augmented datasets (first N records per file) using
  scripts/augment_game_choice_bins_gemini.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_JOBS = [
    ("Escalation_Game", "data_creation/scenario_creation/langgraph_creation/Escalation_Game_all_data_samples.json", 4),
    ("Prisoners_Dilemma", "data_creation/scenario_creation/langgraph_creation/Prisoners_Dilemma_all_data_samples.json", 4),
    ("Stag_Hunt", "data_creation/scenario_creation/langgraph_creation/Stag_Hunt_all_data_samples.json", 4),
    ("Trust_Game_Trustor", "data_creation/scenario_creation/langgraph_creation/Trust_Game_Trustor_all_data_samples.json", 5),
    ("Ultimatum_Game_Proposer", "data_creation/scenario_creation/langgraph_creation/Ultimatum_Game_Proposer_all_data_samples.json", 5),
    ("diplomacy_Trust_Game_Trustee", "data_creation/scenario_creation/langgraph_creation/diplomacy_Trust_Game_Trustee_all_data_samples.json", 5),
    ("diplomacy_Trust_Game_Trustor", "data_creation/scenario_creation/langgraph_creation/diplomacy_Trust_Game_Trustor_all_data_samples.json", 5),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch augment first N records for multiple game datasets.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to write new JSON files")
    parser.add_argument("--per-file", type=int, default=500, help="How many records to augment per input file")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name")
    parser.add_argument("--workers", type=int, default=32, help="Parallel workers inside each file augmentation")
    parser.add_argument("--keep-pivot-in-meta", action="store_true", help="Keep hidden pivot M in metadata for bins=4")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    augment_script = repo_root / "scripts" / "augment_game_choice_bins_gemini.py"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name, rel_in, bins in DEFAULT_JOBS:
        inp = repo_root / rel_in
        if not inp.exists():
            print(f"[{name}] missing input, skipping: {inp}", file=sys.stderr)
            continue

        out = args.out_dir / f"{Path(rel_in).stem}.aug_bins{bins}.{args.model}.first{args.per_file}.json"

        cmd = [
            sys.executable,
            str(augment_script),
            "--input",
            str(inp),
            "--output",
            str(out),
            "--bins",
            str(bins),
            "--model",
            args.model,
            "--limit",
            str(args.per_file),
            "--workers",
            str(args.workers),
        ]
        if args.keep_pivot_in_meta and bins == 4:
            cmd.append("--keep-pivot-in-meta")

        print(f"[{name}] -> {out}", file=sys.stderr)
        subprocess.run(cmd, check=True)

    print(f"done: {args.out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

