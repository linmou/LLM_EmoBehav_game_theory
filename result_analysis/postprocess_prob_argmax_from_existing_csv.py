"""
Postprocess an existing prob-argmax scoring CSV to add behavior labels and summary distributions.

Input (in --run_dir):
- raw_results.json (to map (emotion,intensity,item_id,option_id) -> behavior)
- option_prob_argmax_matches_score.csv or prob_argmax_matches_score.csv (existing scored output)

Outputs (written into --run_dir):
- prob_argmax_matches_score.csv (renamed + enriched with chosen_behavior/predicted_behavior)
- behavior_prob_argmax_matches_score.csv
- summary_predicted_option_argmax_ratio.csv
- summary_predicted_behavior_argmax_ratio.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from result_analysis.score_game_theory_option_prob_match import (
    behavior_match_rows,
    predicted_behavior_argmax_ratios,
    predicted_option_argmax_ratios,
)


def _read_csv_rows(path: Path) -> List[Dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _load_raw_records(raw_results_path: Path) -> List[Mapping[str, object]]:
    raw = json.loads(raw_results_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError("raw_results.json must be a list of records")
    out: List[Mapping[str, object]] = []
    for r in raw:
        if isinstance(r, dict):
            out.append(r)
    return out


def _index_behaviors(
    behavior_rows: Sequence[Mapping[str, object]],
) -> Dict[Tuple[str, float, int], Tuple[str, str]]:
    out: Dict[Tuple[str, float, int], Tuple[str, str]] = {}
    for r in behavior_rows:
        key = (str(r["emotion"]), float(r["intensity"]), int(float(r["item_id"])))
        out[key] = (str(r["chosen_behavior"]), str(r["predicted_behavior"]))
    return out


def postprocess_run_dir(*, run_dir: Path, scored_csv: Optional[Path] = None) -> Path:
    raw_results_path = run_dir / "raw_results.json"
    if not raw_results_path.exists():
        raise FileNotFoundError(f"Missing {raw_results_path}")

    scored_path = scored_csv
    if scored_path is None:
        cand1 = run_dir / "prob_argmax_matches_score.csv"
        cand2 = run_dir / "option_prob_argmax_matches_score.csv"
        scored_path = cand1 if cand1.exists() else cand2
    if not scored_path.exists():
        raise FileNotFoundError(f"Missing scored CSV {scored_path}")

    raw_records = _load_raw_records(raw_results_path)
    scored_rows = _read_csv_rows(scored_path)

    behavior_rows = behavior_match_rows(raw_records=raw_records, scored_rows=scored_rows)
    behavior_by_key = _index_behaviors(behavior_rows)

    enriched_rows: List[Dict[str, object]] = []
    for r in scored_rows:
        key = (str(r["emotion"]), float(r["intensity"]), int(float(r["item_id"])))
        chosen_behavior, predicted_behavior = behavior_by_key[key]
        rr = dict(r)
        rr["chosen_behavior"] = chosen_behavior
        rr["predicted_behavior"] = predicted_behavior
        enriched_rows.append(rr)

    out_main = run_dir / "prob_argmax_matches_score.csv"
    _write_csv(out_main, enriched_rows)
    _write_csv(run_dir / "behavior_prob_argmax_matches_score.csv", behavior_rows)
    _write_csv(run_dir / "summary_predicted_option_argmax_ratio.csv", predicted_option_argmax_ratios(scored_rows))
    _write_csv(
        run_dir / "summary_predicted_behavior_argmax_ratio.csv",
        predicted_behavior_argmax_ratios(behavior_rows),
    )
    return out_main


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Postprocess an existing option-prob-argmax scoring CSV using raw_results.json.")
    p.add_argument("--run_dir", type=Path, required=True)
    p.add_argument("--scored_csv", type=Path, default=None)
    args = p.parse_args(argv)
    postprocess_run_dir(run_dir=args.run_dir, scored_csv=args.scored_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

