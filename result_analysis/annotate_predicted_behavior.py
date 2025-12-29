"""
Annotate option-probability scoring CSV with predicted behavior labels.

Input:
- option_prob_argmax_matches_score_*.csv (from score_game_theory_option_prob_match.py)
- raw_results.json (same run dir) for per-(emotion,intensity,item_id) shuffled option texts

Output:
- CSV with new columns:
  - predicted_option_text
  - predicted_behavior (cooperate/defect/unknown)
  - chosen_option_text (if chosen_option_id present)
  - chosen_behavior (if chosen_option_id present)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple


def classify_pd_behavior(option_text: str) -> str:
    t = option_text.strip().lower()
    defect_words = ("defect", "betray", "cheat", "backstab")
    coop_words = ("cooperate", "stay silent", "remain silent", "keep quiet")

    if any(w in t for w in defect_words):
        return "defect"
    if any(w in t for w in coop_words):
        return "cooperate"
    return "unknown"


def _load_raw_options_by_key(raw_results_path: Path) -> Dict[Tuple[str, float, int], Dict[int, str]]:
    raw = json.loads(raw_results_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError("raw_results.json must be a list")
    out: Dict[Tuple[str, float, int], Dict[int, str]] = {}
    for r in raw:
        if not isinstance(r, dict):
            continue
        emotion = r.get("emotion")
        intensity = r.get("intensity")
        item_id = r.get("item_id")
        if emotion is None or intensity is None or item_id is None:
            continue
        md = r.get("metadata") or {}
        opts = (md.get("item_metadata") or {}).get("options")
        if not isinstance(opts, list):
            continue
        m: Dict[int, str] = {}
        for opt in opts:
            if not isinstance(opt, dict):
                continue
            if "id" not in opt or "text" not in opt:
                continue
            opt_id = int(opt["id"])
            text = str(opt["text"])
            behavior = opt.get("behavior")
            if behavior is None:
                m[opt_id] = json.dumps({"text": text, "behavior": ""}, ensure_ascii=False)
            else:
                m[opt_id] = json.dumps({"text": text, "behavior": str(behavior)}, ensure_ascii=False)
        out[(str(emotion), float(intensity), int(item_id))] = m
    return out


def _unpack_option_payload(payload: str) -> Tuple[str, str]:
    try:
        obj = json.loads(payload)
        if isinstance(obj, dict):
            return str(obj.get("text", "")), str(obj.get("behavior", ""))
    except Exception:
        pass
    return payload, ""


def annotate_csv(
    *,
    scored_csv_path: Path,
    raw_results_path: Path,
    out_csv_path: Path,
) -> Path:
    options_by_key = _load_raw_options_by_key(raw_results_path)

    with scored_csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if not reader.fieldnames:
            raise ValueError("Missing CSV header")
        fieldnames = list(reader.fieldnames)

    for col in [
        "predicted_option_text",
        "predicted_behavior",
        "chosen_option_text",
        "chosen_behavior",
    ]:
        if col not in fieldnames:
            fieldnames.append(col)

    for row in rows:
        item_id = int(float(row["item_id"]))
        emotion = str(row["emotion"])
        intensity = float(row["intensity"])
        pred_id = int(float(row["predicted_option_id"]))
        chosen_raw = row.get("chosen_option_id")
        chosen_id = None if chosen_raw in (None, "", "nan", "NaN") else int(float(chosen_raw))

        opts = options_by_key.get((emotion, intensity, item_id))
        if not opts:
            raise KeyError(
                f"Missing options for emotion={emotion} intensity={intensity} item_id={item_id} in raw_results.json"
            )

        pred_payload = opts.get(pred_id, "")
        pred_text, pred_behavior = _unpack_option_payload(pred_payload)
        row["predicted_option_text"] = pred_text
        row["predicted_behavior"] = pred_behavior or classify_pd_behavior(pred_text)

        if chosen_id is None:
            row["chosen_option_text"] = ""
            row["chosen_behavior"] = ""
        else:
            chosen_payload = opts.get(chosen_id, "")
            chosen_text, chosen_behavior = _unpack_option_payload(chosen_payload)
            row["chosen_option_text"] = chosen_text
            row["chosen_behavior"] = chosen_behavior or classify_pd_behavior(chosen_text)

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return out_csv_path


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Annotate scoring CSV with predicted behavior labels.")
    p.add_argument("--run_dir", type=Path, required=True)
    p.add_argument("--scored_csv", type=Path, required=True)
    p.add_argument("--out_csv", type=Path, default=None)
    args = p.parse_args(argv)

    raw_results_path = args.run_dir / "raw_results.json"
    if not raw_results_path.exists():
        raise FileNotFoundError(f"Missing {raw_results_path}")

    out_csv_path = args.out_csv or args.scored_csv.with_name(args.scored_csv.stem + "_annotated.csv")
    annotate_csv(scored_csv_path=args.scored_csv, raw_results_path=raw_results_path, out_csv_path=out_csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
