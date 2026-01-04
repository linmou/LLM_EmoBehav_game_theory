"""Utility to summarize lexical gradients inside the escalation Diplomacy dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


AGGRESSIVE_TOKENS = {
    "escalate",
    "advance",
    "push",
    "attack",
    "drive",
    "pressure",
    "contest",
    "strike",
    "invade",
    "reinforce",
    "support",
    "aggressive",
    "command",
    "deploy",
    "surge",
}

DEESCALATION_TOKENS = {
    "withdraw",
    "hold",
    "maintain",
    "delay",
    "avoid",
    "reduce",
    "de-escalate",
    "cede",
    "retain",
    "watch",
    "monitor",
    "quiet",
    "peace",
}


def _tokenize(text: str) -> List[str]:
    return [token.strip(".,:;!?\"'").lower() for token in text.split()]


def _classify_option(text: str) -> str:
    tokens = set(_tokenize(text))
    if tokens & AGGRESSIVE_TOKENS:
        return "aggressive"
    if tokens & DEESCALATION_TOKENS:
        return "deescalate"
    return "neutral"


def _load_records(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def summarize_dataset(path: Path) -> Dict[str, Dict]:
    gradient_stats: Dict[int, Counter] = defaultdict(Counter)
    label_stats: Counter = Counter()
    per_scenario: List[Tuple[str, List[str]]] = []

    for record in _load_records(path):
        label = record.get("whose_option") or record.get("your_country") or "UNKNOWN"
        label_stats[label] += 1
        options = record.get("gradient_options") or record.get("options") or []
        bucket_labels = []
        for opt in options:
            text = opt.get("text") if isinstance(opt, dict) else str(opt)
            option_id = int(opt.get("id", len(bucket_labels) + 1)) if isinstance(opt, dict) else len(bucket_labels) + 1
            category = _classify_option(text)
            gradient_stats[option_id][category] += 1
            bucket_labels.append(category)
        per_scenario.append((record.get("scenario", "unknown"), bucket_labels))

    summary = {
        "total_scenarios": sum(label_stats.values()),
        "whose_option_counts": label_stats,
        "gradient_bucket_stats": gradient_stats,
    }

    # Convert Counters to serializable dicts
    summary["whose_option_counts"] = dict(summary["whose_option_counts"])
    summary["gradient_bucket_stats"] = {
        bucket: dict(counter) for bucket, counter in summary["gradient_bucket_stats"].items()
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to Diplomacy escalation jsonl file")
    parser.add_argument("--output", type=Path, required=True, help="Where to store the summary JSON")
    args = parser.parse_args()

    summary = summarize_dataset(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    total = summary["total_scenarios"]
    print(f"Total scenarios: {total}")
    print("Whose option distribution (top 5):")
    sorted_labels = sorted(summary["whose_option_counts"].items(), key=lambda kv: kv[1], reverse=True)
    for label, count in sorted_labels[:5]:
        print(f"  {label}: {count}")
    print("Gradient bucket aggression ratios:")
    for bucket, counts in sorted(summary["gradient_bucket_stats"].items()):
        bucket_total = sum(counts.values())
        aggressive_share = counts.get("aggressive", 0) / bucket_total if bucket_total else 0.0
        deescalate_share = counts.get("deescalate", 0) / bucket_total if bucket_total else 0.0
        print(
            f"  Option {bucket}: aggressive={aggressive_share:.2f}, "
            f"de-escalate={deescalate_share:.2f}, total={bucket_total}"
        )
    print(f"Summary saved to {args.output}")


if __name__ == "__main__":
    main()
