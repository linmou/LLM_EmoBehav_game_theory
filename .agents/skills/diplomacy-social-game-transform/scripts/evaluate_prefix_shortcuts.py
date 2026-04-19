#!/usr/bin/env python3
# Purpose: measure first-token and first-two-token lexical shortcut accuracy for behavior choices in few-shot files or transformed corpora.

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9']+")
PLACEHOLDER_PATTERNS = [
    (re.compile(r"\b(st\.?\s+petersburg|constantinople|rumania|marseilles|sweden|belgium|gascony|tuscany|piedmont|serbia|munich|venice|naples|ankara|smyrna|greece|tunisia|moscow|kiel|berlin|rome|edinburgh|sevastopol)\b", re.IGNORECASE), "<loc>"),
    (re.compile(r"\b(austria|england|france|germany|italy|russia|turkey)\b", re.IGNORECASE), "<player>"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(text: str) -> str:
    value = text.lower()
    for pattern, replacement in PLACEHOLDER_PATTERNS:
        value = pattern.sub(replacement, value)
    return " ".join(TOKEN_RE.findall(value))


def iter_labeled_choices(payload) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if not isinstance(payload, list):
        raise ValueError("input JSON must be a list")

    for item in payload:
        if not isinstance(item, dict):
            continue
        choice_block = None
        if isinstance(item.get("output"), dict):
            choice_block = item["output"].get("behavior_choices")
        if choice_block is None:
            choice_block = item.get("behavior_choices")
        if not isinstance(choice_block, dict):
            continue

        for label, value in choice_block.items():
            if isinstance(label, str) and label.strip() and isinstance(value, str) and value.strip():
                rows.append((normalize_text(value), label))

    if not rows:
        raise ValueError("no behavior choices found")
    return rows


def prefix_of(text: str, token_count: int) -> str:
    tokens = text.split()
    return " ".join(tokens[:token_count]) if tokens else ""


def balanced_accuracy(samples: list[tuple[str, str]], token_count: int) -> tuple[float, Counter]:
    correct = defaultdict(int)
    total = defaultdict(int)
    informative_prefixes: Counter = Counter()

    for index, (text, label) in enumerate(samples):
        prefix = prefix_of(text, token_count)
        total[label] += 1

        counts = Counter()
        for other_index, (other_text, other_label) in enumerate(samples):
            if index == other_index:
                continue
            if prefix_of(other_text, token_count) == prefix:
                counts[other_label] += 1

        if counts:
            predicted = counts.most_common(1)[0][0]
            informative_prefixes[prefix] += sum(counts.values())
        else:
            predicted = Counter(other_label for other_i, (_, other_label) in enumerate(samples) if other_i != index).most_common(1)[0][0]

        if predicted == label:
            correct[label] += 1

    recalls = []
    for label, label_total in total.items():
        recalls.append(correct[label] / label_total if label_total else 0.0)
    return sum(recalls) / len(recalls), informative_prefixes


def main() -> int:
    args = parse_args()
    samples = iter_labeled_choices(read_json(Path(args.json_path)))

    first_acc, first_prefixes = balanced_accuracy(samples, 1)
    second_acc, second_prefixes = balanced_accuracy(samples, 2)

    print(f"samples={len(samples)}")
    print(f"first_token_accuracy={first_acc:.4f}")
    print(f"first_two_token_accuracy={second_acc:.4f}")

    print("top_first_token_prefixes:")
    for prefix, count in first_prefixes.most_common(10):
        print(f"  {prefix or '<empty>'}: {count}")

    print("top_first_two_token_prefixes:")
    for prefix, count in second_prefixes.most_common(10):
        print(f"  {prefix or '<empty>'}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
