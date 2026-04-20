#!/usr/bin/env python3
# Purpose: measure normalized prefix and first-divergence lexical shortcut accuracy for behavior choices in few-shot files or transformed corpora.

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9']+")
NUMBER_PATTERN = re.compile(
    r"\b(?:\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)\b",
    re.IGNORECASE,
)
PLAYER_LINE_PLACEHOLDER = "player_line"
PLAYER_POWERS = (
    "austria",
    "england",
    "france",
    "germany",
    "italy",
    "russia",
    "turkey",
)
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


def participant_names_from_item(item: dict) -> list[str]:
    participant_block = item.get("participants")
    if not isinstance(participant_block, list) and isinstance(item.get("output"), dict):
        participant_block = item["output"].get("participants")
    if not isinstance(participant_block, list):
        return []

    names: list[str] = []
    for participant in participant_block:
        if not isinstance(participant, dict):
            continue
        name = participant.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def normalize_player_line_phrases(text: str, participant_names: list[str]) -> str:
    value = re.sub(r"\byour\s+line\b", PLAYER_LINE_PLACEHOLDER, text)
    names = {name.lower() for name in participant_names if name.strip()}
    names.update(PLAYER_POWERS)
    for name in sorted(names, key=len, reverse=True):
        pattern = re.compile(rf"\b{re.escape(name)}'s\s+line\b")
        value = pattern.sub(PLAYER_LINE_PLACEHOLDER, value)
    return value


def normalize_text(text: str, participant_names: list[str] | None = None) -> str:
    value = text.lower()
    value = NUMBER_PATTERN.sub("<num>", value)
    value = normalize_player_line_phrases(value, participant_names or [])
    for pattern, replacement in PLACEHOLDER_PATTERNS:
        value = pattern.sub(replacement, value)
    return " ".join(TOKEN_RE.findall(value))


def iter_choice_sets(payload) -> list[list[tuple[str, str]]]:
    choice_sets: list[list[tuple[str, str]]] = []
    if not isinstance(payload, list):
        raise ValueError("input JSON must be a list")

    for item in payload:
        if not isinstance(item, dict):
            continue
        participant_names = participant_names_from_item(item)
        choice_block = None
        if isinstance(item.get("output"), dict):
            choice_block = item["output"].get("behavior_choices")
        if choice_block is None:
            choice_block = item.get("behavior_choices")
        if not isinstance(choice_block, dict):
            continue

        local_rows: list[tuple[str, str]] = []
        for label, value in choice_block.items():
            if isinstance(label, str) and label.strip() and isinstance(value, str) and value.strip():
                local_rows.append((normalize_text(value, participant_names), label))

        if local_rows:
            choice_sets.append(local_rows)

    if not choice_sets:
        raise ValueError("no behavior choices found")
    return choice_sets


def iter_labeled_choices(payload) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for choice_set in iter_choice_sets(payload):
        rows.extend(choice_set)
    return rows


def prefix_of(text: str, token_count: int) -> str:
    tokens = text.split()
    return " ".join(tokens[:token_count]) if tokens else ""


def strip_shared_prefix(choice_set: list[tuple[str, str]]) -> list[tuple[str, str]]:
    token_rows = [text.split() for text, _ in choice_set]
    shared_length = 0
    while True:
        if any(len(tokens) <= shared_length for tokens in token_rows):
            break
        candidate = token_rows[0][shared_length]
        if any(tokens[shared_length] != candidate for tokens in token_rows[1:]):
            break
        shared_length += 1

    stripped_rows: list[tuple[str, str]] = []
    for (text, label), tokens in zip(choice_set, token_rows):
        remainder = " ".join(tokens[shared_length:])
        stripped_rows.append((remainder, label))
    return stripped_rows


def balanced_accuracy(samples: list[tuple[str, str]], token_count: int) -> tuple[float, Counter]:
    correct: defaultdict[str, int] = defaultdict(int)
    total: defaultdict[str, int] = defaultdict(int)
    informative_prefixes: Counter = Counter()

    for index, (text, label) in enumerate(samples):
        prefix = prefix_of(text, token_count)
        total[label] += 1

        counts: Counter = Counter()
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
    choice_sets = iter_choice_sets(read_json(Path(args.json_path)))
    samples = [row for choice_set in choice_sets for row in choice_set]
    divergence_samples = [
        row for choice_set in choice_sets for row in strip_shared_prefix(choice_set)
    ]

    prefix1_acc, prefix1_prefixes = balanced_accuracy(samples, 1)
    prefix2_acc, prefix2_prefixes = balanced_accuracy(samples, 2)
    divergence1_acc, divergence1_prefixes = balanced_accuracy(divergence_samples, 1)
    divergence2_acc, divergence2_prefixes = balanced_accuracy(divergence_samples, 2)

    print(f"samples={len(samples)}")
    print(f"prefix1_acc={prefix1_acc:.4f}")
    print(f"prefix2_acc={prefix2_acc:.4f}")
    print(f"divergence1_acc={divergence1_acc:.4f}")
    print(f"divergence2_acc={divergence2_acc:.4f}")

    print("top_prefix1_prefixes:")
    for prefix, count in prefix1_prefixes.most_common(10):
        print(f"  {prefix or '<empty>'}: {count}")

    print("top_prefix2_prefixes:")
    for prefix, count in prefix2_prefixes.most_common(10):
        print(f"  {prefix or '<empty>'}: {count}")

    print("top_divergence1_prefixes:")
    for prefix, count in divergence1_prefixes.most_common(10):
        print(f"  {prefix or '<empty>'}: {count}")

    print("top_divergence2_prefixes:")
    for prefix, count in divergence2_prefixes.most_common(10):
        print(f"  {prefix or '<empty>'}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
