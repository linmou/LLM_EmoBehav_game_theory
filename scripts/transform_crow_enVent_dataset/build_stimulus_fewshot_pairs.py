"""
scripts/build_stimulus_fewshot_pairs.py

Purpose:
  Build a small few-shot pair set by matching crowd-enVent stimuli to the
  closest text-style stimuli (same emotion) using simple string similarity.

Output:
  JSONL with records:
    {"emotion": "...", "crowd": "...", "text": "...", "score": 0.0}
"""

from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "but",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "me",
    "my",
    "no",
    "not",
    "of",
    "on",
    "or",
    "our",
    "out",
    "she",
    "so",
    "some",
    "someone",
    "something",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "too",
    "under",
    "up",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "without",
    "would",
    "you",
    "your",
    "yours",
    "first",
    "time",
}


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\.{2,}", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _content_tokens(s: str) -> set[str]:
    tokens = {t for t in _norm(s).split(" ") if t and t not in STOPWORDS}
    return tokens


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _best_match(source: str, candidates: Iterable[str]) -> Tuple[str, float]:
    best_text = ""
    best_score = -1.0
    n_source = _norm(source)
    tok_source = _content_tokens(source)
    for cand in candidates:
        tok_cand = _content_tokens(cand)
        seq = SequenceMatcher(None, n_source, _norm(cand)).ratio()
        jac = _jaccard(tok_source, tok_cand)
        score = 0.4 * seq + 0.6 * jac
        if score > best_score:
            best_text = cand
            best_score = score
    return best_text, best_score


def _load_json_list(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError(f"Expected a JSON list[str] in {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Match crowd-enVent stimuli to text-style stimuli to create few-shot pairs."
    )
    parser.add_argument(
        "--crowd_dir",
        default="data/stimulus/crowd-enVent",
        help="Directory containing crowd-enVent emotion JSON files",
    )
    parser.add_argument(
        "--text_dir",
        default="data/stimulus/text",
        help="Directory containing text emotion JSON files",
    )
    parser.add_argument(
        "--out_path",
        default="data/stimulus/few_shot/crowd_enVent_to_text_pairs.v1.jsonl",
        help="Where to write JSONL few-shot pairs",
    )
    parser.add_argument(
        "--per_emotion",
        type=int,
        default=4,
        help="Max pairs to keep per emotion (greedy by similarity score)",
    )
    parser.add_argument(
        "--min_score",
        type=float,
        default=0.55,
        help="Minimum similarity score for keeping a pair",
    )
    args = parser.parse_args()

    crowd_dir = Path(args.crowd_dir)
    text_dir = Path(args.text_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_pairs: List[Dict[str, object]] = []
    for emo in EMOTIONS:
        crowd_path = crowd_dir / f"{emo}.json"
        text_path = text_dir / f"{emo}.json"
        if not crowd_path.exists() or not text_path.exists():
            raise FileNotFoundError(f"Missing {emo}.json in {crowd_dir} or {text_dir}")

        crowd = _load_json_list(crowd_path)
        text = _load_json_list(text_path)

        scored: List[Tuple[float, str, str]] = []
        for c in crowd:
            best_t, score = _best_match(c, text)
            scored.append((score, c, best_t))
        scored.sort(reverse=True)

        used_text: set[str] = set()
        kept = 0
        for score, c, t in scored:
            if kept >= args.per_emotion:
                break
            if score < args.min_score:
                break
            if t in used_text:
                continue
            used_text.add(t)
            all_pairs.append({"emotion": emo, "crowd": c, "text": t, "score": score})
            kept += 1

    with out_path.open("w", encoding="utf-8") as f:
        for rec in all_pairs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_pairs)} pairs to {out_path}")


if __name__ == "__main__":
    main()
