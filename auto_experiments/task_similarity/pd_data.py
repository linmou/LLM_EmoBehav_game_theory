"""
Responsible: auto_experiments/task-similarity/pd_data.py
Purpose: Load Prisoner's Dilemma scenarios and build paired prompts for RepReader training/validation.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from .pd_prompt_builder import PromptPair, build_pair


@dataclass
class PDPairBundle:
    pairs: List[PromptPair]
    train_pairs: List[PromptPair]
    test_pairs: List[PromptPair]


def load_pairs(json_path: Path, seed: int = 0) -> List[PromptPair]:
    rng = random.Random(seed)
    data = json.loads(json_path.read_text())
    pairs: List[PromptPair] = []
    for entry in data:
        pairs.append(build_pair(entry, rng))
    return pairs


def split_pairs(pairs: Sequence[PromptPair], train_ratio: float = 0.5, seed: int = 0) -> Tuple[List[PromptPair], List[PromptPair]]:
    rng = random.Random(seed)
    idxs = list(range(len(pairs)))
    rng.shuffle(idxs)
    cut = int(len(idxs) * train_ratio)
    train_idx = idxs[:cut]
    test_idx = idxs[cut:]
    train_pairs = [pairs[i] for i in train_idx]
    test_pairs = [pairs[i] for i in test_idx]
    return train_pairs, test_pairs


def build_repreader_dataset(pairs: Sequence[PromptPair]) -> dict:
    """
    Build data/labels dictionaries compatible with neuro_manipulation.utils.get_rep_reader.

    Data order: positive (defect) first, negative (cooperate) second for each pair.
    Labels: [1, 0] per pair to mark the defection prompt as the positive example.
    """
    data: List[str] = []
    labels: List[List[int]] = []
    for pair in pairs:
        data.extend([pair.positive, pair.negative])
        labels.append([1, 0])
    return {"data": data, "labels": labels}


def build_pd_pair_bundle(json_path: Path, seed: int = 0) -> PDPairBundle:
    pairs = load_pairs(json_path, seed=seed)
    train_pairs, test_pairs = split_pairs(pairs, seed=seed)
    return PDPairBundle(pairs=pairs, train_pairs=train_pairs, test_pairs=test_pairs)
