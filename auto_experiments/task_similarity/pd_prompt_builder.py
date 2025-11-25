"""
Responsible: auto_experiments/task-similarity/pd_prompt_builder.py
Purpose: Build Prisoner's Dilemma prompts with randomized option ordering and paired assistant labels.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict


@dataclass
class PairMeta:
    opt_a: str
    opt_b: str
    defect_label: str
    cooperate_label: str


@dataclass
class PromptPair:
    positive: str
    negative: str
    meta: PairMeta


def _build_prompt(description: str, opt_a: str, opt_b: str, assistant_label: str) -> str:
    body = (
        "User: Imagine you are Alice and consider the following scenario: "
        f"{description}\n"
        "Choices:\n"
        f"A) {opt_a}\n"
        f"B) {opt_b}\n"
        f"Assistant: {assistant_label}"
    )
    return body


def build_pair(entry: Dict[str, object], rng: random.Random) -> PromptPair:
    """
    Construct a positive/negative prompt pair for a single scenario.

    Positive = defection label; Negative = cooperation label.
    Randomly assigns option order per scenario using the provided RNG.
    """
    choices = entry["behavior_choices"]
    description = entry["description"]
    defect_text = choices["defect"]
    cooperate_text = choices["cooperate"]

    defect_first = rng.random() < 0.5
    if defect_first:
        opt_a, opt_b = defect_text, cooperate_text
        defect_label, cooperate_label = "A", "B"
    else:
        opt_a, opt_b = cooperate_text, defect_text
        defect_label, cooperate_label = "B", "A"

    positive = _build_prompt(description, opt_a, opt_b, defect_label)
    negative = _build_prompt(description, opt_a, opt_b, cooperate_label)

    meta = PairMeta(
        opt_a=opt_a,
        opt_b=opt_b,
        defect_label=defect_label,
        cooperate_label=cooperate_label,
    )
    return PromptPair(positive=positive, negative=negative, meta=meta)
