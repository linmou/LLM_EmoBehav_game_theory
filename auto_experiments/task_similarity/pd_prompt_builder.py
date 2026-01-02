"""
Responsible: auto_experiments/task_similarity/pd_prompt_builder.py
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
    description: str


@dataclass
class PromptPair:
    positive: str  # defection answer included
    negative: str  # cooperation answer included
    meta: PairMeta


def _build_prompt(description: str, opt_a: str, opt_b: str, assistant_label: str, assistant_text: str) -> str:
    body = (
        "User: Imagine you are Alice and consider the following scenario: "
        f"{description}\n"
        "Choices:\n"
        f"A) {opt_a}\n"
        f"B) {opt_b}\n"
        f"Assistant: {assistant_label}) {assistant_text}"
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
        defect_answer, coop_answer = defect_text, cooperate_text
    else:
        opt_a, opt_b = cooperate_text, defect_text
        defect_label, cooperate_label = "B", "A"
        defect_answer, coop_answer = defect_text, cooperate_text

    positive = _build_prompt(description, opt_a, opt_b, defect_label, defect_answer)
    negative = _build_prompt(description, opt_a, opt_b, cooperate_label, coop_answer)

    meta = PairMeta(
        opt_a=opt_a,
        opt_b=opt_b,
        defect_label=defect_label,
        cooperate_label=cooperate_label,
        description=description,
    )
    return PromptPair(positive=positive, negative=negative, meta=meta)


def build_inference_prompt(description: str, opt_a: str, opt_b: str) -> str:
    return (
        "User: Imagine you are Alice and consider the following scenario: "
        f"{description}\n"
        "Choices:\n"
        f"A) {opt_a}\n"
        f"B) {opt_b}\n"
        "Assistant:"
    )
