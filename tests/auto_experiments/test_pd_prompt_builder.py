"""Tests: auto_experiments/task-similarity/pd_prompt_builder.py
Purpose: validate Prisoner's Dilemma prompt construction and label mapping."""

import random

from auto_experiments.task_similarity import pd_prompt_builder as builder


def test_pair_labels_defect_first():
    entry = {
        "description": "desc",
        "behavior_choices": {"defect": "D-option", "cooperate": "C-option"},
    }
    rng = random.Random(1)  # deterministic ordering
    pair = builder.build_pair(entry, rng)

    assert pair.meta.opt_a == "D-option"
    assert pair.meta.opt_b == "C-option"
    assert pair.meta.defect_label == "A"
    assert pair.meta.cooperate_label == "B"
    assert "Assistant: A)" in pair.positive
    assert "D-option" in pair.positive
    assert "Assistant: B)" in pair.negative
    assert "C-option" in pair.negative


def test_pair_labels_cooperate_first():
    entry = {
        "description": "desc",
        "behavior_choices": {"defect": "D-option", "cooperate": "C-option"},
    }
    rng = random.Random(0)  # different ordering
    pair = builder.build_pair(entry, rng)

    assert pair.meta.opt_a == "C-option"
    assert pair.meta.opt_b == "D-option"
    assert pair.meta.defect_label == "B"
    assert pair.meta.cooperate_label == "A"
    assert "Assistant: B)" in pair.positive
    assert "D-option" in pair.positive
    assert "Assistant: A)" in pair.negative
    assert "C-option" in pair.negative
