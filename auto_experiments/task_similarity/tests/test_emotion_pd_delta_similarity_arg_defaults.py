"""
Responsible: auto_experiments/task_similarity/emotion_pd_delta_similarity.py
Purpose: Ensure CLI defaults stay stable (especially dataset split defaults).
"""

from __future__ import annotations


def test_default_split_is_all() -> None:
    from auto_experiments.task_similarity.emotion_pd_delta_similarity import build_arg_parser

    p = build_arg_parser()
    assert p.get_default("split") == "all"

