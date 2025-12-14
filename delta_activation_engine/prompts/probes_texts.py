"""
Responsible: delta_activation_engine/prompts/probes_texts.py
Purpose: Provide a small fixed set of neutral instruction probes.
"""

from __future__ import annotations

from typing import List


def get_generic_probes() -> List[str]:
    # Canonical five templates (parity with tests and original reference)
    return [
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{task} Input:{input}\n\n### Response:",
        "The task described below requires a response that completes the request accurately.\n\n### Instruction:\n{task} Input:{input}\n\n### Response:",
        "Below is a description of a task. Provide a response that aligns with the requirements.\n\n### Instruction:\n{task} Input:{input}\n\n### Response:",
        "The following instruction outlines a task. Generate a response that meets the specified request.\n\n### Instruction:\n{task} Input:{input}\n\n### Response:",
        "You are given an instruction and input. Write a response that completes the task as requested.\n\n### Instruction:\n{task} Input:{input}\n\n### Response:",
    ]
