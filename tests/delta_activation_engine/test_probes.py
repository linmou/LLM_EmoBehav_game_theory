# Responsible file: delta_activation_engine/probes.py
# Purpose: Provide the 5 generic instruction templates (mirroring delta_activations.py) and ensure stability.

import hashlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from delta_activation_engine.prompts.probes_texts import get_generic_probes


def stable_hash(texts):
    # Deterministic hash ignoring ordering changes is optional; we keep order strict here.
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode('utf-8'))
    return h.hexdigest()


def test_generic_probes_content_and_count():
    probes = get_generic_probes()
    assert isinstance(probes, list)
    assert len(probes) == 5

    # Expect the canonical templates; any drift should fail this test
    expected = [
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{task} Input:{input}\n\n### Response:",
        "The task described below requires a response that completes the request accurately.\n\n### Instruction:\n{task} Input:{input}\n\n### Response:",
        "Below is a description of a task. Provide a response that aligns with the requirements.\n\n### Instruction:\n{task} Input:{input}\n\n### Response:",
        "The following instruction outlines a task. Generate a response that meets the specified request.\n\n### Instruction:\n{task} Input:{input}\n\n### Response:",
        "You are given an instruction and input. Write a response that completes the task as requested.\n\n### Instruction:\n{task} Input:{input}\n\n### Response:",
    ]
    assert probes == expected

    # Additional stability check via hash
    assert stable_hash(probes) == stable_hash(expected)
