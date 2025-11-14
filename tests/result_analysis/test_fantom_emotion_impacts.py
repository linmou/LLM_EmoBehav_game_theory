"""
tests/result_analysis/test_fantom_emotion_impacts.py
Purpose: TDD for parsing Fantom run outputs and computing emotion deltas
Targets: result_analysis/fantom_emotion_impacts.py
"""

import math
from pathlib import Path
import sys

import pytest


SAMPLE_RUN = Path(
    "results/fantom/Llama-3.2-1B-Instruct_fantom_full_answerability_binary_accessible_20250929_153207"
)


@pytest.mark.skipif(not SAMPLE_RUN.exists(), reason="Sample Fantom run not present")
def test_parse_neutral_mean():
    # Ensure repo root is on sys.path for local package imports
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from result_analysis.fantom_emotion_impacts import read_summary_overall

    rows = read_summary_overall(SAMPLE_RUN)
    # sanity: neutral row exists and value matches the CSV (approx)
    assert "neutral" in rows
    neutral = rows["neutral"].mean_of_means
    assert math.isclose(neutral, 0.383292, rel_tol=0, abs_tol=1e-6)


@pytest.mark.skipif(not SAMPLE_RUN.exists(), reason="Sample Fantom run not present")
def test_compute_deltas_single_run():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from result_analysis.fantom_emotion_impacts import compute_emotion_deltas

    deltas = compute_emotion_deltas(SAMPLE_RUN)
    # anger mean 0.079443 - neutral 0.383292 = -0.303849
    assert "anger" in deltas
    assert math.isclose(deltas["anger"], 0.079443 - 0.383292, rel_tol=0, abs_tol=1e-6)
