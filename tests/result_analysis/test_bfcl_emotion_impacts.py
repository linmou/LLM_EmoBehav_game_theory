"""
tests/result_analysis/test_bfcl_emotion_impacts.py
Purpose: TDD for parsing BFCL run outputs and computing emotion deltas; per-category aggregation.
Targets: result_analysis/fantom_emotion_impacts.py (reader),
         result_analysis/generate_bfcl_emotion_summary.py (BFCL-specific aggregation).
"""

import math
from pathlib import Path
import sys

import pytest


SAMPLE_RUN = Path(
    "results/bfcl/live/Qwen3-0.6B_bfcl_live_simple_20250928_020225"
)


def _ensure_repo_on_path():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


@pytest.mark.skipif(not SAMPLE_RUN.exists(), reason="Sample BFCL run not present")
def test_parse_neutral_and_delta():
    _ensure_repo_on_path()
    from result_analysis.fantom_emotion_impacts import read_summary_overall, compute_emotion_deltas

    rows = read_summary_overall(SAMPLE_RUN)
    assert "neutral" in rows
    assert math.isclose(rows["neutral"].mean_of_means, 0.558559, rel_tol=0, abs_tol=1e-6)
    deltas = compute_emotion_deltas(SAMPLE_RUN)
    # anger 0.474903 - neutral 0.558559 = -0.083656
    assert math.isclose(deltas["anger"], 0.474903 - 0.558559, rel_tol=0, abs_tol=1e-6)


@pytest.mark.skipif(not SAMPLE_RUN.exists(), reason="Sample BFCL run not present")
def test_parse_bfcl_category_and_aggregate():
    _ensure_repo_on_path()
    from result_analysis.generate_bfcl_emotion_summary import parse_bfcl_category, RunInfo, aggregate_by_model_and_category

    assert parse_bfcl_category("live_simple") == "simple"
    assert parse_bfcl_category("live_parallel") == "parallel"
    assert parse_bfcl_category("live_parallel_multiple") == "parallel_multiple"
    assert parse_bfcl_category("live_multiple") == "multiple"

    run = RunInfo(
        path=SAMPLE_RUN,
        model="Qwen3-0.6B",
        task_type="live_simple",
    )
    agg = aggregate_by_model_and_category([run])
    assert "Qwen3-0.6B" in agg
    stats = agg["Qwen3-0.6B"]["simple"]
    assert math.isclose(stats["neutral_mean_avg"], 0.558559, rel_tol=0, abs_tol=1e-6)
    assert math.isclose(stats["delta_anger"], 0.474903 - 0.558559, rel_tol=0, abs_tol=1e-6)
