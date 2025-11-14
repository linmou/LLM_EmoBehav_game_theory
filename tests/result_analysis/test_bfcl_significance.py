"""
tests/result_analysis/test_bfcl_significance.py
Purpose: TDD for paired t significance using summary_by_repeat.csv on BFCL runs.
Targets: result_analysis/bfcl_significance.py utilities.
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
def test_paired_t_significance_known_run():
    _ensure_repo_on_path()
    from result_analysis.bfcl_significance import paired_t_from_summary_by_repeat

    res = paired_t_from_summary_by_repeat(SAMPLE_RUN)
    anger = res["anger"]
    # df should be n-1 = 2 for 3 repeats
    assert anger["df"] == 2
    # t value approx -24.568 as computed from repeat means
    assert math.isclose(anger["t_stat"], -24.56798, rel_tol=0, abs_tol=1e-3)
    assert anger["significant"] is True


@pytest.mark.skipif(not SAMPLE_RUN.exists(), reason="Sample BFCL run not present")
def test_aggregate_significance_counts():
    _ensure_repo_on_path()
    from result_analysis.bfcl_significance import RunInfo, aggregate_significance_by_model

    runs = [RunInfo(path=SAMPLE_RUN, model="Qwen3-0.6B", task_type="live_simple")]
    agg = aggregate_significance_by_model(runs)
    # With one run, rate equals boolean
    anger_rate = agg["Qwen3-0.6B"]["anger"]["sig_rate"]
    assert math.isclose(anger_rate, 1.0, rel_tol=0, abs_tol=1e-9)
