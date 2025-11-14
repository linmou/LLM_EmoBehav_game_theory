"""
tests/result_analysis/test_fantom_emotion_by_task.py
Purpose: TDD for per-task-family aggregation on Fantom outputs.
Targets: result_analysis/generate_fantom_emotion_summary.py functions.
"""

import math
from pathlib import Path
import sys

import pytest


SAMPLE_RUN = Path(
    "results/fantom/Llama-3.2-1B-Instruct_fantom_full_answerability_binary_accessible_20250929_153207"
)


def _ensure_repo_on_path():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


@pytest.mark.skipif(not SAMPLE_RUN.exists(), reason="Sample Fantom run not present")
def test_parse_task_family():
    _ensure_repo_on_path()
    from result_analysis.generate_fantom_emotion_summary import parse_task_family

    fam = parse_task_family("full_answerability_binary_accessible")
    assert fam == "answerability"
    assert parse_task_family("short_fact") == "fact"
    assert parse_task_family("full_infoaccessibility_list_inaccessible") == "infoaccessibility"
    assert parse_task_family("short_belief_choice_inaccessible") == "belief_choice"


@pytest.mark.skipif(not SAMPLE_RUN.exists(), reason="Sample Fantom run not present")
def test_aggregate_by_model_and_task_single_run():
    _ensure_repo_on_path()
    from result_analysis.generate_fantom_emotion_summary import RunInfo, aggregate_by_model_and_task

    run = RunInfo(
        path=SAMPLE_RUN,
        model="Llama-3.2-1B-Instruct",
        task_type="full_answerability_binary_accessible",
    )
    agg = aggregate_by_model_and_task([run])
    assert "Llama-3.2-1B-Instruct" in agg
    per_task = agg["Llama-3.2-1B-Instruct"]
    assert "answerability" in per_task
    stats = per_task["answerability"]
    # From summary_overall.csv for this run
    # neutral mean_of_means = 0.383292
    assert math.isclose(stats["neutral_mean_avg"], 0.383292, rel_tol=0, abs_tol=1e-6)
    # anger mean 0.079443 - neutral 0.383292 = -0.303849
    assert math.isclose(stats["delta_anger"], 0.079443 - 0.383292, rel_tol=0, abs_tol=1e-6)

