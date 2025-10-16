"""
Tests for result_analysis/cognitive_complexity metrics.
This suite validates:
- Neutral average score per item from detailed_results.csv
- cognitive_complexity ratio = avg(thinking neutral) / avg(no-thinking neutral)

We use real Qwen3-32B files present in the repo to anchor expectations.
"""

from pathlib import Path
import sys, os

# Ensure repo root on sys.path for local package imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


THINKING = Path(
    "results/fantom_qwen3/thinking-nogen/"
    "Qwen3-32B-AWQ_fantom_short_belief_choice_accessible_20250929_152542/"
    "detailed_results.csv"
)
NO_THINKING = Path(
    "results/fantom_qwen3/no-thinking-nogen/"
    "Qwen3-32B-AWQ_fantom_short_belief_choice_accessible_20250929_110307/"
    "detailed_results.csv"
)


def test_files_exist():
    assert THINKING.is_file(), f"Missing: {THINKING}"
    assert NO_THINKING.is_file(), f"Missing: {NO_THINKING}"


def test_neutral_avg_scores_sample():
    # Lazy import to keep test collection fast if files are missing
    from result_analysis.cognitive_complexity.metrics import neutral_avg_by_item

    # Item chosen from exploratory inspection
    item_id = "fantom_bc_172"
    avg_no = neutral_avg_by_item(NO_THINKING)[item_id]
    avg_think = neutral_avg_by_item(THINKING)[item_id]

    # Expectation based on actual CSV contents:
    # no-thinking neutral scores: [1.0, 0.0, 0.0] -> avg 1/3
    # thinking   neutral scores: [1.0, 1.0, 1.0] -> avg 1.0
    assert abs(avg_no - (1.0 / 3.0)) < 1e-9
    assert abs(avg_think - 1.0) < 1e-9


def test_cognitive_complexity_ratio():
    from result_analysis.cognitive_complexity.metrics import cognitive_complexity_ratio

    item_id = "fantom_bc_172"
    ratio = cognitive_complexity_ratio(THINKING, NO_THINKING, item_id)
    assert abs(ratio - 3.0) < 1e-9
