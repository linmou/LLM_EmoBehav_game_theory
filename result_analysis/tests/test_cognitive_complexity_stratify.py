"""
Tests for stratification utilities used in cognitive complexity analysis.
"""

from pathlib import Path
import sys

# Ensure import path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_tertiles_basic():
    from result_analysis.cognitive_complexity.stratify import tertile_bins

    values = [1, 2, 3, 4, 5, 6]
    # Expect roughly equal-sized bins: [1,2], [3,4], [5,6]
    lo, mid, hi = tertile_bins(values)
    assert lo == (1, 2)
    assert mid == (3, 4)
    assert hi == (5, 6)

