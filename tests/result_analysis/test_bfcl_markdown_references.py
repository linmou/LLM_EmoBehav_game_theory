"""
tests/result_analysis/test_bfcl_markdown_references.py
Purpose: Ensure BFCL generated markdown includes reference result paths.
Targets: result_analysis/generate_bfcl_emotion_summary.py and generate_bfcl_significance_summary.py
"""

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
def test_emotion_summaries_have_references():
    _ensure_repo_on_path()
    from result_analysis import generate_bfcl_emotion_summary as gen

    gen.main()
    for md in [
        Path("result_analysis/bfcl_emotion_summary.md"),
        Path("result_analysis/bfcl_emotion_by_category_summary.md"),
    ]:
        txt = md.read_text()
        # require at least the base results dir path appears
        assert "results/bfcl/live/" in txt
        # and ideally a concrete sample run path
        assert str(SAMPLE_RUN) in txt


@pytest.mark.skipif(not SAMPLE_RUN.exists(), reason="Sample BFCL run not present")
def test_significance_summaries_have_references():
    _ensure_repo_on_path()
    from result_analysis import generate_bfcl_significance_summary as gensig

    gensig.main()
    for md in [
        Path("result_analysis/bfcl_emotion_significance_summary.md"),
        Path("result_analysis/bfcl_emotion_by_category_significance.md"),
    ]:
        txt = md.read_text()
        assert "results/bfcl/live/" in txt
        assert str(SAMPLE_RUN) in txt
