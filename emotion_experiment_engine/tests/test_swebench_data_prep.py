"""
Responsible file: emotion_experiment_engine/swebench_data_prep.py
Purpose: TDD for Phase 0 data preparation automation (offline retrieval + text dataset).
"""

import subprocess
from pathlib import Path
from typing import List

import pytest


def _capture_calls(monkeypatch: pytest.MonkeyPatch, expected_cwd: Path) -> List[List[str]]:
    recorded: List[List[str]] = []

    def fake_run(cmd, check, cwd=None):  # pragma: no cover - trivial
        recorded.append(cmd)
        assert check is True
        assert cwd == str(expected_cwd)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return recorded


def test_prepare_data_invokes_both_commands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from emotion_experiment_engine import swebench_data_prep

    swebench_root = tmp_path / "SWE-bench"
    swebench_root.mkdir()

    cache_root = tmp_path / "cache"
    calls = _capture_calls(monkeypatch, swebench_root)

    swebench_data_prep.prepare_data(
        dataset_name="SWE-bench/SWE-bench_Lite",
        swebench_root=swebench_root,
        cache_root=cache_root,
        python_executable="python",
    )

    # Directories created
    assert (cache_root / "retrieval_results").is_dir()
    assert (cache_root / "datasets").is_dir()

    # Ensure expected commands invoked
    assert len(calls) == 2
    retrieval_cmd, text_cmd = calls

    assert retrieval_cmd[:4] == [
        "python",
        "-m",
        "swebench.inference.make_datasets.bm25_retrieval",
        "--dataset_name_or_path",
    ]
    assert "--output_dir" in retrieval_cmd

    assert text_cmd[:4] == [
        "python",
        "-m",
        "swebench.inference.make_datasets.create_text_dataset",
        "--dataset_name_or_path",
    ]
    assert "--retrieval_file" in text_cmd


def test_prepare_data_skip_retrieval(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from emotion_experiment_engine import swebench_data_prep

    swebench_root = tmp_path / "SWE-bench"
    swebench_root.mkdir()

    cache_root = tmp_path / "cache"
    retrieval_dir = cache_root / "retrieval_results"
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    existing_file = retrieval_dir / "SWE-bench_SWE-bench_Lite.retrieval.jsonl"
    existing_file.write_text("{}\n")

    calls = _capture_calls(monkeypatch, swebench_root)

    swebench_data_prep.prepare_data(
        dataset_name="SWE-bench/SWE-bench_Lite",
        swebench_root=swebench_root,
        cache_root=cache_root,
        python_executable="python",
        skip_retrieval=True,
    )

    assert len(calls) == 1
    text_cmd = calls[0]
    assert "--retrieval_file" in text_cmd
    # Uses existing retrieval artifacts rather than creating new ones
    assert existing_file.as_posix() in text_cmd


def test_prepare_data_dry_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from emotion_experiment_engine import swebench_data_prep

    swebench_root = tmp_path / "SWE-bench"
    swebench_root.mkdir()

    cache_root = tmp_path / "cache"

    calls = _capture_calls(monkeypatch, swebench_root)

    plan = swebench_data_prep.prepare_data(
        dataset_name="SWE-bench/SWE-bench_Lite",
        swebench_root=swebench_root,
        cache_root=cache_root,
        python_executable="python",
        dry_run=True,
    )

    assert calls == []
    assert isinstance(plan, list)
    assert all(isinstance(cmd, list) for cmd in plan)
    assert len(plan) == 2
