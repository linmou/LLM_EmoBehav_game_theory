"""
Tests for result_analysis/humaneval_significance.py
Purpose: verify paired t vs neutral on HumanEval detailed_results.csv and run discovery.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest


def _write_detailed_csv(p: Path, rows: list[dict[str, str]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "emotion",
                "intensity",
                "item_id",
                "task_name",
                "response",
                "ground_truth",
                "score",
                "benchmark",
                "repeat_id",
                "error",
            ],
        )
        w.writeheader()
        w.writerows(rows)


def test_paired_t_vs_neutral_significant(tmp_path: Path) -> None:
    # Build a simple run where neutral succeeds and emotion fails consistently.
    run_dir = tmp_path / "results" / "humaneval" / "ModelA_run"
    rows: list[dict[str, str]] = []
    for i in range(5):
        rows.append(
            {
                "emotion": "neutral",
                "intensity": "0.0",
                "item_id": f"HumanEval/{i}",
                "task_name": "main",
                "response": "",
                "ground_truth": "",
                "score": "1.0",
                "benchmark": "humaneval",
                "repeat_id": "0",
                "error": "",
            }
        )
        rows.append(
            {
                "emotion": "anger",
                "intensity": "1.5",
                "item_id": f"HumanEval/{i}",
                "task_name": "main",
                "response": "",
                "ground_truth": "",
                "score": "0.0",
                "benchmark": "humaneval",
                "repeat_id": "0",
                "error": "",
            }
        )
    _write_detailed_csv(run_dir / "detailed_results.csv", rows)

    # Import after writing file so discovery has something to read.
    from result_analysis.humaneval_significance import (
        paired_t_vs_neutral_from_detailed,
    )

    res = paired_t_vs_neutral_from_detailed(run_dir)
    assert "anger" in res
    d = res["anger"]
    assert d["n_pairs"] == 5
    # With zero variance and negative mean delta, t = -inf, which should be significant.
    assert d["significant"] is True
    assert d["t_stat"] < 0


def test_paired_t_vs_neutral_nodiff(tmp_path: Path) -> None:
    # Build a run where emotion equals neutral; should be non-significant with t=0.
    run_dir = tmp_path / "results" / "humaneval" / "ModelB_run"
    rows: list[dict[str, str]] = []
    for i in range(4):
        for emo in ("neutral", "happiness"):
            rows.append(
                {
                    "emotion": emo,
                    "intensity": "1.5" if emo != "neutral" else "0.0",
                    "item_id": f"HumanEval/{i}",
                    "task_name": "main",
                    "response": "",
                    "ground_truth": "",
                    "score": "1.0" if i % 2 == 0 else "0.0",
                    "benchmark": "humaneval",
                    "repeat_id": "0",
                    "error": "",
                }
            )
    _write_detailed_csv(run_dir / "detailed_results.csv", rows)

    from result_analysis.humaneval_significance import (
        paired_t_vs_neutral_from_detailed,
    )

    res = paired_t_vs_neutral_from_detailed(run_dir)
    d = res["happiness"]
    assert d["n_pairs"] == 4
    assert d["t_stat"] == 0.0
    assert d["significant"] is False


def test_discover_runs(tmp_path: Path) -> None:
    base = tmp_path / "results" / "humaneval"
    # Valid run
    _write_detailed_csv(
        (base / "ModelC_run" / "detailed_results.csv"),
        [
            {
                "emotion": "neutral",
                "intensity": "0.0",
                "item_id": "HumanEval/0",
                "task_name": "main",
                "response": "",
                "ground_truth": "",
                "score": "1.0",
                "benchmark": "humaneval",
                "repeat_id": "0",
                "error": "",
            },
            {
                "emotion": "anger",
                "intensity": "1.5",
                "item_id": "HumanEval/0",
                "task_name": "main",
                "response": "",
                "ground_truth": "",
                "score": "0.0",
                "benchmark": "humaneval",
                "repeat_id": "0",
                "error": "",
            },
        ],
    )
    # Non-run dir
    (base / "misc").mkdir(parents=True, exist_ok=True)

    from result_analysis.humaneval_significance import discover_humaneval_runs

    runs = discover_humaneval_runs(base)
    assert len(runs) == 1
    assert runs[0].model.startswith("ModelC")

