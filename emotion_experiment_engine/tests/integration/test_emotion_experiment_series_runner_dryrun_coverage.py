"""Tests for emotion_experiment_engine/emotion_experiment_series_runner.py dry-run coverage.

Purpose: Ensure --dry-run validates all benchmark entries, not just the first few.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import patch

import yaml

from emotion_experiment_engine.emotion_experiment_series_runner import (
    MemoryExperimentSeriesRunner,
)


class TestDryRunCoverage(unittest.TestCase):
    """Integration tests to ensure dry-run validates full benchmark list."""

    def test_dry_run_validates_all_benchmarks(self) -> None:
        # We craft 4 benchmarks; the last one should be reached during dry-run.
        with tempfile.TemporaryDirectory() as td:
            config: Dict[str, Any] = {
                "experiment_name": "dryrun_coverage",
                "models": [
                    "/fake/model/a",
                    "/fake/model/b",
                    "/fake/model/c",
                    "/fake/model/d",
                ],
                "emotions": ["anger"],
                "intensities": [1.0],
                "benchmarks": [
                    {"name": "game_theory_decision", "task_type": "A"},
                    {"name": "game_theory_decision", "task_type": "B"},
                    {"name": "game_theory_decision", "task_type": "C"},
                    {
                        "name": "game_theory_decision",
                        "task_type": "Ultimatum_Game_Responder",
                    },
                ],
                # Avoid writing into repo-level `results/` (which may be a symlink).
                "output_dir": str(Path(td) / "out"),
                "batch_size": 1,
            }
            config_path = Path(td) / "cfg.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            runner = MemoryExperimentSeriesRunner(str(config_path), dry_run=True)

            seen: List[str] = []

            def _fake_setup(benchmark_config: Dict[str, Any], model_name: str):
                seen.append(str(benchmark_config.get("task_type")))
                if benchmark_config.get("task_type") == "Ultimatum_Game_Responder":
                    raise RuntimeError("Ultimatum_Game_Responder")
                bench = SimpleNamespace(get_data_path=lambda: Path("/tmp/fake.json"))
                cfg = SimpleNamespace(output_dir="/tmp/out", benchmark=bench)
                return SimpleNamespace(config=cfg, emotion_datasets={})

            with patch.object(runner, "setup_experiment", side_effect=_fake_setup):
                with self.assertRaisesRegex(RuntimeError, "Ultimatum_Game_Responder"):
                    runner.dry_run_series()

            # The point: dry-run must reach the last benchmark.
            self.assertIn("Ultimatum_Game_Responder", seen)
