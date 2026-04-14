"""
tests/test_qwen25_positive_margin_repro.py
Purpose: Verify the lightweight one-command Qwen2.5 positive-margin reproduction wrapper builds the intended step sequence and shell interface.
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "qwen25_positive_margin_table"
    / "run_qwen25_positive_margin_table.py"
)

SHELL_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "qwen25_positive_margin_table"
    / "run_qwen25_positive_margin_table.sh"
)


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestQwen25PositiveMarginRepro(unittest.TestCase):
    def test_build_steps_table_only_runs_reporter_only(self):
        module = _load_module(MODULE_PATH, "qwen25_positive_margin_repro")

        steps = module.build_steps(mode="table-only")

        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].name, "positive_margin_table")
        self.assertIn("scripts/qwen25_positive_margin_table/build_positive_margin_table.py", " ".join(steps[0].argv))

    def test_default_mode_is_rerun(self):
        module = _load_module(MODULE_PATH, "qwen25_positive_margin_repro")

        parser = module.build_parser()
        args = parser.parse_args([])

        self.assertEqual(args.mode, "rerun")

    def test_build_steps_rerun_runs_three_model_sweeps_then_report(self):
        module = _load_module(MODULE_PATH, "qwen25_positive_margin_repro")

        steps = module.build_steps(
            mode="rerun",
            stimulus_data_dir="data/stimulus/crowd-enVent_textlike_disentangled_v16_family_constrained_search",
            results_root=Path("/tmp/qwen25_v16_results"),
        )

        self.assertEqual(
            [step.name for step in steps],
            [
                "selfreport_sweep_qwen2p5-0p5b-instruct",
                "selfreport_sweep_qwen2p5-1p5b-instruct",
                "selfreport_sweep_qwen2p5-3b-instruct",
                "positive_margin_table",
            ],
        )
        self.assertIn("scripts/qwen25_positive_margin_table/run_selfreport_qwen25_sweep.py", " ".join(steps[0].argv))
        self.assertIn("--stimulus-data-dir", steps[0].argv)
        self.assertIn(
            "data/stimulus/crowd-enVent_textlike_disentangled_v16_family_constrained_search",
            steps[0].argv,
        )
        self.assertIn("--model-path", steps[0].argv)
        self.assertIn("Qwen2.5-0.5B-Instruct", " ".join(steps[0].argv))
        self.assertIn("--output-root", steps[0].argv)
        self.assertIn("self_report_logprob", " ".join(steps[0].argv))

        self.assertIn("Qwen2.5-1.5B-Instruct", " ".join(steps[1].argv))
        self.assertIn("Qwen2.5-3B-Instruct", " ".join(steps[2].argv))
        self.assertIn("scripts/qwen25_positive_margin_table/build_positive_margin_table.py", " ".join(steps[3].argv))
        self.assertIn("/tmp/qwen25_v16_results", " ".join(steps[3].argv))

    def test_run_step_executes_configured_command(self):
        module = _load_module(MODULE_PATH, "qwen25_positive_margin_repro")

        with tempfile.TemporaryDirectory() as tmp_dir:
            sentinel = Path(tmp_dir) / "sentinel.txt"
            step = module.Step(
                name="sentinel",
                argv=["/bin/sh", "-c", f"printf 'ok' > '{sentinel}'"],
            )

            module.run_step(step)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "ok")

    def test_shell_wrapper_targets_python_orchestrator(self):
        shell_text = SHELL_PATH.read_text(encoding="utf-8")

        self.assertTrue(shell_text.startswith("#!/usr/bin/env bash"))
        self.assertIn("Purpose: reproduce the Qwen2.5 positive-margin table", shell_text)
        self.assertIn("run_qwen25_positive_margin_table.py", shell_text)


if __name__ == "__main__":
    unittest.main()
