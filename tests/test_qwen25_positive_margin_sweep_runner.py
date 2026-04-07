"""
tests/test_qwen25_positive_margin_sweep_runner.py
Purpose: Verify the standalone scripts/qwen25_positive_margin_table self-report sweep runner accepts the stimulus reader override and emits model-specific output roots.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "qwen25_positive_margin_table"
    / "run_selfreport_qwen25_sweep.py"
)


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestQwen25PositiveMarginSweepRunner(unittest.TestCase):
    def test_parse_csv_arg_returns_trimmed_typed_values(self):
        module = _load_module(MODULE_PATH, "qwen25_positive_margin_sweep_runner")

        values = module.parse_csv_arg(" 1 , 2 , 4 ", cast=int)

        self.assertEqual(values, [1, 2, 4])

    def test_parse_csv_arg_returns_none_for_missing_value(self):
        module = _load_module(MODULE_PATH, "qwen25_positive_margin_sweep_runner")

        self.assertIsNone(module.parse_csv_arg(None, cast=float))

    def test_resolve_output_root_uses_model_default_when_cli_override_missing(self):
        module = _load_module(MODULE_PATH, "qwen25_positive_margin_sweep_runner")

        root = module.resolve_output_root(
            model_path="/home/jjl7137/huggingface_models/Qwen/Qwen2.5-3B-Instruct",
            cli_output_root=None,
        )

        self.assertEqual(
            root,
            module.default_output_root_for_model(
                "/home/jjl7137/huggingface_models/Qwen/Qwen2.5-3B-Instruct"
            ),
        )

    def test_base_config_accepts_stimulus_data_dir_override(self):
        module = _load_module(MODULE_PATH, "qwen25_positive_margin_sweep_runner")

        config = module._base_config(
            "/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct",
            stimulus_data_dir="data/stimulus/crowd-enVent_textlike_disentangled_v16_family_constrained_search",
        )

        self.assertEqual(
            config["repe_eng_config"]["data_dir"],
            "data/stimulus/crowd-enVent_textlike_disentangled_v16_family_constrained_search",
        )

    def test_default_output_root_for_model_uses_qwen25_layout(self):
        module = _load_module(MODULE_PATH, "qwen25_positive_margin_sweep_runner")

        root = module.default_output_root_for_model(
            "/home/jjl7137/huggingface_models/Qwen/Qwen2.5-1.5B-Instruct",
            results_root=Path("/tmp/qwen25_results"),
        )

        self.assertEqual(
            root,
            Path("/tmp/qwen25_results") / "self_report_logprob_multimodel" / "qwen2p5-1p5b-instruct",
        )


if __name__ == "__main__":
    unittest.main()
