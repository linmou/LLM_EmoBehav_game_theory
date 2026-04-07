"""
tests/test_qwen25_positive_margin_table_builder.py
Purpose: Verify the standalone scripts/qwen25_positive_margin_table table builder aggregates Qwen2.5 saved summaries without depending on auto_experiments runner files.
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
    / "build_positive_margin_table.py"
)


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestQwen25PositiveMarginTableBuilder(unittest.TestCase):
    def test_qwen25_model_roots_include_legacy_and_multimodel_locations(self):
        module = _load_module(MODULE_PATH, "qwen25_positive_margin_table_builder")

        with tempfile.TemporaryDirectory() as tmp_dir:
            results_root = Path(tmp_dir)
            (results_root / "self_report_logprob").mkdir()
            multimodel = results_root / "self_report_logprob_multimodel"
            (multimodel / "qwen2p5-1p5b-instruct").mkdir(parents=True)
            (multimodel / "qwen2p5-3b-instruct").mkdir(parents=True)
            roots = module.qwen25_model_roots(results_root)

        self.assertEqual(
            roots,
            {
                "qwen2p5-0p5b-instruct": results_root / "self_report_logprob",
                "qwen2p5-1p5b-instruct": results_root / "self_report_logprob_multimodel" / "qwen2p5-1p5b-instruct",
                "qwen2p5-3b-instruct": results_root / "self_report_logprob_multimodel" / "qwen2p5-3b-instruct",
            },
        )

    def test_aggregate_positive_margin_table_counts_positive_layers(self):
        module = _load_module(MODULE_PATH, "qwen25_positive_margin_table_builder")

        with tempfile.TemporaryDirectory() as tmp_dir:
            results_root = Path(tmp_dir)
            model_05b = results_root / "self_report_logprob"
            model_15b = results_root / "self_report_logprob_multimodel" / "qwen2p5-1p5b-instruct"
            model_3b = results_root / "self_report_logprob_multimodel" / "qwen2p5-3b-instruct"
            model_05b.mkdir(parents=True)
            model_15b.mkdir(parents=True)
            model_3b.mkdir(parents=True)

            rows = [
                (model_05b, "anger", 1, 1.0, 0.10),
                (model_05b, "anger", 2, 1.0, -0.20),
                (model_15b, "anger", 1, 1.0, 0.30),
                (model_3b, "anger", 1, 1.0, 0.00),
                (model_3b, "anger", 2, 2.0, 0.40),
                (model_15b, "fear", 5, 2.0, 0.20),
            ]
            for model_root, emotion, layer_1based, intensity, margin in rows:
                run_dir = model_root / f"{emotion}_layer_{layer_1based}_intensity_{str(intensity).replace('.', 'p')}"
                run_dir.mkdir(parents=True)
                (run_dir / "target_option_softmax_by_steer.csv").write_text(
                    (
                        "steer_emotion,delta_p_target_vs_top_p_non_target_mean,layer_1based,intensity\n"
                        f"{emotion},{margin},{layer_1based},{intensity}\n"
                    ),
                    encoding="utf-8",
                )
                (run_dir / "run_metadata.json").write_text(
                    (
                        "{"
                        f"\"steer_emotion\":\"{emotion}\","
                        f"\"layer_1based\":{layer_1based},"
                        f"\"intensity\":{float(intensity)}"
                        "}"
                    ),
                    encoding="utf-8",
                )

            table = module.aggregate_positive_margin_table(results_root)

        anger_1 = table[(table["emotion"] == "anger") & (table["intensity"] == 1.0)].iloc[0]
        self.assertEqual(int(anger_1["positive_layers"]), 2)
        self.assertEqual(int(anger_1["total_layers"]), 4)
        self.assertAlmostEqual(float(anger_1["positive_fraction"]), 0.5)
        self.assertEqual(str(anger_1["display"]), "2/4 (0.50)")


if __name__ == "__main__":
    unittest.main()
