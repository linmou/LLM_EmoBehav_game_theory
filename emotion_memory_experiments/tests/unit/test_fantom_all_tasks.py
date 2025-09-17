"""
Responsible files:
- scripts/convert_fantom.py (adds conversions for all FANToM tasks)
- emotion_memory_experiments/datasets/fantom.py (extend parsing/evaluation)

Purpose:
- Red phase: ensure additional FANToM task types load from data/fantom and
  basic evaluation semantics work.
"""

import unittest
from typing import Optional

from emotion_memory_experiments.benchmark_component_registry import (
    create_benchmark_components,
)
from emotion_memory_experiments.data_models import BenchmarkConfig


class TestFantomAllTasks(unittest.TestCase):
    def _cfg(self, task: str) -> BenchmarkConfig:
        cfg = BenchmarkConfig(
            name="fantom",
            task_type=task,
            data_path=None,
            base_data_dir="data/fantom",
            sample_limit=3,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None,
        )
        cfg.data_path = cfg.get_data_path()
        return cfg

    def _build(self, task: str):
        # Minimal prompt_format stub using Fantom wrapper contract indirectly
        class _PF:
            def build(self, system_prompt: str, user_messages, **kwargs):
                return f"{system_prompt}\n{user_messages}"

        cfg = self._cfg(task)
        return create_benchmark_components(
            benchmark_name=cfg.name,
            task_type=cfg.task_type,
            config=cfg,
            prompt_format=_PF(),
        )

    def test_answerability_list_inaccessible(self):
        _, _, ds = self._build("short_answerability_list_inaccessible")
        self.assertGreater(len(ds), 0)
        ex = ds[0]
        gt = ex["ground_truth"]
        self.assertIsInstance(gt, list)
        # JSON list answer should match
        resp = {"reational": "r", "answer": gt}
        self.assertEqual(ds.evaluate_response(str(resp), gt, "short_answerability_list_inaccessible", ex["prompt"]), 1.0)
        # Comma-separated case-insensitive order-insensitive
        s = ", ".join(gt[::-1])
        self.assertEqual(ds.evaluate_response(s, gt, "short_answerability_list_inaccessible", ex["prompt"]), 1.0)

    def test_infoaccessibility_binary_inaccessible(self):
        _, _, ds = self._build("short_infoaccessibility_binary_inaccessible")
        self.assertGreater(len(ds), 0)
        ex = ds[0]
        gt = ex["ground_truth"]
        self.assertIn(gt, ("yes", "no"))
        self.assertEqual(ds.evaluate_response(gt.upper(), gt, "short_infoaccessibility_binary_inaccessible", ex["prompt"]), 1.0)

    def test_infoaccessibility_list_inaccessible(self):
        _, _, ds = self._build("short_infoaccessibility_list_inaccessible")
        self.assertGreater(len(ds), 0)
        ex = ds[0]
        gt = ex["ground_truth"]
        self.assertIsInstance(gt, list)
        # Allow JSON list
        resp = {"reational": "r", "answer": gt}
        self.assertEqual(ds.evaluate_response(str(resp), gt, "short_infoaccessibility_list_inaccessible", ex["prompt"]), 1.0)

    def test_belief_choice_accessible(self):
        _, _, ds = self._build("short_belief_choice_accessible")
        self.assertGreater(len(ds), 0)
        ex = ds[0]
        opts = ex["item"].metadata.get("options")  # type: ignore[attr-defined]
        gt_text = ex["ground_truth"][0]
        self.assertIn(gt_text, opts)
        # Exact text
        self.assertEqual(ds.evaluate_response(gt_text, [gt_text], "short_belief_choice_accessible", ex["prompt"]), 1.0)

    def test_belief_gen_inaccessible(self):
        _, _, ds = self._build("short_belief_gen_inaccessible")
        self.assertGreater(len(ds), 0)
        ex = ds[0]
        gt = ex["ground_truth"]
        self.assertIsInstance(gt, str)
        # Exact normalized match
        self.assertEqual(ds.evaluate_response(gt, gt, "short_belief_gen_inaccessible", ex["prompt"]), 1.0)

    def test_fact(self):
        _, _, ds = self._build("short_fact")
        self.assertGreater(len(ds), 0)
        ex = ds[0]
        gt = ex["ground_truth"]
        self.assertIsInstance(gt, str)
        self.assertEqual(ds.evaluate_response(gt, gt, "short_fact", ex["prompt"]), 1.0)


if __name__ == "__main__":
    unittest.main()

