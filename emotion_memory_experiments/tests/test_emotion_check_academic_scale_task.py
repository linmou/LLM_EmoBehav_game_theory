"""
Tests for registering and loading the new task (emotion_check, academic_scale).

This file verifies two things:
1) The benchmark component registry recognizes (emotion_check, academic_scale)
   and returns the correct dataset and answer wrapper wiring.
2) The EmotionCheckDataset can parse the academic scales JSONL schema located at
   data/emotion_scales/emotion_check_academic_scales.jsonl and produce items
   whose ground truth is adapted by the EmotionAnswerWrapper to the active emotion.
"""

import unittest
from pathlib import Path

from emotion_memory_experiments.benchmark_component_registry import (
    create_benchmark_components,
)
from emotion_memory_experiments.data_models import BenchmarkConfig
from emotion_memory_experiments.datasets.emotion_check import EmotionCheckDataset


class DummyPromptFormat:
    """Minimal prompt format with a build() API used by wrappers in tests."""

    def build(self, system_text: str, user_messages, enable_thinking: bool = False) -> str:
        if isinstance(user_messages, list):
            user = "\n".join(user_messages)
        else:
            user = str(user_messages)
        return f"{system_text}\n{user}"


class TestEmotionCheckAcademicScaleTask(unittest.TestCase):
    """Validate registration and loading of (emotion_check, academic_scale)."""

    def setUp(self):
        self.data_file = Path(
            "data/emotion_scales/emotion_check_academic_scales.jsonl"
        )
        # Safety: ensure the test data file exists in repo
        if not self.data_file.exists():
            self.skipTest(f"Missing test data file: {self.data_file}")

    def test_create_components_with_academic_scale(self):
        """I am starting with a failing test. This is the Red phase.

        Registry should accept (emotion_check, academic_scale), create an
        EmotionCheckDataset wired with an EmotionAnswerWrapper partial that
        adapts ground truth to the active emotion.
        """
        cfg = BenchmarkConfig(
            name="emotion_check",
            task_type="academic_scale",
            data_path=self.data_file,
            base_data_dir=str(self.data_file.parent),
            sample_limit=3,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None,
        )

        prompt_format = DummyPromptFormat()

        # Should not raise; returns (prompt_wrapper_partial, answer_wrapper_partial, dataset)
        prompt_wrap, answer_wrap, dataset = create_benchmark_components(
            benchmark_name=cfg.name,
            task_type=cfg.task_type,
            config=cfg,
            prompt_format=prompt_format,
            emotion="anger",
        )

        # Dataset type
        self.assertIsInstance(dataset, EmotionCheckDataset)

        # Basic dataset sanity: has items
        self.assertGreater(len(dataset), 0)

        # __getitem__ should return composite ground_truth for academic_scale
        sample = dataset[0]
        self.assertIn("prompt", sample)
        self.assertIn("ground_truth", sample)
        gt = sample["ground_truth"]
        # Composite dict with active emotion (from wrapper) and target emotion (from item)
        self.assertIsInstance(gt, dict)
        self.assertIn("active", gt)
        self.assertIn("target", gt)
        # Active should be the activated emotion passed above
        self.assertEqual(gt["active"], "anger")
        # Target should equal the item's original ground truth (target emotion from file)
        self.assertEqual(gt["target"], sample["item"].ground_truth)

    def test_evaluate_response_scoring(self):
        """Check per-item scoring for academic_scale is directionally correct."""
        cfg = BenchmarkConfig(
            name="emotion_check",
            task_type="academic_scale",
            data_path=self.data_file,
            base_data_dir=str(self.data_file.parent),
            sample_limit=1,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None,
        )

        prompt_format = DummyPromptFormat()
        _, _, dataset = create_benchmark_components(
            benchmark_name=cfg.name,
            task_type=cfg.task_type,
            config=cfg,
            prompt_format=prompt_format,
            emotion="anger",
        )

        sample = dataset[0]
        gt = sample["ground_truth"]  # dict with active/target
        prompt = sample["prompt"]

        # If target==active and rating is high, score ~ 1; low -> ~ 0
        s_high = dataset.evaluate_response("5", gt, "academic_scale", prompt)
        s_low = dataset.evaluate_response("1", gt, "academic_scale", prompt)
        self.assertGreater(s_high, 0.8)
        self.assertLess(s_low, 0.2)

        # If we switch active to a different emotion, high rating should reduce score
        gt_mismatch = {**gt, "active": "happiness" if gt.get("target") != "happiness" else "fear"}
        s_mismatch = dataset.evaluate_response("5", gt_mismatch, "academic_scale", prompt)
        self.assertLess(s_mismatch, 0.3)


if __name__ == "__main__":
    unittest.main()
