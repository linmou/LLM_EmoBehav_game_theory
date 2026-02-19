"""
Test responsibility:
- emotion_experiment_engine/datasets/emotion_check.py
- emotion_experiment_engine/memory_prompt_wrapper.py

Purpose:
- lock behavior for the new emotion_scale task based on subjective sentences
- ensure Gemini evaluation config is forwarded without dropping client routing
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from emotion_experiment_engine.benchmark_component_registry import (
    create_benchmark_components,
)
from emotion_experiment_engine.benchmark_prompt_wrapper import get_benchmark_prompt_wrapper
from emotion_experiment_engine.data_models import BenchmarkConfig
from emotion_experiment_engine.datasets.emotion_check import EmotionCheckDataset


class DummyPromptFormat:
    """Minimal PromptFormat stub used to inspect wrapper output."""

    def __init__(self) -> None:
        self.last_system_prompt = ""
        self.last_user_messages = []

    def build(self, system_prompt, user_messages, assistant_messages=None, enable_thinking=False):
        del assistant_messages, enable_thinking
        self.last_system_prompt = system_prompt
        self.last_user_messages = list(user_messages)
        return f"SYSTEM:{system_prompt}\nUSER:{self.last_user_messages[0]}"


class TestEmotionScaleTask(unittest.TestCase):
    def _write_subjective_data(self, path: Path) -> None:
        records = [
            {"id": 0, "sentence": "I feel better after talking with close friends."},
            {"id": 1, "sentence": "Waiting for uncertain news keeps me up at night."},
        ]
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _build_config(self, data_path: Path, llm_eval_config=None) -> BenchmarkConfig:
        return BenchmarkConfig(
            name="emotion_check",
            task_type="emotion_scale",
            data_path=data_path,
            base_data_dir=str(data_path.parent),
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=llm_eval_config,
        )

    def test_emotion_scale_parses_subjective_sentences_and_adapts_ground_truth(self):
        """I am starting with a failing test. This is the Red phase."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "emotion_check_emotion_scale.jsonl"
            self._write_subjective_data(data_path)
            cfg = self._build_config(data_path)
            prompt_format = DummyPromptFormat()

            _, _, dataset = create_benchmark_components(
                benchmark_name=cfg.name,
                task_type=cfg.task_type,
                config=cfg,
                prompt_format=prompt_format,
                emotion="anger",
            )

            self.assertIsInstance(dataset, EmotionCheckDataset)
            self.assertEqual(len(dataset), 2)

            sample = dataset[0]
            self.assertEqual(sample["ground_truth"], "anger")
            self.assertIn("close friends", sample["item"].input_text)
            self.assertEqual(sample["item"].metadata["category"], "emotion_scale")

    def test_emotion_scale_prompt_is_open_ended(self):
        """I am starting with a failing test. This is the Red phase."""
        prompt_format = DummyPromptFormat()
        wrapper = get_benchmark_prompt_wrapper(
            "emotion_check", "emotion_scale", prompt_format
        )

        wrapper(
            context=None,
            question="I feel uneasy when plans change unexpectedly.",
            user_messages="Respond in one natural sentence.",
            enable_thinking=False,
            augmentation_config=None,
            answer=None,
            emotion="fear",
            options=None,
        )
        user_text = prompt_format.last_user_messages[0]
        self.assertIn("I feel uneasy when plans change unexpectedly.", user_text)
        self.assertNotIn("listed options", user_text.lower())
        self.assertNotIn("copy the chosen option text", user_text.lower())
        # emotion_scale must rely on steering vectors only, no explicit text hint
        self.assertNotIn("You currently feel", user_text)

    def test_emotion_scale_forwards_gemini_llm_eval_config(self):
        """I am starting with a failing test. This is the Red phase."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "emotion_check_emotion_scale.jsonl"
            self._write_subjective_data(data_path)
            llm_cfg = {
                "client": "gemini",
                "model": "gemini-2.5-flash",
                "temperature": 0.0,
                "evaluation_prompt": "Question: {question}\nResponse: {response}",
            }
            cfg = self._build_config(data_path, llm_eval_config=llm_cfg)
            prompt_format = DummyPromptFormat()
            _, _, dataset = create_benchmark_components(
                benchmark_name=cfg.name,
                task_type=cfg.task_type,
                config=cfg,
                prompt_format=prompt_format,
                emotion="anger",
            )

            with patch(
                "emotion_experiment_engine.evaluation_utils.llm_evaluate_response"
            ) as mocked_eval:
                mocked_eval.return_value = {"emotion": "anger", "confidence": 1.0}
                score = dataset.evaluate_response(
                    response="I am furious about this unfair decision.",
                    ground_truth="anger",
                    task_name="emotion_scale",
                    prompt="I feel better after talking with close friends.",
                )

            self.assertEqual(score, 1.0)
            forwarded_cfg = mocked_eval.call_args.kwargs["llm_eval_config"]
            self.assertEqual(forwarded_cfg["client"], "gemini")
            self.assertEqual(forwarded_cfg["model"], "gemini-2.5-flash")

    def test_emotion_scale_batch_keeps_eval_details_for_persistence(self):
        """I am starting with a failing test. This is the Red phase."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "emotion_check_emotion_scale.jsonl"
            self._write_subjective_data(data_path)
            llm_cfg = {
                "client": "gemini",
                "model": "gemini-2.5-flash",
                "temperature": 0.0,
                "evaluation_prompt": "Question: {question}\nResponse: {response}",
            }
            cfg = self._build_config(data_path, llm_eval_config=llm_cfg)
            prompt_format = DummyPromptFormat()
            _, _, dataset = create_benchmark_components(
                benchmark_name=cfg.name,
                task_type=cfg.task_type,
                config=cfg,
                prompt_format=prompt_format,
                emotion="anger",
            )

            with patch(
                "emotion_experiment_engine.evaluation_utils.llm_evaluate_response"
            ) as mocked_eval:
                mocked_eval.side_effect = [
                    {"emotion": "anger", "confidence": 0.92},
                    {"emotion": "neutral", "confidence": 0.71},
                ]
                scores = dataset.evaluate_batch(
                    responses=["I am furious about this.", "I feel calm now."],
                    ground_truths=["anger", "fear"],
                    task_names=["emotion_scale", "emotion_scale"],
                    prompts=["q1", "q2"],
                )

            self.assertEqual(scores[0], 0.92)
            self.assertEqual(scores[1], 0.0)
            details = getattr(dataset, "_last_eval_details", None)
            self.assertIsInstance(details, list)
            self.assertEqual(len(details), 2)
            self.assertIsNotNone(details[0])
            self.assertIsNotNone(details[1])
            self.assertEqual(details[0]["predicted_emotion"], "anger")
            self.assertEqual(details[0]["confidence"], 0.92)
            self.assertTrue(details[0]["matched_ground_truth"])
            self.assertEqual(details[0]["judge_client"], "gemini")
            self.assertEqual(details[0]["judge_model"], "gemini-2.5-flash")
            self.assertEqual(details[1]["predicted_emotion"], "neutral")
            self.assertFalse(details[1]["matched_ground_truth"])


if __name__ == "__main__":
    unittest.main()
