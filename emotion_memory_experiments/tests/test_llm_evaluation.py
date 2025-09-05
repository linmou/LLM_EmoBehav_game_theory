"""
Test file for: LLM-based evaluation system (TDD Red phase)
Purpose: Test async LLM evaluation functions with mocked OpenAI API calls

This test suite defines the expected behavior for the new LLM evaluation system.
These tests will initially FAIL and drive the implementation.
"""

import unittest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import tempfile
import json
from pathlib import Path

from ..evaluation_utils import llm_evaluate_response, llm_evaluate_batch
from ..datasets.longbench import LongBenchDataset
from ..data_models import BenchmarkConfig


class TestLLMEvaluation(unittest.TestCase):
    """Test LLM evaluation functions with mocked OpenAI API"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_files = []

    def tearDown(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            if temp_file.exists():
                temp_file.unlink()

    @patch("emotion_memory_experiments.evaluation_utils.openai.AsyncOpenAI")
    def test_llm_evaluate_passkey_correct(self, mock_openai_class):
        """Test LLM evaluation for passkey extraction - should return 1.0 for correct"""
        # Setup mock
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        # Mock API response indicating correct
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "CORRECT"
        mock_client.chat.completions.create.return_value = mock_response

        # Test async function
        async def run_test():
            score = await llm_evaluate_response("The passkey is 42", "42", "passkey")
            return score

        result = asyncio.run(run_test())

        self.assertEqual(result, 1.0, "Should return 1.0 for correct passkey response")

        # Verify API was called with proper prompt
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        self.assertIn("passkey is 42", call_args[1]["messages"][0]["content"])
        self.assertIn("42", call_args[1]["messages"][0]["content"])

    @patch("emotion_memory_experiments.evaluation_utils.openai.AsyncOpenAI")
    def test_llm_evaluate_qa_incorrect(self, mock_openai_class):
        """Test LLM evaluation for QA - should return 0.0 for incorrect"""
        # Setup mock
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        # Mock API response indicating incorrect
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "INCORRECT"
        mock_client.chat.completions.create.return_value = mock_response

        # Test async function
        async def run_test():
            score = await llm_evaluate_response("I don't know", "Tokyo", "qa")
            return score

        result = asyncio.run(run_test())

        self.assertEqual(result, 0.0, "Should return 0.0 for incorrect QA response")

    @patch("emotion_memory_experiments.evaluation_utils.openai.AsyncOpenAI")
    def test_llm_evaluate_multiple_choice_correct(self, mock_openai_class):
        """Test LLM evaluation for multiple choice - should return 1.0 for correct"""
        # Setup mock
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        # Mock API response indicating correct
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "CORRECT"
        mock_client.chat.completions.create.return_value = mock_response

        # Test async function
        async def run_test():
            score = await llm_evaluate_response("The answer is B", ["B"], "choice")
            return score

        result = asyncio.run(run_test())

        self.assertEqual(result, 1.0, "Should return 1.0 for correct multiple choice")

    @patch("emotion_memory_experiments.evaluation_utils.openai.AsyncOpenAI")
    def test_llm_evaluate_api_failure(self, mock_openai_class):
        """Test LLM evaluation handles API failures by raising exception"""
        # Setup mock to raise exception
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        # Test that exception is raised (no silent failures)
        async def run_test():
            with self.assertRaises(RuntimeError) as cm:
                await llm_evaluate_response("test", "expected", "task")
            self.assertIn("LLM evaluation failed", str(cm.exception))

        asyncio.run(run_test())

    @patch("emotion_memory_experiments.evaluation_utils.llm_evaluate_response")
    def test_llm_evaluate_batch_success(self, mock_single_eval):
        """Test batch evaluation with multiple responses"""

        # Setup mock to return different scores
        async def mock_eval(response, ground_truth, task_name):
            if "correct" in response.lower():
                return 1.0
            else:
                return 0.0

        mock_single_eval.side_effect = mock_eval

        # Test batch evaluation
        async def run_test():
            responses = ["This is correct", "This is wrong", "Another correct answer"]
            ground_truths = ["expected1", "expected2", "expected3"]
            task_names = ["task1", "task1", "task1"]

            scores = await llm_evaluate_batch(responses, ground_truths, task_names)
            return scores

        result = asyncio.run(run_test())

        self.assertEqual(len(result), 3, "Should return scores for all 3 responses")
        self.assertEqual(result[0], 1.0, "First response should be correct")
        self.assertEqual(result[1], 0.0, "Second response should be incorrect")
        self.assertEqual(result[2], 1.0, "Third response should be correct")


class TestDatasetBatchEvaluation(unittest.TestCase):
    """Test dataset batch evaluation integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_files = []

    def tearDown(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            if temp_file.exists():
                temp_file.unlink()

    def _create_test_dataset(self):
        """Helper to create a test dataset"""
        # Create temporary data file
        temp_file = Path(tempfile.mktemp(suffix=".jsonl"))
        test_data = [
            {
                "id": "test_1",
                "context": "test context",
                "input": "test input",
                "answer": "test_answer",
            }
        ]

        with open(temp_file, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        self.temp_files.append(temp_file)

        # Create dataset config
        config = BenchmarkConfig(
            name="test_benchmark",
            task_type="test_task",
            data_path=temp_file,
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
        )

        return LongBenchDataset(config)

    @patch("emotion_memory_experiments.evaluation_utils.llm_evaluate_response")
    def test_dataset_evaluate_batch(self, mock_llm_eval):
        """Test dataset batch evaluation method"""

        # Setup mock
        async def mock_eval(response, ground_truth, task_name):
            return 1.0 if response == "correct" else 0.0

        mock_llm_eval.side_effect = mock_eval

        # Create dataset
        dataset = self._create_test_dataset()

        # Test batch evaluation (note: evaluate_batch is sync, but calls async internally)
        responses = ["correct", "wrong", "correct"]
        ground_truths = ["ans1", "ans2", "ans3"]
        task_names = ["task1", "task1", "task1"]

        result = dataset.evaluate_batch(responses, ground_truths, task_names)

        self.assertEqual(len(result), 3, "Should return 3 scores")
        self.assertEqual(result[0], 1.0, "First should be correct")
        self.assertEqual(result[1], 0.0, "Second should be incorrect")
        self.assertEqual(result[2], 1.0, "Third should be correct")


if __name__ == "__main__":
    unittest.main()
