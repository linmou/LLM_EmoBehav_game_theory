"""
Test file for testing comprehensive task and evaluation methods based on InfiniteBench and LongBench.
This file validates the enhanced benchmark adapters functionality.
"""

import unittest
from pathlib import Path
from typing import Any, Dict, List

import sys
sys.path.append(str(Path(__file__).parent.parent))

from adapters import InfiniteBenchAdapter, LongBenchAdapter
from data_models import BenchmarkConfig, BenchmarkItem


class TestComprehensiveEvaluation(unittest.TestCase):
    """Test comprehensive evaluation methods for both InfiniteBench and LongBench"""

    def setUp(self):
        """Set up test configurations and mock data"""
        self.infinitebench_config = BenchmarkConfig(
            name="infinitebench",
            data_path=Path("test_data.json"),  # Mock path
            task_type="passkey",
            evaluation_method="exact_match"
        )
        
        self.longbench_config = BenchmarkConfig(
            name="longbench", 
            data_path=Path("test_data.json"),  # Mock path
            task_type="narrativeqa",
            evaluation_method="f1_score"
        )

    def test_infinitebench_task_routing(self):
        """Test InfiniteBench task-specific evaluation routing"""
        adapter = InfiniteBenchAdapter(self.infinitebench_config)
        
        # Test task complexity classification
        self.assertEqual(adapter.get_evaluation_complexity("passkey"), "simple")
        self.assertEqual(adapter.get_evaluation_complexity("longbook_qa_eng"), "complex")
        
        # Test metrics mapping
        passkey_metrics = adapter.get_task_metrics("passkey")
        self.assertIn("exact_match", passkey_metrics)
        
        qa_metrics = adapter.get_task_metrics("longbook_qa_eng")
        self.assertIn("f1_score", qa_metrics)
        self.assertIn("precision", qa_metrics)
        self.assertIn("recall", qa_metrics)

    def test_longbench_task_routing(self):
        """Test LongBench task-specific evaluation routing"""
        adapter = LongBenchAdapter(self.longbench_config)
        
        # Test task complexity classification
        self.assertEqual(adapter.get_evaluation_complexity("trec"), "simple")
        self.assertEqual(adapter.get_evaluation_complexity("narrativeqa"), "complex")
        
        # Test metrics mapping
        trec_metrics = adapter.get_task_metrics("trec")
        self.assertIn("accuracy", trec_metrics)
        
        qa_metrics = adapter.get_task_metrics("narrativeqa")
        self.assertIn("f1_score", qa_metrics)

    def test_infinitebench_passkey_evaluation(self):
        """Test InfiniteBench passkey evaluation method"""
        adapter = InfiniteBenchAdapter(self.infinitebench_config)
        
        # Test exact match
        score = adapter.evaluate_response("12345", "12345", "passkey")
        self.assertEqual(score, 1.0)
        
        # Test mismatch
        score = adapter.evaluate_response("54321", "12345", "passkey") 
        self.assertEqual(score, 0.0)
        
        # Test extraction from longer response
        score = adapter.evaluate_response("The answer is 12345 and that's correct.", "12345", "passkey")
        self.assertEqual(score, 1.0)

    def test_infinitebench_kv_retrieval_evaluation(self):
        """Test InfiniteBench KV retrieval evaluation method"""
        adapter = InfiniteBenchAdapter(self.infinitebench_config)
        
        # Test exact match
        score = adapter.evaluate_response("apple", "apple", "kv_retrieval")
        self.assertEqual(score, 1.0)
        
        # Test word in response
        score = adapter.evaluate_response("The fruit is apple.", "apple", "kv_retrieval")
        self.assertEqual(score, 1.0)
        
        # Test mismatch
        score = adapter.evaluate_response("banana", "apple", "kv_retrieval")
        self.assertEqual(score, 0.0)

    def test_longbench_qa_f1_evaluation(self):
        """Test LongBench QA F1 evaluation method"""
        adapter = LongBenchAdapter(self.longbench_config)
        
        # Test perfect match
        score = adapter.evaluate_response("The capital of France is Paris", ["The capital of France is Paris"], "narrativeqa")
        self.assertEqual(score, 1.0)
        
        # Test partial match should give partial F1
        score = adapter.evaluate_response("Paris", ["The capital of France is Paris"], "narrativeqa")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)
        
        # Test no match
        score = adapter.evaluate_response("London", ["The capital of France is Paris"], "narrativeqa")
        self.assertEqual(score, 0.0)

    def test_longbench_classification_evaluation(self):
        """Test LongBench classification evaluation method"""
        adapter = LongBenchAdapter(self.longbench_config)
        
        # Test exact classification match
        score = adapter.evaluate_response("positive", ["positive"], "trec")
        self.assertEqual(score, 1.0)
        
        # Test classification in longer response
        score = adapter.evaluate_response("This is a positive sentiment.", ["positive"], "trec")
        self.assertEqual(score, 1.0)
        
        # Test mismatch
        score = adapter.evaluate_response("negative", ["positive"], "trec")
        self.assertEqual(score, 0.0)

    def test_detailed_metrics_infinitebench(self):
        """Test detailed metrics for InfiniteBench tasks"""
        adapter = InfiniteBenchAdapter(self.infinitebench_config)
        
        # Test detailed metrics for QA task
        metrics = adapter.evaluate_with_detailed_metrics(
            "Paris is the capital", ["Paris is the capital of France"], "longbook_qa_eng"
        )
        
        self.assertIn("f1_score", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("overall_score", metrics)
        
        # Test simple task returns basic metrics
        metrics = adapter.evaluate_with_detailed_metrics("12345", "12345", "passkey")
        self.assertIn("accuracy", metrics)
        self.assertIn("overall_score", metrics)

    def test_detailed_metrics_longbench(self):
        """Test detailed metrics for LongBench tasks"""
        adapter = LongBenchAdapter(self.longbench_config)
        
        # Test detailed metrics for QA task
        metrics = adapter.evaluate_with_detailed_metrics(
            "Paris", ["The capital of France is Paris"], "narrativeqa"
        )
        
        self.assertIn("f1_score", metrics)
        self.assertIn("precision", metrics) 
        self.assertIn("recall", metrics)
        self.assertIn("overall_score", metrics)

    def test_batch_evaluation(self):
        """Test batch evaluation for both adapters"""
        infinitebench_adapter = InfiniteBenchAdapter(self.infinitebench_config)
        longbench_adapter = LongBenchAdapter(self.longbench_config)
        
        # Test InfiniteBench batch evaluation
        responses = ["12345", "54321", "67890"]
        ground_truths = ["12345", "12345", "67890"]
        task_names = ["passkey", "passkey", "passkey"]
        
        scores = infinitebench_adapter.evaluate_batch(responses, ground_truths, task_names)
        self.assertEqual(len(scores), 3)
        self.assertEqual(scores[0], 1.0)  # Match
        self.assertEqual(scores[1], 0.0)  # Mismatch
        self.assertEqual(scores[2], 1.0)  # Match
        
        # Test LongBench batch evaluation
        responses = ["positive", "negative", "neutral"]
        ground_truths = [["positive"], ["negative"], ["neutral"]]
        task_names = ["trec", "trec", "trec"]
        
        scores = longbench_adapter.evaluate_batch(responses, ground_truths, task_names)
        self.assertEqual(len(scores), 3)
        self.assertEqual(scores[0], 1.0)
        self.assertEqual(scores[1], 1.0)
        self.assertEqual(scores[2], 1.0)

    def test_length_based_evaluation(self):
        """Test length-based evaluation (LongBench-E style)"""
        adapter = LongBenchAdapter(self.longbench_config)
        
        responses = ["answer1", "answer2", "answer3", "answer4"]
        ground_truths = [["answer1"], ["answer2"], ["answer3"], ["answer4"]]
        task_names = ["trec", "trec", "trec", "trec"]
        lengths = [2000, 5000, 10000, 15000]  # Different length categories
        
        scores_by_length = adapter.evaluate_by_length(responses, ground_truths, task_names, lengths)
        
        self.assertIn("0-4k", scores_by_length)
        self.assertIn("4-8k", scores_by_length)
        self.assertIn("8k+", scores_by_length)
        
        # All should be 100% since answers match exactly
        self.assertEqual(scores_by_length["0-4k"], 100.0)
        self.assertEqual(scores_by_length["4-8k"], 100.0)
        self.assertEqual(scores_by_length["8k+"], 100.0)


if __name__ == "__main__":
    unittest.main()