#!/usr/bin/env python3
"""
Test file responsible for testing evaluation functions with realistic benchmark data
Purpose: Validate that evaluation methods work correctly with real-world data formats
"""

import unittest
import json
import os
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from emotion_memory_experiments.smart_datasets import SmartMemoryBenchmarkDataset
from emotion_memory_experiments.data_models import BenchmarkConfig
from emotion_memory_experiments.evaluation_utils import (
    get_score_one, qa_f1_score, first_int_match
)


class TestRealisticDataValidation(unittest.TestCase):
    """Test evaluation methods with realistic benchmark data formats"""

    def setUp(self):
        """Set up test data directory"""
        self.test_data_dir = Path(__file__).parent / "data"
        self.test_data_dir.mkdir(exist_ok=True)

    def test_passkey_with_realistic_data(self):
        """Test passkey evaluation with real prediction formats"""
        test_file = self.test_data_dir / "realistic_passkey_test.jsonl"
        
        # Create dataset
        config = BenchmarkConfig(
            name="infinitebench",
            task_type="passkey",
            data_path=test_file
        )
        dataset = SmartMemoryBenchmarkDataset(config=config)
        
        # Test cases from realistic data
        test_cases = [
            ("The passkey is 12345", "12345", 1.0),  # Natural language response
            ("Based on the document, the passkey mentioned is 67890", "67890", 1.0),  # Verbose response
            ("I cannot find any passkey in the text", "11111", 0.0),  # Failed extraction
            ("The pass key is 99888.", "99888", 1.0),  # With punctuation
            ("Looking through the text, the passkey appears to be 54321", "54321", 1.0)  # Casual tone
        ]
        
        for prediction, ground_truth, expected_score in test_cases:
            with self.subTest(prediction=prediction[:50]):
                score = dataset.evaluate_response(prediction, ground_truth, "passkey")
                self.assertEqual(score, expected_score,
                    f"Passkey evaluation failed for: {prediction}")

    def test_math_with_realistic_data(self):
        """Test mathematical evaluation with realistic number extraction"""
        test_file = self.test_data_dir / "realistic_math_test.jsonl"
        
        # Create dataset
        config = BenchmarkConfig(
            name="infinitebench",
            task_type="math_find",
            data_path=test_file
        )
        dataset = SmartMemoryBenchmarkDataset(config=config)
        
        test_cases = [
            ("42", 42, 1.0),  # Simple number
            ("The answer is 73", 73, 1.0),  # With context
            ("After calculating: 156", 156, 1.0),  # With prefix
            ("I calculated the result as 89.5", 89.5, 1.0),  # Float
            ("The mathematical result is 203", 199, 0.0)  # Wrong answer
        ]
        
        for prediction, ground_truth, expected_score in test_cases:
            with self.subTest(prediction=prediction):
                score = dataset.evaluate_response(prediction, ground_truth, "math_find")
                self.assertEqual(score, expected_score,
                    f"Math evaluation failed for: {prediction} vs {ground_truth}")

    def test_qa_with_realistic_data(self):
        """Test QA evaluation with realistic F1 scoring"""
        test_file = self.test_data_dir / "realistic_qa_test.jsonl"
        
        # Create dataset
        config = BenchmarkConfig(
            name="longbench",
            task_type="longbook_qa_eng", 
            data_path=test_file
        )
        dataset = SmartMemoryBenchmarkDataset(config=config)
        
        test_cases = [
            ("The capital of France is Paris", "Paris is the capital city of France"),
            ("Einstein developed the theory of relativity", "Albert Einstein formulated the theory of relativity"),
            ("The Great Wall of China was built over many centuries", "The Great Wall of China was constructed over multiple centuries")
        ]
        
        for prediction, ground_truth in test_cases:
            with self.subTest(prediction=prediction[:30]):
                score = dataset.evaluate_response(prediction, ground_truth, "longbook_qa_eng")
                # Should have decent F1 score due to overlapping tokens
                self.assertGreater(score, 0.3, 
                    f"QA F1 score too low for: {prediction}")
                self.assertLessEqual(score, 1.0,
                    f"QA F1 score too high for: {prediction}")

    def test_locomo_with_realistic_data(self):
        """Test LoCoMo evaluation with realistic conversation data"""
        test_file = self.test_data_dir / "realistic_locomo_test.json"
        
        # Create dataset
        config = BenchmarkConfig(
            name="locomo",
            task_type="locomo",
            data_path=test_file
        )
        dataset = SmartMemoryBenchmarkDataset(config=config)
        
        test_cases = [
            ("John was a software engineer", "software engineer"),  # Exact match
            ("Sarah moved to New York in March 2020", "March 2020"),  # Substring match
            ("The conference had around 250 people", "250 attendees")  # Partial match
        ]
        
        for prediction, ground_truth in test_cases:
            with self.subTest(prediction=prediction[:30]):
                score = dataset.evaluate_response(prediction, ground_truth, "locomo")
                # Should have decent F1 score
                self.assertGreater(score, 0.2,
                    f"LoCoMo F1 score too low for: {prediction}")

    def test_file_loading_integration(self):
        """Test that datasets can properly load realistic test files"""
        # Test JSONL loading
        passkey_config = BenchmarkConfig(name="infinitebench", task_type="passkey", data_path=self.test_data_dir / "realistic_passkey_test.jsonl")
        passkey_dataset = SmartMemoryBenchmarkDataset(config=passkey_config)
        self.assertGreater(len(passkey_dataset), 0, "Failed to load JSONL passkey data")
        
        # Test JSON loading
        locomo_config = BenchmarkConfig(name="locomo", task_type="locomo", data_path=self.test_data_dir / "realistic_locomo_test.json")
        locomo_dataset = SmartMemoryBenchmarkDataset(config=locomo_config)
        self.assertGreater(len(locomo_dataset), 0, "Failed to load JSON LoCoMo data")

    def test_evaluation_routing_with_realistic_data(self):
        """Test that task-specific evaluation routing works with realistic data"""
        passkey_file = self.test_data_dir / "realistic_passkey_test.jsonl"
        math_file = self.test_data_dir / "realistic_math_test.jsonl"
        qa_file = self.test_data_dir / "realistic_qa_test.jsonl"
        
        # Test passkey routing
        passkey_config = BenchmarkConfig(name="infinitebench", task_type="passkey", data_path=passkey_file)
        passkey_dataset = SmartMemoryBenchmarkDataset(config=passkey_config)
        score = passkey_dataset.evaluate_response("The passkey is 12345", "12345", "passkey")
        self.assertEqual(score, 1.0, "Passkey routing failed")
        
        # Test math routing  
        math_config = BenchmarkConfig(name="infinitebench", task_type="math_find", data_path=math_file)
        math_dataset = SmartMemoryBenchmarkDataset(config=math_config)
        score = math_dataset.evaluate_response("42", 42, "math_find")
        self.assertEqual(score, 1.0, "Math routing failed")
        
        # Test QA routing
        qa_config = BenchmarkConfig(name="longbench", task_type="longbook_qa_eng", data_path=qa_file)
        qa_dataset = SmartMemoryBenchmarkDataset(config=qa_config)
        score = qa_dataset.evaluate_response("Paris", "Paris is the capital", "longbook_qa_eng")
        self.assertGreater(score, 0.0, "QA routing failed")

    def test_edge_cases_with_realistic_data(self):
        """Test edge cases that might occur with realistic data"""
        test_file = self.test_data_dir / "realistic_passkey_test.jsonl"
        config = BenchmarkConfig(name="infinitebench", task_type="passkey", data_path=test_file)
        dataset = SmartMemoryBenchmarkDataset(config=config)
        
        # Empty prediction
        score = dataset.evaluate_response("", "12345", "passkey")
        self.assertEqual(score, 0.0, "Empty prediction should score 0.0")
        
        # Multiple numbers in prediction (extracts first number, not target)
        score = dataset.evaluate_response("I see numbers 999 and 12345 in the text", "12345", "passkey")
        self.assertEqual(score, 0.0, "Should extract first number (999), which doesn't match target (12345)")
        
        # Test case where first number matches
        score = dataset.evaluate_response("I see numbers 12345 and 999 in the text", "12345", "passkey")
        self.assertEqual(score, 1.0, "Should extract first number correctly when it matches")


if __name__ == '__main__':
    unittest.main()