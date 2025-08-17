"""
Unit tests for benchmark adapters.
"""
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import json

from ..benchmark_adapters import InfiniteBenchAdapter, LoCoMoAdapter, get_adapter
from ..data_models import BenchmarkConfig, BenchmarkItem
from .test_utils import (
    create_mock_passkey_data, create_mock_locomo_data, 
    create_temp_data_file, cleanup_temp_files
)


class TestInfiniteBenchAdapter(unittest.TestCase):
    """Test InfiniteBench adapter functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_files = []
    
    def tearDown(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                temp_file.unlink()
            except FileNotFoundError:
                pass
    
    def test_load_jsonl_data(self):
        """Test loading JSONL format data"""
        # Create test data
        test_data = create_mock_passkey_data(3)
        temp_file = create_temp_data_file(test_data, 'jsonl')
        self.temp_files.append(temp_file)
        
        config = BenchmarkConfig(
            name="infinitebench",
            data_path=temp_file,
            task_type="passkey",
            evaluation_method="get_score_one_passkey"
        )
        
        adapter = InfiniteBenchAdapter(config)
        items = adapter.load_data()
        
        self.assertEqual(len(items), 3)
        self.assertIsInstance(items[0], BenchmarkItem)
        self.assertEqual(items[0].id, 0)
        self.assertIn("passkey", items[0].input_text.lower())
    
    def test_load_json_data(self):
        """Test loading JSON format data"""
        test_data = create_mock_passkey_data(2)
        temp_file = create_temp_data_file(test_data, 'json')
        self.temp_files.append(temp_file)
        
        config = BenchmarkConfig(
            name="infinitebench",
            data_path=temp_file,
            task_type="passkey",
            evaluation_method="get_score_one_passkey"
        )
        
        adapter = InfiniteBenchAdapter(config)
        items = adapter.load_data()
        
        self.assertEqual(len(items), 2)
        self.assertIsInstance(items[0], BenchmarkItem)
    
    def test_sample_limit(self):
        """Test sample limit functionality"""
        test_data = create_mock_passkey_data(10)
        temp_file = create_temp_data_file(test_data, 'jsonl')
        self.temp_files.append(temp_file)
        
        config = BenchmarkConfig(
            name="infinitebench",
            data_path=temp_file,
            task_type="passkey",
            evaluation_method="get_score_one_passkey",
            sample_limit=3
        )
        
        adapter = InfiniteBenchAdapter(config)
        items = adapter.load_data()
        
        self.assertEqual(len(items), 3)
    
    def test_create_prompt_with_context(self):
        """Test prompt creation with separate context"""
        item = BenchmarkItem(
            id="test",
            input_text="What is the passkey?",
            context="Some long context with hidden information",
            ground_truth="12345"
        )
        
        config = BenchmarkConfig(
            name="infinitebench",
            data_path=Path("dummy"),
            task_type="passkey",
            evaluation_method="test"
        )
        
        adapter = InfiniteBenchAdapter(config)
        prompt = adapter.create_prompt(item)
        
        self.assertIn("Some long context", prompt)
        self.assertIn("What is the passkey?", prompt)
        self.assertIn("Question:", prompt)
        self.assertIn("Answer:", prompt)
    
    def test_create_prompt_without_context(self):
        """Test prompt creation without separate context"""
        item = BenchmarkItem(
            id="test",
            input_text="What is the answer to life?",
            ground_truth="42"
        )
        
        config = BenchmarkConfig(
            name="infinitebench",
            data_path=Path("dummy"),
            task_type="qa",
            evaluation_method="test"
        )
        
        adapter = InfiniteBenchAdapter(config)
        prompt = adapter.create_prompt(item)
        
        self.assertIn("What is the answer to life?", prompt)
        self.assertIn("Answer:", prompt)
        self.assertNotIn("Question:", prompt)
    
    @patch('emotion_memory_experiments.benchmark_adapters.get_score_one')
    def test_evaluate_response_with_infinitebench(self, mock_get_score_one):
        """Test evaluation using InfiniteBench functions"""
        mock_get_score_one.return_value = 0.85
        
        config = BenchmarkConfig(
            name="infinitebench",
            data_path=Path("dummy"),
            task_type="passkey",
            evaluation_method="get_score_one_passkey"
        )
        
        adapter = InfiniteBenchAdapter(config)
        score = adapter.evaluate_response("12345", "12345", "passkey")
        
        mock_get_score_one.assert_called_once_with("12345", "12345", "passkey", "emotion_model")
        self.assertEqual(score, 0.85)
    
    def test_evaluate_response_fallback(self):
        """Test fallback evaluation when InfiniteBench functions unavailable"""
        config = BenchmarkConfig(
            name="infinitebench",
            data_path=Path("dummy"),
            task_type="passkey",
            evaluation_method="test"
        )
        
        adapter = InfiniteBenchAdapter(config)
        
        # Mock the get_score_one to be None (import failed)
        with patch('emotion_memory_experiments.benchmark_adapters.get_score_one', None):
            # Exact match
            score = adapter.evaluate_response("12345", "12345", "passkey")
            self.assertEqual(score, 1.0)
            
            # No match
            score = adapter.evaluate_response("54321", "12345", "passkey")
            self.assertEqual(score, 0.0)
    
    def test_file_not_found(self):
        """Test error handling for missing data file"""
        config = BenchmarkConfig(
            name="infinitebench",
            data_path=Path("nonexistent_file.jsonl"),
            task_type="passkey",
            evaluation_method="test"
        )
        
        adapter = InfiniteBenchAdapter(config)
        
        with self.assertRaises(FileNotFoundError):
            adapter.load_data()


class TestLoCoMoAdapter(unittest.TestCase):
    """Test LoCoMo adapter functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_files = []
    
    def tearDown(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                temp_file.unlink()
            except FileNotFoundError:
                pass
    
    def test_load_locomo_data(self):
        """Test loading LoCoMo conversation data"""
        test_data = create_mock_locomo_data(2)
        temp_file = create_temp_data_file(test_data, 'json')
        self.temp_files.append(temp_file)
        
        config = BenchmarkConfig(
            name="locomo",
            data_path=temp_file,
            task_type="conversational_qa",
            evaluation_method="custom"
        )
        
        adapter = LoCoMoAdapter(config)
        items = adapter.load_data()
        
        # Each sample has 2 QA pairs
        self.assertEqual(len(items), 4)
        self.assertIsInstance(items[0], BenchmarkItem)
        self.assertIn("Alice_0", items[0].context)
    
    def test_format_conversation(self):
        """Test conversation formatting"""
        conversation = {
            'speaker_a': 'Alice',
            'speaker_b': 'Bob',
            'session_1': [
                {'speaker': 'Alice', 'text': 'Hello Bob'},
                {'speaker': 'Bob', 'text': 'Hi Alice'}
            ],
            'session_1_date_time': '2024-01-01 10:00:00',
            'session_2': [
                {'speaker': 'Alice', 'text': 'How are you?'},
                {'speaker': 'Bob', 'text': 'I am fine'}
            ],
            'session_2_date_time': '2024-01-01 11:00:00'
        }
        
        config = BenchmarkConfig(
            name="locomo",
            data_path=Path("dummy"),
            task_type="conversational_qa",
            evaluation_method="test"
        )
        
        adapter = LoCoMoAdapter(config)
        formatted = adapter._format_conversation(conversation)
        
        self.assertIn("SESSION_1", formatted)
        self.assertIn("SESSION_2", formatted)
        self.assertIn("2024-01-01 10:00:00", formatted)
        self.assertIn("Alice: Hello Bob", formatted)
        self.assertIn("Bob: Hi Alice", formatted)
    
    def test_create_prompt(self):
        """Test prompt creation for LoCoMo"""
        item = BenchmarkItem(
            id="test",
            input_text="What did Alice say?",
            context="Alice: Hello\nBob: Hi",
            ground_truth="Hello"
        )
        
        config = BenchmarkConfig(
            name="locomo",
            data_path=Path("dummy"),
            task_type="conversational_qa",
            evaluation_method="test"
        )
        
        adapter = LoCoMoAdapter(config)
        prompt = adapter.create_prompt(item)
        
        self.assertIn("Alice: Hello", prompt)
        self.assertIn("What did Alice say?", prompt)
        self.assertIn("Question:", prompt)
        self.assertIn("Answer:", prompt)
    
    def test_evaluate_response_token_overlap(self):
        """Test LoCoMo evaluation using token overlap"""
        config = BenchmarkConfig(
            name="locomo",
            data_path=Path("dummy"),
            task_type="conversational_qa",
            evaluation_method="test"
        )
        
        adapter = LoCoMoAdapter(config)
        
        # Perfect overlap
        score = adapter.evaluate_response("hello world", "hello world", "conversational_qa")
        self.assertEqual(score, 1.0)
        
        # Partial overlap
        score = adapter.evaluate_response("hello", "hello world", "conversational_qa")
        self.assertEqual(score, 0.5)
        
        # No overlap
        score = adapter.evaluate_response("goodbye", "hello world", "conversational_qa")
        self.assertEqual(score, 0.0)
        
        # Empty ground truth
        score = adapter.evaluate_response("anything", "", "conversational_qa")
        self.assertEqual(score, 0.0)


class TestAdapterFactory(unittest.TestCase):
    """Test adapter factory function"""
    
    def test_get_infinitebench_adapter(self):
        """Test getting InfiniteBench adapter"""
        config = BenchmarkConfig(
            name="infinitebench",
            data_path=Path("test.jsonl"),
            task_type="passkey",
            evaluation_method="test"
        )
        
        adapter = get_adapter(config)
        self.assertIsInstance(adapter, InfiniteBenchAdapter)
    
    def test_get_locomo_adapter(self):
        """Test getting LoCoMo adapter"""
        config = BenchmarkConfig(
            name="locomo",
            data_path=Path("test.json"),
            task_type="conversational_qa",
            evaluation_method="test"
        )
        
        adapter = get_adapter(config)
        self.assertIsInstance(adapter, LoCoMoAdapter)
    
    def test_unknown_benchmark_error(self):
        """Test error for unknown benchmark"""
        config = BenchmarkConfig(
            name="unknown_benchmark",
            data_path=Path("test.json"),
            task_type="unknown",
            evaluation_method="test"
        )
        
        with self.assertRaises(ValueError) as context:
            get_adapter(config)
        
        self.assertIn("Unknown benchmark", str(context.exception))
    
    def test_case_insensitive_names(self):
        """Test case-insensitive benchmark names"""
        config = BenchmarkConfig(
            name="INFINITEBENCH",
            data_path=Path("test.jsonl"),
            task_type="passkey",
            evaluation_method="test"
        )
        
        adapter = get_adapter(config)
        self.assertIsInstance(adapter, InfiniteBenchAdapter)
        
        config.name = "LoCo_Mo"
        adapter = get_adapter(config)
        self.assertIsInstance(adapter, LoCoMoAdapter)


if __name__ == '__main__':
    unittest.main()