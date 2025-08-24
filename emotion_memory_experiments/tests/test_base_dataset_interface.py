#!/usr/bin/env python3
"""
Test file for: BaseBenchmarkDataset abstract interface (TDD Red phase)
Purpose: Define the abstract base class interface and ensure it enforces required methods

This test suite establishes the contract for the abstract base class that all
specialized dataset implementations must follow.
"""

import unittest
import tempfile
import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, MagicMock

import pytest
from torch.utils.data import DataLoader

# These imports will fail initially (Red phase) - that's expected!
try:
    from emotion_memory_experiments.datasets.base import BaseBenchmarkDataset
except ImportError:
    BaseBenchmarkDataset = None

from emotion_memory_experiments.data_models import BenchmarkConfig, BenchmarkItem


class TestBaseBenchmarkDatasetInterface(unittest.TestCase):
    """Test abstract base class interface and contract enforcement"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = BenchmarkConfig(
            name="test_benchmark",
            task_type="test_task",
            data_path=Path("test_data.json")
        )
        
        # Create a simple mock tokenizer
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        self.mock_tokenizer.decode.return_value = "decoded text"
    
    def test_base_dataset_is_abstract(self):
        """Test that BaseBenchmarkDataset cannot be instantiated directly"""
        if BaseBenchmarkDataset is None:
            self.skipTest("BaseBenchmarkDataset not implemented yet (Red phase)")
        
        with pytest.raises(TypeError):
            # Should raise TypeError because abstract methods not implemented
            BaseBenchmarkDataset(config=self.mock_config)
    
    def test_base_dataset_requires_abstract_methods(self):
        """Test that subclasses must implement abstract methods"""
        if BaseBenchmarkDataset is None:
            self.skipTest("BaseBenchmarkDataset not implemented yet (Red phase)")
        
        class IncompleteDataset(BaseBenchmarkDataset):
            """Dataset missing abstract methods - should fail to instantiate"""
            pass
        
        with pytest.raises(TypeError) as exc_info:
            IncompleteDataset(config=self.mock_config)
        
        # Should mention which abstract methods are missing
        error_msg = str(exc_info.value)
        expected_methods = ["_load_and_parse_data", "evaluate_response", "get_task_metrics"]
        for method in expected_methods:
            self.assertIn(method, error_msg)
    
    def test_base_dataset_with_all_abstract_methods(self):
        """Test that complete implementation can be instantiated"""
        if BaseBenchmarkDataset is None:
            self.skipTest("BaseBenchmarkDataset not implemented yet (Red phase)")
        
        class CompleteDataset(BaseBenchmarkDataset):
            """Complete dataset implementation for testing"""
            
            def _load_and_parse_data(self) -> List[BenchmarkItem]:
                return [
                    BenchmarkItem(
                        id="test_1",
                        input_text="Test question?",
                        context="Test context",
                        ground_truth="Test answer",
                        metadata={"test": True}
                    )
                ]
            
            def evaluate_response(self, response: str, ground_truth: Any, task_name: str) -> float:
                return 1.0 if response == ground_truth else 0.0
            
            def get_task_metrics(self, task_name: str) -> List[str]:
                return ["accuracy"]
        
        # Should be able to instantiate complete implementation
        dataset = CompleteDataset(config=self.mock_config)
        
        # Should have proper interface
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.config, self.mock_config)
        
        # Should be able to get items
        item = dataset[0]
        self.assertIn('item', item)
        self.assertIn('prompt', item)
        self.assertIn('ground_truth', item)
    
    def test_base_dataset_common_functionality_interface(self):
        """Test that base class provides expected common functionality"""
        if BaseBenchmarkDataset is None:
            self.skipTest("BaseBenchmarkDataset not implemented yet (Red phase)")
        
        class TestDataset(BaseBenchmarkDataset):
            def _load_and_parse_data(self):
                return [BenchmarkItem(id="1", input_text="Q", context="C", ground_truth="A")]
            def evaluate_response(self, response, ground_truth, task_name):
                return 1.0
            def get_task_metrics(self, task_name):
                return ["accuracy"]
        
        dataset = TestDataset(config=self.mock_config)
        
        # Test PyTorch Dataset interface
        self.assertTrue(hasattr(dataset, '__len__'))
        self.assertTrue(hasattr(dataset, '__getitem__'))
        
        # Test batch processing interface
        self.assertTrue(hasattr(dataset, 'collate_fn'))
        self.assertTrue(callable(dataset.collate_fn))
        
        # Test common functionality
        self.assertTrue(hasattr(dataset, '_apply_truncation'))
        self.assertTrue(callable(dataset._apply_truncation))
    
    def test_base_dataset_initialization_parameters(self):
        """Test that base class accepts expected initialization parameters"""
        if BaseBenchmarkDataset is None:
            self.skipTest("BaseBenchmarkDataset not implemented yet (Red phase)")
        
        class TestDataset(BaseBenchmarkDataset):
            def _load_and_parse_data(self):
                return [BenchmarkItem(id="1", input_text="Q", context="C", ground_truth="A")]
            def evaluate_response(self, response, ground_truth, task_name):
                return 1.0
            def get_task_metrics(self, task_name):
                return ["accuracy"]
        
        prompt_wrapper = lambda c, q: f"Context: {c} Question: {q}"
        
        dataset = TestDataset(
            config=self.mock_config,
            prompt_wrapper=prompt_wrapper,
            max_context_length=100,
            tokenizer=self.mock_tokenizer,
            truncation_strategy="right"
        )
        
        # Should store all parameters
        self.assertEqual(dataset.config, self.mock_config)
        self.assertEqual(dataset.prompt_wrapper, prompt_wrapper)
        self.assertEqual(dataset.max_context_length, 100)
        self.assertEqual(dataset.tokenizer, self.mock_tokenizer)
        self.assertEqual(dataset.truncation_strategy, "right")
    
    def test_truncation_functionality(self):
        """Test context truncation functionality"""
        if BaseBenchmarkDataset is None:
            self.skipTest("BaseBenchmarkDataset not implemented yet (Red phase)")
        
        class TestDataset(BaseBenchmarkDataset):
            def _load_and_parse_data(self):
                return [BenchmarkItem(
                    id="1", 
                    input_text="Q", 
                    context="This is a very long context that should be truncated",
                    ground_truth="A"
                )]
            def evaluate_response(self, response, ground_truth, task_name):
                return 1.0
            def get_task_metrics(self, task_name):
                return ["accuracy"]
        
        dataset = TestDataset(
            config=self.mock_config,
            max_context_length=3,  # Limit to 3 tokens
            tokenizer=self.mock_tokenizer,
            truncation_strategy="right"
        )
        
        long_item = BenchmarkItem(
            id="test",
            context="This is a very long context that should be truncated",
            input_text="Question",
            ground_truth="Answer"
        )
        
        truncated_items = dataset._apply_truncation([long_item])
        
        # Should return truncated version
        self.assertEqual(len(truncated_items), 1)
        self.assertIsInstance(truncated_items[0], BenchmarkItem)
        
        # Should have truncation metadata
        self.assertIn('truncation_info', truncated_items[0].metadata)
        self.assertTrue(truncated_items[0].metadata['truncation_info']['was_truncated'])
    
    def test_collate_function_interface(self):
        """Test collate function produces expected batch format"""
        if BaseBenchmarkDataset is None:
            self.skipTest("BaseBenchmarkDataset not implemented yet (Red phase)")
        
        class TestDataset(BaseBenchmarkDataset):
            def _load_and_parse_data(self):
                return [
                    BenchmarkItem(id="1", input_text="Q1", context="C1", ground_truth="A1"),
                    BenchmarkItem(id="2", input_text="Q2", context="C2", ground_truth="A2")
                ]
            def evaluate_response(self, response, ground_truth, task_name):
                return 1.0
            def get_task_metrics(self, task_name):
                return ["accuracy"]
        
        dataset = TestDataset(config=self.mock_config)
        
        # Create mock batch items (what __getitem__ would return)
        batch_items = [
            {'prompt': 'Prompt 1', 'item': dataset.items[0], 'ground_truth': 'A1'},
            {'prompt': 'Prompt 2', 'item': dataset.items[1], 'ground_truth': 'A2'}
        ]
        
        batch = dataset.collate_fn(batch_items)
        
        # Should have expected batch format
        self.assertIn('prompts', batch)
        self.assertIn('items', batch)
        self.assertIn('ground_truths', batch)
        
        self.assertEqual(len(batch['prompts']), 2)
        self.assertEqual(len(batch['items']), 2)
        self.assertEqual(len(batch['ground_truths']), 2)
        
        self.assertEqual(batch['prompts'], ['Prompt 1', 'Prompt 2'])
        self.assertEqual(batch['ground_truths'], ['A1', 'A2'])
    
    def test_dataloader_compatibility(self):
        """Test that dataset works with PyTorch DataLoader"""
        if BaseBenchmarkDataset is None:
            self.skipTest("BaseBenchmarkDataset not implemented yet (Red phase)")
        
        class TestDataset(BaseBenchmarkDataset):
            def _load_and_parse_data(self):
                return [
                    BenchmarkItem(id=str(i), input_text=f"Q{i}", context=f"C{i}", ground_truth=f"A{i}")
                    for i in range(5)
                ]
            def evaluate_response(self, response, ground_truth, task_name):
                return 1.0
            def get_task_metrics(self, task_name):
                return ["accuracy"]
        
        dataset = TestDataset(config=self.mock_config)
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)
        
        batches = list(dataloader)
        
        # Should create proper batches
        self.assertEqual(len(batches), 3)  # 5 items, batch_size=2 -> 3 batches
        
        # First batch should have 2 items
        self.assertEqual(len(batches[0]['prompts']), 2)
        self.assertEqual(len(batches[0]['items']), 2)
        self.assertEqual(len(batches[0]['ground_truths']), 2)
        
        # Last batch should have 1 item
        self.assertEqual(len(batches[2]['prompts']), 1)


if __name__ == '__main__':
    unittest.main()