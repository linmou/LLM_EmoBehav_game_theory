#!/usr/bin/env python3
"""
Test file for: Direct dataset creation from BenchmarkConfig (TDD Red phase)
Purpose: Test the new direct dataset creation approach that eliminates the adapter layer

These tests define the desired behavior for direct dataset creation from configuration,
replacing the current adapter-based approach with a more efficient direct approach.
"""

import unittest
import tempfile
import json
from pathlib import Path
from typing import Any, Dict, List

# Import the new modules we want to create (these don't exist yet - RED PHASE)
from emotion_memory_experiments.smart_datasets import create_dataset_from_config, SmartMemoryBenchmarkDataset
from emotion_memory_experiments.data_models import BenchmarkConfig, BenchmarkItem


class TestDirectDatasetCreation(unittest.TestCase):
    """Test direct dataset creation from BenchmarkConfig without adapters"""
    
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
    
    def _create_temp_infinitebench_data(self, count: int = 3) -> Path:
        """Create temporary InfiniteBench test data"""
        test_data = []
        for i in range(count):
            test_data.append({
                "id": i,
                "input": f"What is the passkey? The passkey is hidden in this context: random text {12345 + i} more text",
                "answer": str(12345 + i),
                "task_name": "passkey"
            })
        
        temp_file = Path(tempfile.mktemp(suffix='.jsonl'))
        with open(temp_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        
        self.temp_files.append(temp_file)
        return temp_file
    
    def _create_temp_longbench_data(self, count: int = 2) -> Path:
        """Create temporary LongBench test data"""
        test_data = []
        for i in range(count):
            test_data.append({
                "id": f"longbook_qa_{i}",
                "input": f"Based on the book, what is the main theme? Book content: This is a story about {['friendship', 'love'][i % 2]}.",
                "answers": [f"{['friendship', 'love'][i % 2]} is the main theme"],
                "task_name": "longbook_qa_eng"
            })
        
        temp_file = Path(tempfile.mktemp(suffix='.json'))
        with open(temp_file, 'w') as f:
            json.dump(test_data, f)
        
        self.temp_files.append(temp_file)
        return temp_file
    
    def _create_temp_locomo_data(self, count: int = 2) -> Path:
        """Create temporary LoCoMo test data"""
        test_data = []
        for i in range(count):
            test_data.append({
                "sample_id": f"sample_{i}",
                "conversation": {
                    "session_1": [
                        {"speaker": "Alice", "text": f"Hello, my budget is ${30000 + i*1000}"},
                        {"speaker": "Bob", "text": "That's good to know"}
                    ],
                    "session_1_date_time": "2024-01-01 10:00:00"
                },
                "qa": [
                    {
                        "question": "What is Alice's budget?",
                        "answer": f"${30000 + i*1000}",
                        "category": "budget"
                    }
                ]
            })
        
        temp_file = Path(tempfile.mktemp(suffix='.json'))
        with open(temp_file, 'w') as f:
            json.dump(test_data, f)
        
        self.temp_files.append(temp_file)
        return temp_file
    
    def test_create_infinitebench_dataset_directly(self):
        """Test direct creation of InfiniteBench dataset from config"""
        # Create test data
        temp_file = self._create_temp_infinitebench_data(3)
        
        # Create config
        config = BenchmarkConfig(
            name="infinitebench",
            task_type="passkey",
            data_path=temp_file
        )
        
        # This should work directly without adapter
        dataset = create_dataset_from_config(config)
        
        # Verify dataset properties
        # After migration: should be specialized dataset class
        from emotion_memory_experiments.datasets.base import BaseBenchmarkDataset
        self.assertIsInstance(dataset, BaseBenchmarkDataset)
        self.assertEqual(len(dataset), 3)
        
        # Test dataset item structure
        item = dataset[0]
        self.assertIn('item', item)
        self.assertIn('prompt', item)
        self.assertIn('ground_truth', item)
        
        benchmark_item = item['item']
        self.assertIsInstance(benchmark_item, BenchmarkItem)
        self.assertEqual(benchmark_item.id, 0)
        self.assertIn("passkey", benchmark_item.input_text)
        self.assertEqual(benchmark_item.ground_truth, "12345")
    
    def test_create_longbench_dataset_directly(self):
        """Test direct creation of LongBench dataset from config"""
        temp_file = self._create_temp_longbench_data(2)
        
        config = BenchmarkConfig(
            name="longbench", 
            task_type="longbook_qa_eng",
            data_path=temp_file
        )
        
        dataset = create_dataset_from_config(config)
        
        # After migration: should be specialized dataset class
        from emotion_memory_experiments.datasets.base import BaseBenchmarkDataset
        self.assertIsInstance(dataset, BaseBenchmarkDataset)
        self.assertEqual(len(dataset), 2)
        
        item = dataset[0]
        benchmark_item = item['item']
        self.assertEqual(benchmark_item.id, "longbook_qa_0")
        self.assertIn("friendship", str(benchmark_item.ground_truth))
    
    def test_create_locomo_dataset_directly(self):
        """Test direct creation of LoCoMo dataset from config"""
        temp_file = self._create_temp_locomo_data(1)
        
        config = BenchmarkConfig(
            name="locomo",
            task_type="conversational_qa", 
            data_path=temp_file
        )
        
        dataset = create_dataset_from_config(config)
        
        # After migration: should be specialized dataset class
        from emotion_memory_experiments.datasets.base import BaseBenchmarkDataset
        self.assertIsInstance(dataset, BaseBenchmarkDataset)
        self.assertEqual(len(dataset), 1)  # 1 sample with 1 QA pair
        
        item = dataset[0]
        benchmark_item = item['item']
        self.assertIn("Alice's budget", benchmark_item.input_text)
        self.assertEqual(benchmark_item.ground_truth, "$30000")
    
    def test_direct_dataset_with_sample_limit(self):
        """Test sample limiting works with direct creation"""
        temp_file = self._create_temp_infinitebench_data(10)
        
        config = BenchmarkConfig(
            name="infinitebench",
            task_type="passkey",
            data_path=temp_file,
            sample_limit=3
        )
        
        dataset = create_dataset_from_config(config)
        
        self.assertEqual(len(dataset), 3)
    
    def test_direct_dataset_evaluation_capability(self):
        """Test that datasets can evaluate responses directly"""
        temp_file = self._create_temp_infinitebench_data(1)
        
        config = BenchmarkConfig(
            name="infinitebench",
            task_type="passkey", 
            data_path=temp_file
        )
        
        dataset = create_dataset_from_config(config)
        
        # Test evaluation method exists
        self.assertTrue(hasattr(dataset, 'evaluate_response'))
        
        # Test exact match evaluation
        score = dataset.evaluate_response("12345", "12345", "passkey")
        self.assertEqual(score, 1.0)
        
        # Test no match
        score = dataset.evaluate_response("wrong", "12345", "passkey") 
        self.assertEqual(score, 0.0)
    
    def test_direct_dataset_with_prompt_wrapper(self):
        """Test direct dataset creation with prompt wrapper"""
        temp_file = self._create_temp_infinitebench_data(1)
        
        config = BenchmarkConfig(
            name="infinitebench",
            task_type="passkey",
            data_path=temp_file
        )
        
        def test_prompt_wrapper(context, question):
            return f"CUSTOM: {context}\nQ: {question}\nA:"
        
        dataset = create_dataset_from_config(config, prompt_wrapper=test_prompt_wrapper)
        
        item = dataset[0]
        self.assertIn("CUSTOM:", item['prompt'])
        self.assertIn("Q:", item['prompt'])
        self.assertIn("A:", item['prompt'])
    
    def test_unknown_benchmark_error(self):
        """Test error for unknown benchmark type"""
        temp_file = self._create_temp_infinitebench_data(1)
        
        config = BenchmarkConfig(
            name="unknown_benchmark",
            task_type="unknown_task",
            data_path=temp_file
        )
        
        with self.assertRaises(ValueError) as context:
            create_dataset_from_config(config)
        
        self.assertIn("Unknown benchmark", str(context.exception))
    
    def test_file_not_found_error(self):
        """Test error handling for missing data file"""
        config = BenchmarkConfig(
            name="infinitebench",
            task_type="passkey",
            data_path=Path("nonexistent_file.jsonl")
        )
        
        with self.assertRaises(FileNotFoundError):
            create_dataset_from_config(config)


if __name__ == '__main__':
    unittest.main()