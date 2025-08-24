#!/usr/bin/env python3
"""
Test file for: Unified SmartDataset class (TDD Red phase)
Purpose: Test the unified SmartMemoryBenchmarkDataset that combines data loading and evaluation

These tests define the behavior of the SmartDataset class that replaces
the separate adapter + dataset pattern with a unified approach.
"""

import unittest
import tempfile
import json
from pathlib import Path
from torch.utils.data import DataLoader

# Import the new modules we want to create (these don't exist yet - RED PHASE)  
from emotion_memory_experiments.smart_datasets import SmartMemoryBenchmarkDataset
from emotion_memory_experiments.data_models import BenchmarkConfig, BenchmarkItem


class TestSmartDatasetClass(unittest.TestCase):
    """Test unified SmartDataset class functionality"""
    
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
    
    def _create_test_infinitebench_data(self) -> Path:
        """Create test InfiniteBench data"""
        test_data = [
            {
                "id": 0,
                "input": "What is the passkey? Context: hidden 12345 text",
                "answer": "12345",
                "task_name": "passkey"
            },
            {
                "id": 1, 
                "input": "Find the key: secret 67890 data",
                "answer": "67890",
                "task_name": "passkey"
            }
        ]
        
        temp_file = Path(tempfile.mktemp(suffix='.jsonl'))
        with open(temp_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        
        self.temp_files.append(temp_file)
        return temp_file
    
    def test_smart_dataset_inherits_from_pytorch_dataset(self):
        """Test that SmartDataset is a proper PyTorch Dataset"""
        temp_file = self._create_test_infinitebench_data()
        
        config = BenchmarkConfig(
            name="infinitebench",
            task_type="passkey",
            data_path=temp_file
        )
        
        # Direct instantiation (not through factory)
        dataset = SmartMemoryBenchmarkDataset(config=config)
        
        # Should implement PyTorch Dataset interface
        self.assertTrue(hasattr(dataset, '__len__'))
        self.assertTrue(hasattr(dataset, '__getitem__'))
        
        # Should work with DataLoader using custom collate function
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)
        batches = list(dataloader)
        self.assertEqual(len(batches), 1)  # 2 items, batch_size=2
    
    def test_smart_dataset_loads_data_automatically(self):
        """Test that SmartDataset loads data in __init__"""
        temp_file = self._create_test_infinitebench_data()
        
        config = BenchmarkConfig(
            name="infinitebench", 
            task_type="passkey",
            data_path=temp_file
        )
        
        dataset = SmartMemoryBenchmarkDataset(config=config)
        
        # Data should be loaded automatically
        self.assertEqual(len(dataset), 2)
        
        # Items should be BenchmarkItem objects
        item_dict = dataset[0]
        self.assertIn('item', item_dict)
        self.assertIsInstance(item_dict['item'], BenchmarkItem)
    
    def test_smart_dataset_benchmark_specific_evaluation(self):
        """Test that SmartDataset provides benchmark-specific evaluation"""
        temp_file = self._create_test_infinitebench_data()
        
        config = BenchmarkConfig(
            name="infinitebench",
            task_type="passkey", 
            data_path=temp_file
        )
        
        dataset = SmartMemoryBenchmarkDataset(config=config)
        
        # Should have evaluation methods
        self.assertTrue(hasattr(dataset, 'evaluate_response'))
        
        # InfiniteBench evaluation should work
        score = dataset.evaluate_response("12345", "12345", "passkey")
        self.assertEqual(score, 1.0)
    
    def test_smart_dataset_with_prompt_wrapper(self):
        """Test SmartDataset with custom prompt wrapper"""
        temp_file = self._create_test_infinitebench_data()
        
        config = BenchmarkConfig(
            name="infinitebench",
            task_type="passkey",
            data_path=temp_file
        )
        
        def custom_wrapper(context, question):
            return f"WRAPPER: {context}\nASK: {question}\nANSWER:"
        
        dataset = SmartMemoryBenchmarkDataset(
            config=config, 
            prompt_wrapper=custom_wrapper
        )
        
        item = dataset[0]
        self.assertIn("WRAPPER:", item['prompt'])
        self.assertIn("ASK:", item['prompt'])
        self.assertIn("ANSWER:", item['prompt'])
    
    def test_smart_dataset_collate_function(self):
        """Test SmartDataset provides collate function for DataLoader"""
        temp_file = self._create_test_infinitebench_data()
        
        config = BenchmarkConfig(
            name="infinitebench",
            task_type="passkey",
            data_path=temp_file  
        )
        
        dataset = SmartMemoryBenchmarkDataset(config=config)
        
        # Should have collate function
        self.assertTrue(hasattr(dataset, 'collate_fn'))
        
        # Test collate function works
        batch_items = [dataset[0], dataset[1]]
        collated = dataset.collate_fn(batch_items)
        
        self.assertIn('prompts', collated)
        self.assertIn('items', collated)
        self.assertIn('ground_truths', collated)
        self.assertEqual(len(collated['prompts']), 2)
    
    def test_smart_dataset_benchmark_specific_parsing(self):
        """Test that different benchmarks parse data correctly"""
        # Create LoCoMo test data
        locomo_data = [{
            "sample_id": "test_sample",
            "conversation": {
                "session_1": [
                    {"speaker": "Alice", "text": "My budget is $50000"},
                    {"speaker": "Bob", "text": "Good to know"}
                ],
                "session_1_date_time": "2024-01-01 10:00:00"
            },
            "qa": [{
                "question": "What is Alice's budget?",
                "answer": "$50000", 
                "category": "budget"
            }]
        }]
        
        temp_file = Path(tempfile.mktemp(suffix='.json'))
        with open(temp_file, 'w') as f:
            json.dump(locomo_data, f)
        self.temp_files.append(temp_file)
        
        config = BenchmarkConfig(
            name="locomo",
            task_type="conversational_qa",
            data_path=temp_file
        )
        
        dataset = SmartMemoryBenchmarkDataset(config=config)
        
        # Should parse LoCoMo data correctly
        self.assertEqual(len(dataset), 1)
        item = dataset[0]
        benchmark_item = item['item']
        
        self.assertIn("Alice's budget", benchmark_item.input_text)
        self.assertIn("Alice: My budget is $50000", benchmark_item.context)
        self.assertEqual(benchmark_item.ground_truth, "$50000")
    
    def test_smart_dataset_detailed_metrics(self):
        """Test SmartDataset provides detailed evaluation metrics"""
        temp_file = self._create_test_infinitebench_data()
        
        config = BenchmarkConfig(
            name="infinitebench",
            task_type="passkey",
            data_path=temp_file
        )
        
        dataset = SmartMemoryBenchmarkDataset(config=config)
        
        # Should have detailed metrics method
        self.assertTrue(hasattr(dataset, 'evaluate_with_detailed_metrics'))
        
        metrics = dataset.evaluate_with_detailed_metrics("12345", "12345", "passkey")
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('overall_score', metrics)
        self.assertEqual(metrics['overall_score'], 1.0)
    
    def test_smart_dataset_task_metrics_info(self):
        """Test SmartDataset provides task metrics information"""
        temp_file = self._create_test_infinitebench_data()
        
        config = BenchmarkConfig(
            name="infinitebench",
            task_type="passkey", 
            data_path=temp_file
        )
        
        dataset = SmartMemoryBenchmarkDataset(config=config)
        
        # Should provide task metrics info
        self.assertTrue(hasattr(dataset, 'get_task_metrics'))
        
        metrics = dataset.get_task_metrics("passkey")
        self.assertIsInstance(metrics, list)
        self.assertIn("exact_match", metrics)
    
    def test_smart_dataset_batch_evaluation(self):
        """Test SmartDataset can evaluate batches efficiently"""
        temp_file = self._create_test_infinitebench_data()
        
        config = BenchmarkConfig(
            name="infinitebench",
            task_type="passkey",
            data_path=temp_file
        )
        
        dataset = SmartMemoryBenchmarkDataset(config=config)
        
        # Should support batch evaluation
        self.assertTrue(hasattr(dataset, 'evaluate_batch'))
        
        responses = ["12345", "wrong"]
        ground_truths = ["12345", "67890"]
        task_names = ["passkey", "passkey"]
        
        scores = dataset.evaluate_batch(responses, ground_truths, task_names)
        
        self.assertEqual(len(scores), 2)
        self.assertEqual(scores[0], 1.0)  # Correct
        self.assertEqual(scores[1], 0.0)  # Wrong


if __name__ == '__main__':
    unittest.main()