#!/usr/bin/env python3
"""
Test file for: Integration and Behavioral Equivalence Testing (TDD Phase 5)
Purpose: Ensure new specialized datasets produce identical results to original SmartDataset

This comprehensive test suite validates that the refactored specialized datasets
(InfiniteBenchDataset, LongBenchDataset, LoCoMoDataset) produce functionally 
equivalent results to the original SmartMemoryBenchmarkDataset implementation.
"""

import unittest
import tempfile
import json
from pathlib import Path
from typing import Any, Dict, List

# Import both old and new implementations for comparison
from emotion_memory_experiments.smart_datasets import SmartMemoryBenchmarkDataset
from emotion_memory_experiments.dataset_factory import create_dataset_from_config
from emotion_memory_experiments.data_models import BenchmarkConfig, BenchmarkItem


class TestIntegrationAndBehavioralEquivalence(unittest.TestCase):
    """Test behavioral equivalence between old and new dataset implementations"""
    
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
    
    def _create_infinitebench_test_data(self, num_items: int = 5) -> Path:
        """Create InfiniteBench test data for comparison"""
        test_data = []
        for i in range(num_items):
            test_data.append({
                "id": i,
                "input": f"What is the passkey? Hidden in text: key_{12345 + i}",
                "answer": str(12345 + i),
                "task_name": "passkey"
            })
        
        temp_file = Path(tempfile.mktemp(suffix='.jsonl'))
        with open(temp_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        
        self.temp_files.append(temp_file)
        return temp_file
    
    def _create_longbench_test_data(self, num_items: int = 3) -> Path:
        """Create LongBench test data for comparison"""
        test_data = []
        for i in range(num_items):
            test_data.append({
                "id": f"narrativeqa_{i}",
                "input": f"Based on the story, what is the character's goal in chapter {i}?",
                "answers": [f"The character wants to find item {i}", f"Finding item {i}"],
                "task_name": "narrativeqa"
            })
        
        temp_file = Path(tempfile.mktemp(suffix='.json'))
        with open(temp_file, 'w') as f:
            json.dump(test_data, f)
        
        self.temp_files.append(temp_file)
        return temp_file
    
    def _create_locomo_test_data(self, num_samples: int = 2) -> Path:
        """Create LoCoMo test data for comparison"""
        test_data = []
        for i in range(num_samples):
            test_data.append({
                "sample_id": f"sample_{i}",
                "conversation": {
                    "session_1": [
                        {"speaker": "Alice", "text": f"My budget is ${30000 + i * 1000}"},
                        {"speaker": "Bob", "text": f"That's a good budget for option {i}"}
                    ],
                    "session_1_date_time": f"2024-01-0{i+1} 10:00:00"
                },
                "qa": [
                    {
                        "question": f"What is Alice's budget in sample {i}?",
                        "answer": f"${30000 + i * 1000}",
                        "category": "budget",
                        "evidence": ["session_1"]
                    }
                ]
            })
        
        temp_file = Path(tempfile.mktemp(suffix='.json'))
        with open(temp_file, 'w') as f:
            json.dump(test_data, f)
        
        self.temp_files.append(temp_file)
        return temp_file
    
    def test_infinitebench_behavioral_equivalence(self):
        """Test InfiniteBench specialized dataset matches SmartDataset behavior"""
        test_file = self._create_infinitebench_test_data(4)
        
        config = BenchmarkConfig(
            name="infinitebench",
            task_type="passkey",
            data_path=test_file
        )
        
        # Create both implementations
        old_dataset = SmartMemoryBenchmarkDataset(config)
        new_dataset = create_dataset_from_config(config)
        
        # Test: Same dataset length
        self.assertEqual(len(old_dataset), len(new_dataset))
        
        # Test: Identical items produced
        for i in range(len(old_dataset)):
            old_item = old_dataset[i]
            new_item = new_dataset[i]
            
            # Compare core data
            self.assertEqual(old_item['item'].id, new_item['item'].id)
            self.assertEqual(old_item['item'].input_text, new_item['item'].input_text)
            self.assertEqual(old_item['item'].ground_truth, new_item['item'].ground_truth)
            self.assertEqual(old_item['ground_truth'], new_item['ground_truth'])
        
        # Test: Identical evaluation results
        test_cases = [
            ("The passkey is 12345", "12345"),
            ("Found key: 12346", "12346"),
            ("Wrong answer", "12345")
        ]
        
        for response, ground_truth in test_cases:
            old_score = old_dataset.evaluate_response(response, ground_truth, "passkey")
            new_score = new_dataset.evaluate_response(response, ground_truth, "passkey")
            self.assertEqual(old_score, new_score, 
                           f"Evaluation mismatch for '{response}' vs '{ground_truth}': {old_score} != {new_score}")
    
    def test_longbench_behavioral_equivalence(self):
        """Test LongBench specialized dataset matches SmartDataset behavior"""
        test_file = self._create_longbench_test_data(3)
        
        config = BenchmarkConfig(
            name="longbench",
            task_type="narrativeqa",
            data_path=test_file
        )
        
        # Create both implementations
        old_dataset = SmartMemoryBenchmarkDataset(config)
        new_dataset = create_dataset_from_config(config)
        
        # Test: Same dataset length
        self.assertEqual(len(old_dataset), len(new_dataset))
        
        # Test: Identical items produced
        for i in range(len(old_dataset)):
            old_item = old_dataset[i]
            new_item = new_dataset[i]
            
            # Compare core data
            self.assertEqual(old_item['item'].id, new_item['item'].id)
            self.assertEqual(old_item['item'].input_text, new_item['item'].input_text)
            self.assertEqual(old_item['item'].ground_truth, new_item['item'].ground_truth)
        
        # Test: Identical F1 evaluation results
        test_cases = [
            ("The character wants to find item 0", ["The character wants to find item 0"]),
            ("Character seeks item zero", ["The character wants to find item 0"]),
            ("Completely different answer", ["The character wants to find item 0"])
        ]
        
        for response, ground_truth in test_cases:
            old_score = old_dataset.evaluate_response(response, ground_truth, "narrativeqa")
            new_score = new_dataset.evaluate_response(response, ground_truth, "narrativeqa")
            self.assertAlmostEqual(old_score, new_score, places=6,
                                 msg=f"F1 evaluation mismatch for '{response}': {old_score} != {new_score}")
    
    def test_locomo_behavioral_equivalence(self):
        """Test LoCoMo specialized dataset matches SmartDataset behavior"""
        test_file = self._create_locomo_test_data(2)
        
        config = BenchmarkConfig(
            name="locomo",
            task_type="conversational_qa",
            data_path=test_file
        )
        
        # Create both implementations
        old_dataset = SmartMemoryBenchmarkDataset(config)  
        new_dataset = create_dataset_from_config(config)
        
        # Test: Same dataset length (should have 2 QA pairs total)
        self.assertEqual(len(old_dataset), len(new_dataset))
        self.assertEqual(len(old_dataset), 2)
        
        # Test: Identical conversation formatting
        for i in range(len(old_dataset)):
            old_item = old_dataset[i]
            new_item = new_dataset[i]
            
            # Check conversation context is formatted identically
            self.assertEqual(old_item['item'].context, new_item['item'].context)
            self.assertEqual(old_item['item'].input_text, new_item['item'].input_text)  
            self.assertEqual(old_item['item'].ground_truth, new_item['item'].ground_truth)
        
        # Test: Identical F1 scoring with stemming
        test_cases = [
            ("$30000", "$30000"),  # Exact match
            ("30000 dollars", "$30000"),  # Partial match
            ("worked", "working"),  # Stemming test
            ("cars", "car"),  # S-suffix removal
            ("completely different", "$30000")  # No match
        ]
        
        for response, ground_truth in test_cases:
            old_score = old_dataset.evaluate_response(response, ground_truth, "conversational_qa")
            new_score = new_dataset.evaluate_response(response, ground_truth, "conversational_qa")
            self.assertAlmostEqual(old_score, new_score, places=6,
                                 msg=f"LoCoMo F1 mismatch for '{response}' vs '{ground_truth}': {old_score} != {new_score}")
    
    def test_prompt_wrapper_equivalence(self):
        """Test that prompt wrappers work identically in both implementations"""
        test_file = self._create_infinitebench_test_data(2)
        config = BenchmarkConfig(name="infinitebench", task_type="passkey", data_path=test_file)
        
        def test_wrapper(context: str, question: str) -> str:
            return f"CONTEXT: {context}\nQUESTION: {question}\nANSWER:"
        
        # Create datasets with identical prompt wrapper
        old_dataset = SmartMemoryBenchmarkDataset(config, prompt_wrapper=test_wrapper)
        new_dataset = create_dataset_from_config(config, prompt_wrapper=test_wrapper)
        
        # Test: Identical prompt formatting
        for i in range(len(old_dataset)):
            old_prompt = old_dataset[i]['prompt']
            new_prompt = new_dataset[i]['prompt']
            self.assertEqual(old_prompt, new_prompt)
            
            # Should contain wrapper formatting
            self.assertIn("CONTEXT:", old_prompt)
            self.assertIn("QUESTION:", old_prompt)
            self.assertIn("ANSWER:", old_prompt)
    
    def test_truncation_equivalence(self):
        """Test that truncation works identically in both implementations"""
        # Create test data with very long context to ensure truncation
        test_data = [{
            "id": 0,
            "input": "What is the passkey? Find the key in this very very very very very very very very very very long text with many many many many words: key_12345",
            "context": "This is a very very very very very very very very very very very very very very very very very very very very long context that definitely needs truncation when we limit tokens",
            "answer": "12345",
            "task_name": "passkey"
        }]
        
        temp_file = Path(tempfile.mktemp(suffix='.jsonl'))
        with open(temp_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        self.temp_files.append(temp_file)
        
        config = BenchmarkConfig(name="infinitebench", task_type="passkey", data_path=temp_file)
        
        # Mock tokenizer for consistent testing
        class MockTokenizer:
            def encode(self, text, add_special_tokens=True):
                return [1] * len(text.split())  # 1 token per word
            def decode(self, tokens, skip_special_tokens=True):
                return " ".join(["word"] * len(tokens))
        
        tokenizer = MockTokenizer()
        
        # Create datasets with very restrictive truncation (should definitely trigger)
        old_dataset = SmartMemoryBenchmarkDataset(
            config, 
            max_context_length=3,  # Very small limit to force truncation
            tokenizer=tokenizer,
            truncation_strategy="right"
        )
        new_dataset = create_dataset_from_config(
            config,
            max_context_length=3,  # Very small limit to force truncation
            tokenizer=tokenizer, 
            truncation_strategy="right"
        )
        
        # Test: Both datasets should have same number of items
        self.assertEqual(len(old_dataset), len(new_dataset))
        
        # Test: Both should handle truncation consistently
        old_item = old_dataset[0]['item']
        new_item = new_dataset[0]['item']
        
        # Check basic equivalence
        self.assertEqual(old_item.id, new_item.id)
        self.assertEqual(old_item.ground_truth, new_item.ground_truth)
        
        # If truncation metadata exists in either, check consistency
        old_has_truncation = 'truncation_info' in old_item.metadata
        new_has_truncation = 'truncation_info' in new_item.metadata
        
        # They should both handle truncation the same way
        self.assertEqual(old_has_truncation, new_has_truncation)
        
        # If both have truncation metadata, it should match
        if old_has_truncation and new_has_truncation:
            old_truncated = old_item.metadata['truncation_info']['was_truncated']
            new_truncated = new_item.metadata['truncation_info']['was_truncated']
            self.assertEqual(old_truncated, new_truncated)
    
    def test_batch_processing_equivalence(self):
        """Test that batch processing works identically"""
        test_file = self._create_longbench_test_data(4)
        config = BenchmarkConfig(name="longbench", task_type="narrativeqa", data_path=test_file)
        
        old_dataset = SmartMemoryBenchmarkDataset(config)
        new_dataset = create_dataset_from_config(config)
        
        # Test batch creation
        from torch.utils.data import DataLoader
        
        old_dataloader = DataLoader(old_dataset, batch_size=2, collate_fn=old_dataset.collate_fn)
        new_dataloader = DataLoader(new_dataset, batch_size=2, collate_fn=new_dataset.collate_fn)
        
        old_batches = list(old_dataloader)
        new_batches = list(new_dataloader)
        
        # Should have same number of batches
        self.assertEqual(len(old_batches), len(new_batches))
        
        # Each batch should have identical structure
        for old_batch, new_batch in zip(old_batches, new_batches):
            self.assertEqual(len(old_batch['prompts']), len(new_batch['prompts']))
            self.assertEqual(len(old_batch['items']), len(new_batch['items']))
            self.assertEqual(len(old_batch['ground_truths']), len(new_batch['ground_truths']))
            
            # Prompts should be identical
            for old_prompt, new_prompt in zip(old_batch['prompts'], new_batch['prompts']):
                self.assertEqual(old_prompt, new_prompt)
    
    def test_sample_limiting_equivalence(self):
        """Test that sample limiting works identically"""
        test_file = self._create_infinitebench_test_data(10)
        config = BenchmarkConfig(
            name="infinitebench", 
            task_type="passkey", 
            data_path=test_file,
            sample_limit=5
        )
        
        old_dataset = SmartMemoryBenchmarkDataset(config)
        new_dataset = create_dataset_from_config(config)
        
        # Should both respect sample limit
        self.assertEqual(len(old_dataset), 5)
        self.assertEqual(len(new_dataset), 5)
        
        # Should contain same items (first 5)
        for i in range(5):
            old_item = old_dataset[i]['item']
            new_item = new_dataset[i]['item']
            self.assertEqual(old_item.id, new_item.id)
            self.assertEqual(old_item.ground_truth, new_item.ground_truth)
    
    def test_error_handling_equivalence(self):
        """Test that error handling works identically"""
        # Test missing file with known benchmark (both should raise FileNotFoundError)
        config = BenchmarkConfig(
            name="infinitebench",
            task_type="passkey",
            data_path=Path("nonexistent_file.jsonl")
        )
        
        with self.assertRaises(FileNotFoundError):
            SmartMemoryBenchmarkDataset(config)
        
        with self.assertRaises(FileNotFoundError):
            create_dataset_from_config(config)
        
        # Test unknown benchmark with factory (should raise ValueError)
        config_unknown = BenchmarkConfig(
            name="unknown_benchmark",
            task_type="unknown_task", 
            data_path=Path("fake.json")
        )
        
        # Both should now raise ValueError for unknown benchmark (after migration)
        # SmartMemoryBenchmarkDataset now redirects to factory, so both behave the same
        with self.assertRaises(ValueError):
            SmartMemoryBenchmarkDataset(config_unknown)
        
        # Factory should raise ValueError for unknown benchmark  
        with self.assertRaises(ValueError):
            create_dataset_from_config(config_unknown)
    
    def test_comprehensive_integration_all_benchmarks(self):
        """Comprehensive test: All benchmarks work identically through factory"""
        test_configs = [
            ("infinitebench", "passkey", self._create_infinitebench_test_data(3)),
            ("longbench", "narrativeqa", self._create_longbench_test_data(2)),
            ("locomo", "conversational_qa", self._create_locomo_test_data(1))
        ]
        
        for benchmark_name, task_type, test_file in test_configs:
            with self.subTest(benchmark=benchmark_name):
                config = BenchmarkConfig(
                    name=benchmark_name,
                    task_type=task_type,
                    data_path=test_file
                )
                
                # Create both implementations
                old_dataset = SmartMemoryBenchmarkDataset(config)
                new_dataset = create_dataset_from_config(config)
                
                # Comprehensive equivalence checks
                self.assertEqual(len(old_dataset), len(new_dataset))
                self.assertEqual(type(old_dataset.config), type(new_dataset.config))
                
                # Check each item
                for i in range(min(len(old_dataset), 3)):  # Test first 3 items
                    old_item = old_dataset[i]
                    new_item = new_dataset[i]
                    
                    # Core data equivalence
                    self.assertEqual(old_item['item'].id, new_item['item'].id)
                    self.assertEqual(old_item['item'].input_text, new_item['item'].input_text)
                    self.assertEqual(old_item['item'].ground_truth, new_item['item'].ground_truth)
                    self.assertEqual(old_item['ground_truth'], new_item['ground_truth'])
                    
                    # Both should be valid BenchmarkItems
                    self.assertIsInstance(old_item['item'], BenchmarkItem)
                    self.assertIsInstance(new_item['item'], BenchmarkItem)


if __name__ == '__main__':
    unittest.main()