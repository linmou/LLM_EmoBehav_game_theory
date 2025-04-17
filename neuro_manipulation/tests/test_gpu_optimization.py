import unittest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
import gc
import sys

from neuro_manipulation.gpu_optimization import (
    BatchSizeFinder,
    find_optimal_batch_size_for_llm,
    find_optimal_batch_size_for_experiment,
    measure_throughput
)

class MockModel(torch.nn.Module):
    """Mock model that simulates memory usage based on batch size"""
    def __init__(self, memory_per_sample=100*1024*1024, oom_batch_size=None):
        super().__init__()
        self.memory_per_sample = memory_per_sample  # Memory used per sample (100MB default)
        self.oom_batch_size = oom_batch_size  # Batch size at which to simulate OOM
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Add a parameter so next(model.parameters()) works
        self.dummy_param = torch.nn.Parameter(torch.zeros(1, device=self.device))
        
    def forward(self, **kwargs):
        batch_size = next(iter(kwargs.values())).shape[0]
        self._allocate_memory(batch_size)
        return torch.zeros(batch_size, 10, device=self.device)
        
    def generate(self, **kwargs):
        batch_size = next(iter(kwargs.values())).shape[0]
        if self.oom_batch_size is not None and batch_size >= self.oom_batch_size:
            raise RuntimeError("CUDA out of memory")
        self._allocate_memory(batch_size)
        return [{"generated_text": f"Sample {i}"} for i in range(batch_size)]
        
    def _allocate_memory(self, batch_size):
        """Simulate memory allocation based on batch size"""
        # Create tensors that use memory proportional to batch size
        tensors = []
        for _ in range(batch_size):
            tensor_size = int(self.memory_per_sample // (4 * 1024))  # Convert to number of float32 elements
            tensors.append(torch.zeros(tensor_size, device=self.device))
        return tensors
        
    def eval(self):
        pass

class MockTokenizer:
    """Mock tokenizer for testing"""
    def __call__(self, texts, padding=None, max_length=None, truncation=None, return_tensors=None):
        batch_size = len(texts)
        # Create a tensor dict that can be used for model inputs
        encoded = {
            "input_ids": torch.ones(batch_size, 10, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 10)
        }
        
        # Create a class with to() method to handle both approaches
        class EncodingWithTo(dict):
            def __init__(self, data):
                super().__init__(data)
                for k, v in data.items():
                    self[k] = v
                    
            def to(self, device):
                """Move tensors to device"""
                for k, v in self.items():
                    if hasattr(v, 'to'):
                        self[k] = v.to(device)
                return self
                
        return EncodingWithTo(encoded)

@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping GPU tests")
class TestBatchSizeFinder(unittest.TestCase):
    """Tests for the BatchSizeFinder class with CUDA available"""
    
    def setUp(self):
        torch.cuda.empty_cache()
        gc.collect()
        self.model = MockModel()
        
    def test_power_search(self):
        """Test power search strategy"""
        finder = BatchSizeFinder(mode="power", init_val=1, max_trials=5)
        
        def sample_input_fn(batch_size):
            return {"input_ids": torch.ones(batch_size, 10, dtype=torch.long, device=self.model.device)}
            
        batch_size = finder.find(
            model=self.model,
            sample_input_fn=sample_input_fn
        )
        
        self.assertGreater(batch_size, 0)
        
    def test_binary_search(self):
        """Test binary search strategy"""
        finder = BatchSizeFinder(mode="binsearch", init_val=1, max_trials=10)
        
        def sample_input_fn(batch_size):
            return {"input_ids": torch.ones(batch_size, 10, dtype=torch.long, device=self.model.device)}
            
        batch_size = finder.find(
            model=self.model,
            sample_input_fn=sample_input_fn
        )
        
        self.assertGreater(batch_size, 0)
        
    def test_with_oom(self):
        """Test that the finder handles OOM errors correctly"""
        # Create model that fails at batch size 16
        model = MockModel(oom_batch_size=16)
        finder = BatchSizeFinder(mode="binsearch", init_val=1, max_trials=10)
        
        def sample_input_fn(batch_size):
            return {"input_ids": torch.ones(batch_size, 10, dtype=torch.long, device=model.device)}
            
        batch_size = finder.find(
            model=model,
            sample_input_fn=sample_input_fn
        )
        
        # Should find a batch size less than the OOM threshold
        self.assertLess(batch_size, 16)
        self.assertGreater(batch_size, 0)
        
    def test_forward_only(self):
        """Test with forward_only=True"""
        finder = BatchSizeFinder(mode="power", init_val=1, max_trials=5)
        
        def sample_input_fn(batch_size):
            return {"input_ids": torch.ones(batch_size, 10, dtype=torch.long, device=self.model.device)}
            
        batch_size = finder.find(
            model=self.model,
            sample_input_fn=sample_input_fn,
            forward_only=True
        )
        
        self.assertGreater(batch_size, 0)
        
    def test_generation_kwargs(self):
        """Test with generation_kwargs"""
        finder = BatchSizeFinder(mode="power", init_val=1, max_trials=5)
        
        def sample_input_fn(batch_size):
            return {"input_ids": torch.ones(batch_size, 10, dtype=torch.long, device=self.model.device)}
            
        batch_size = finder.find(
            model=self.model,
            sample_input_fn=sample_input_fn,
            generation_kwargs={"max_new_tokens": 20}
        )
        
        self.assertGreater(batch_size, 0)

@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping GPU tests")
class TestUtilityFunctions(unittest.TestCase):
    """Tests for the utility functions with CUDA available"""
    
    def setUp(self):
        torch.cuda.empty_cache()
        gc.collect()
        self.model = MockModel()
        self.tokenizer = MockTokenizer()
        
    def test_find_optimal_batch_size_for_llm(self):
        """Test find_optimal_batch_size_for_llm utility"""
        batch_size = find_optimal_batch_size_for_llm(
            model=self.model,
            tokenizer=self.tokenizer,
            sample_text="Sample text",
            max_length=10
        )
        
        self.assertGreater(batch_size, 0)
        
    def test_find_optimal_batch_size_for_experiment(self):
        """Test find_optimal_batch_size_for_experiment backward compatibility wrapper"""
        # Mock all dependencies to prevent actual initialization
        with patch('neuro_manipulation.datasets.game_scenario_dataset.GameScenarioDataset') as mock_dataset_class, \
             patch('torch.utils.data.DataLoader') as mock_dataloader_class, \
             patch('neuro_manipulation.gpu_optimization.BatchSizeFinder') as mock_finder_class:
             
            # Setup dataset mock
            mock_dataset = MagicMock()
            mock_dataset_class.return_value = mock_dataset
            
            # Setup dataloader mock
            mock_dataloader = MagicMock()
            mock_batch = {"prompt": ["Sample prompt"]}
            mock_dataloader.__iter__.return_value = iter([mock_batch])
            mock_dataloader_class.return_value = mock_dataloader
            
            # Setup BatchSizeFinder mock
            mock_finder = MagicMock()
            mock_finder.find.return_value = 32
            mock_finder_class.return_value = mock_finder
            
            # Create mock prompt_wrapper
            prompt_wrapper = MagicMock()
            prompt_wrapper.__call__ = MagicMock(return_value="Sample prompt")
            
            # Mock configs with required fields
            game_config = {
                "decision_class": "test",
                "data_path": "dummy_path"  # Required by GameScenarioDataset
            }
            exp_config = {
                "experiment": {
                    "system_message_template": "Template",
                    "llm": {
                        "generation_config": {
                            "max_new_tokens": 20,
                            "do_sample": True,
                            "temperature": 0.7,
                            "top_p": 0.9
                        }
                    }
                }
            }
            
            # Call the function
            batch_size = find_optimal_batch_size_for_experiment(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt_wrapper=prompt_wrapper,
                game_config=game_config,
                exp_config=exp_config
            )
            
            # Verify the results
            self.assertEqual(batch_size, 32)
            mock_finder_class.assert_called_once()
        
    def test_measure_throughput(self):
        """Test measure_throughput utility"""
        throughput = measure_throughput(
            model=self.model,
            tokenizer=self.tokenizer,
            sample_text="Sample text",
            max_length=10,
            batch_size=2,
            num_batches=2
        )
        
        self.assertGreater(throughput, 0)

class TestBatchSizeFinderMockedCuda(unittest.TestCase):
    """Tests with mocked CUDA functions to test without GPU"""
    
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.synchronize')
    @patch('torch.cuda.reset_peak_memory_stats')
    @patch('torch.cuda.max_memory_allocated')
    @patch('torch.cuda.empty_cache')
    def test_batch_size_finder_with_mocked_cuda(
        self, mock_empty_cache, mock_max_memory, mock_reset, mock_sync, mock_get_props
    ):
        """Test BatchSizeFinder with mocked CUDA functions"""
        # Mock GPU properties
        mock_device_props = MagicMock()
        mock_device_props.total_memory = 8 * 1024**3  # 8GB
        mock_get_props.return_value = mock_device_props
        
        # Memory usage per batch size
        batch_memory = {
            1: 500 * 1024**2,    # 500MB
            2: 1000 * 1024**2,   # 1GB
            4: 2000 * 1024**2,   # 2GB
            8: 4000 * 1024**2,   # 4GB
            16: 8000 * 1024**2,  # 8GB (over limit)
        }
        
        def side_effect_memory():
            return batch_memory.get(current_batch_size, 8 * 1024**3)  # Default to max memory
            
        mock_max_memory.side_effect = side_effect_memory
        
        # Create mock model with parameters
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device('cpu')
        # Fix: Return a proper iterator with at least one parameter
        mock_model.parameters = MagicMock(return_value=iter([mock_param]))
        mock_model.generate.return_value = [{"generated_text": "test"}]
        mock_model.device = torch.device('cpu')
        
        # Create finder
        finder = BatchSizeFinder(mode="binsearch", safety_margin=0.8)  # 80% of 8GB = 6.4GB
        
        # Keep track of the current batch size for the memory mock
        current_batch_size = 1
        
        # Test
        def sample_input_fn(batch_size):
            nonlocal current_batch_size
            current_batch_size = batch_size
            return {"input_ids": torch.ones(batch_size, 10)}
            
        batch_size = finder.find(
            model=mock_model,
            sample_input_fn=sample_input_fn
        )
        
        # With 80% of 8GB = 6.4GB and our memory pattern, batch size 8 (4GB) should be chosen
        # then reduced by the safety factor (0.95)
        self.assertEqual(batch_size, int(8 * 0.95))

if __name__ == '__main__':
    unittest.main() 