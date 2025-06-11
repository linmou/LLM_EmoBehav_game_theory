import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Attempt to import required modules, skipping test if they are not available
try:
    import torch
    from vllm import LLM
    from transformers import AutoTokenizer
    from sequence_prob_vllm_hook import SequenceProbVLLMHook
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

@unittest.skipIf(not VLLM_AVAILABLE, "vLLM, transformers is not installed")
class TestSequenceProbFix(unittest.TestCase):
    """
    Test to verify that the sequence probability fix is working correctly.
    """

    def setUp(self):
        """Initialize model and tokenizer for testing."""
        self.model_name = "gpt2"
        self.model = LLM(model=self.model_name, tensor_parallel_size=1)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.hook = SequenceProbVLLMHook(self.model, self.tokenizer)

    def test_basic_functionality(self):
        """Test that get_log_prob returns valid results instead of empty list."""
        prompts = ["The capital of France is"]
        targets = ["Paris"]
        
        results = self.hook.get_log_prob(prompts, targets)
        
        # Should return exactly one result
        self.assertEqual(len(results), 1, "Should return exactly one result")
        
        # Result should be a dictionary with expected keys
        result = results[0]
        self.assertIsInstance(result, dict, "Result should be a dictionary")
        
        expected_keys = {'sequence', 'log_prob', 'prob', 'perplexity', 'num_tokens'}
        self.assertEqual(set(result.keys()), expected_keys, "Result should contain expected keys")
        
        # Values should be reasonable
        self.assertEqual(result['sequence'], 'Paris', "Sequence should match input")
        self.assertIsInstance(result['log_prob'], float, "log_prob should be a float")
        self.assertLess(result['log_prob'], 0, "log_prob should be negative")
        self.assertGreater(result['prob'], 0, "prob should be positive")
        self.assertLess(result['prob'], 1, "prob should be less than 1")
        self.assertGreater(result['perplexity'], 1, "perplexity should be greater than 1")
        self.assertEqual(result['num_tokens'], 1, "Paris should be 1 token")

    def test_multiple_targets(self):
        """Test with multiple target sequences."""
        prompts = ["The capital of France is"]
        targets = ["Paris", "London", "Berlin"]
        
        results = self.hook.get_log_prob(prompts, targets)
        
        # Should return results for all targets
        self.assertEqual(len(results), 3, "Should return results for all three targets")
        
        # All results should have valid probabilities
        for i, result in enumerate(results):
            self.assertEqual(result['sequence'], targets[i], f"Result {i} should match target")
            self.assertIsInstance(result['log_prob'], float, f"Result {i} log_prob should be a float")
            self.assertLess(result['log_prob'], 0, f"Result {i} log_prob should be negative")

    def test_token_encoding_variants(self):
        """Test that the fix handles both token encoding variants (with and without space)."""
        # This test verifies that we can find tokens regardless of whether they have leading spaces
        prompts = ["Hello"]
        targets = ["world"]  # This might be encoded differently in different contexts
        
        results = self.hook.get_log_prob(prompts, targets)
        
        # Should successfully find and return a result
        self.assertEqual(len(results), 1, "Should find the target token despite encoding variants")
        self.assertIsInstance(results[0]['log_prob'], float, "Should return a valid log probability")

if __name__ == '__main__':
    unittest.main() 
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Attempt to import required modules, skipping test if they are not available
try:
    import torch
    from vllm import LLM
    from transformers import AutoTokenizer
    from sequence_prob_vllm_hook import SequenceProbVLLMHook
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

@unittest.skipIf(not VLLM_AVAILABLE, "vLLM, transformers is not installed")
class TestSequenceProbFix(unittest.TestCase):
    """
    Test to verify that the sequence probability fix is working correctly.
    """

    def setUp(self):
        """Initialize model and tokenizer for testing."""
        self.model_name = "gpt2"
        self.model = LLM(model=self.model_name, tensor_parallel_size=1)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.hook = SequenceProbVLLMHook(self.model, self.tokenizer)

    def test_basic_functionality(self):
        """Test that get_log_prob returns valid results instead of empty list."""
        prompts = ["The capital of France is"]
        targets = ["Paris"]
        
        results = self.hook.get_log_prob(prompts, targets)
        
        # Should return exactly one result
        self.assertEqual(len(results), 1, "Should return exactly one result")
        
        # Result should be a dictionary with expected keys
        result = results[0]
        self.assertIsInstance(result, dict, "Result should be a dictionary")
        
        expected_keys = {'sequence', 'log_prob', 'prob', 'perplexity', 'num_tokens'}
        self.assertEqual(set(result.keys()), expected_keys, "Result should contain expected keys")
        
        # Values should be reasonable
        self.assertEqual(result['sequence'], 'Paris', "Sequence should match input")
        self.assertIsInstance(result['log_prob'], float, "log_prob should be a float")
        self.assertLess(result['log_prob'], 0, "log_prob should be negative")
        self.assertGreater(result['prob'], 0, "prob should be positive")
        self.assertLess(result['prob'], 1, "prob should be less than 1")
        self.assertGreater(result['perplexity'], 1, "perplexity should be greater than 1")
        self.assertEqual(result['num_tokens'], 1, "Paris should be 1 token")

    def test_multiple_targets(self):
        """Test with multiple target sequences."""
        prompts = ["The capital of France is"]
        targets = ["Paris", "London", "Berlin"]
        
        results = self.hook.get_log_prob(prompts, targets)
        
        # Should return results for all targets
        self.assertEqual(len(results), 3, "Should return results for all three targets")
        
        # All results should have valid probabilities
        for i, result in enumerate(results):
            self.assertEqual(result['sequence'], targets[i], f"Result {i} should match target")
            self.assertIsInstance(result['log_prob'], float, f"Result {i} log_prob should be a float")
            self.assertLess(result['log_prob'], 0, f"Result {i} log_prob should be negative")

    def test_token_encoding_variants(self):
        """Test that the fix handles both token encoding variants (with and without space)."""
        # This test verifies that we can find tokens regardless of whether they have leading spaces
        prompts = ["Hello"]
        targets = ["world"]  # This might be encoded differently in different contexts
        
        results = self.hook.get_log_prob(prompts, targets)
        
        # Should successfully find and return a result
        self.assertEqual(len(results), 1, "Should find the target token despite encoding variants")
        self.assertIsInstance(results[0]['log_prob'], float, "Should return a valid log probability")

if __name__ == '__main__':
    unittest.main() 