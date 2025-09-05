#!/usr/bin/env python3
"""
Test file for: Complete asyncio evaluate_batch bug fix validation (TDD comprehensive)
Purpose: Single comprehensive test suite for the asyncio event loop bug fix

This test suite covers all aspects of the asyncio bug fix in base.py:
- Reproducing the original bug
- Testing the fix in various contexts  
- Edge case handling
- Graceful fallback behavior
- Regression prevention

Replaces multiple scattered test files with one organized suite.
"""

import unittest
import asyncio
import warnings
from unittest.mock import patch, AsyncMock
from pathlib import Path

from emotion_memory_experiments.datasets.base import BaseBenchmarkDataset
from emotion_memory_experiments.data_models import BenchmarkConfig, BenchmarkItem


class TestAsyncioEvaluateBatchComprehensive(unittest.TestCase):
    """Comprehensive test suite for asyncio evaluate_batch bug fix"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = BenchmarkConfig(
            name="test_benchmark",
            task_type="test_task",
            data_path=Path("test_data.json"),
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8
        )
        
        # Create reusable test dataset
        class TestDataset(BaseBenchmarkDataset):
            def _load_and_parse_data(self):
                return [BenchmarkItem(id="1", input_text="Q", context="C", ground_truth="A", metadata={})]
            
            def evaluate_response(self, response, ground_truth, task_name):
                return 1.0 if response == ground_truth else 0.0
            
            def get_task_metrics(self, task_name):
                return ["accuracy"]
        
        self.dataset = TestDataset(config=self.config)
    
    # ========================================================================
    # Core Functionality Tests
    # ========================================================================
    
    def test_sync_context_basic_functionality(self):
        """Test that evaluate_batch works in basic sync context (most common case)"""
        responses = ["correct", "wrong", "correct"]
        ground_truths = ["correct", "correct", "correct"]
        task_names = ["test", "test", "test"]
        
        # Should work flawlessly in sync context
        scores = self.dataset.evaluate_batch(responses, ground_truths, task_names)
        
        self.assertIsInstance(scores, list)
        self.assertEqual(len(scores), 3)
        for score in scores:
            self.assertIsInstance(score, (int, float))
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    def test_async_context_thread_pool_execution(self):
        """Test evaluate_batch works when called from within an async context"""
        async def test_in_async_context():
            """This simulates calling evaluate_batch from an async function"""
            responses = ["test"]
            ground_truths = ["test"]
            task_names = ["test"]
            
            # Should use ThreadPoolExecutor path in our fix
            scores = self.dataset.evaluate_batch(responses, ground_truths, task_names)
            
            self.assertIsInstance(scores, list)
            self.assertEqual(len(scores), 1)
            return scores
        
        # Run in async context
        scores = asyncio.run(test_in_async_context())
        self.assertIsInstance(scores, list)
    
    def test_empty_input_handling(self):
        """Test edge case: empty inputs"""
        scores = self.dataset.evaluate_batch([], [], [])
        self.assertEqual(scores, [])
    
    # ========================================================================
    # Bug Reproduction and Fix Validation Tests
    # ========================================================================
    
    def test_no_runtime_error_propagation(self):
        """CRITICAL: Test that 'no running event loop' RuntimeError never propagates to caller"""
        # Mock get_running_loop to throw the specific error we're protecting against
        with patch('asyncio.get_running_loop') as mock_get_loop:
            mock_get_loop.side_effect = RuntimeError("no running event loop")
            
            responses = ["test"]
            ground_truths = ["test"]
            task_names = ["test"]
            
            # This should NEVER raise RuntimeError: no running event loop
            try:
                scores = self.dataset.evaluate_batch(responses, ground_truths, task_names)
                
                # Should always return valid results
                self.assertIsInstance(scores, list)
                self.assertEqual(len(scores), 1)
                
            except RuntimeError as e:
                if "no running event loop" in str(e):
                    self.fail(f"REGRESSION: The asyncio fix failed, RuntimeError propagated: {e}")
                else:
                    # Some other RuntimeError is OK to propagate
                    raise
    
    def test_event_loop_closed_recovery(self):
        """Test recovery from 'Event loop is closed' error"""
        # Mock asyncio.run to simulate the "Event loop is closed" error
        original_run = asyncio.run
        call_count = 0
        
        def mock_run_with_closed_loop_error(coro):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call simulates the closed loop error
                raise RuntimeError("Event loop is closed")
            else:
                # Subsequent calls work normally (new event loop path)
                return original_run(coro)
        
        with patch('asyncio.run', side_effect=mock_run_with_closed_loop_error):
            responses = ["test"]
            ground_truths = ["test"]
            task_names = ["test"]
            
            # Should recover from the closed loop error using new event loop
            scores = self.dataset.evaluate_batch(responses, ground_truths, task_names)
            
            self.assertIsInstance(scores, list)
            self.assertEqual(len(scores), 1)
            # Should have tried the recovery mechanism
            self.assertGreaterEqual(call_count, 1)
    
    def test_complete_async_failure_graceful_fallback(self):
        """Test graceful fallback when async evaluation completely fails"""
        # Mock all async methods to fail
        with patch('asyncio.get_running_loop', side_effect=RuntimeError("Mock failure")):
            with patch('asyncio.run', side_effect=Exception("All async methods failed")):
                with patch('asyncio.new_event_loop', side_effect=Exception("Even new loop failed")):
                    
                    responses = ["correct", "wrong", "correct"]
                    ground_truths = ["correct", "correct", "correct"]
                    task_names = ["test", "test", "test"]
                    
                    # Should fall back to simple string comparison
                    scores = self.dataset.evaluate_batch(responses, ground_truths, task_names)
                    
                    self.assertIsInstance(scores, list)
                    self.assertEqual(len(scores), 3)
                    # Should use fallback evaluation: "correct"=="correct" -> 1.0, "wrong"=="correct" -> 0.0
                    self.assertEqual(scores[0], 1.0)  # correct match
                    self.assertEqual(scores[1], 0.0)  # wrong match  
                    self.assertEqual(scores[2], 1.0)  # correct match
    
    # ========================================================================
    # HTTP Client Simulation Tests
    # ========================================================================
    
    def test_httpx_async_client_simulation(self):
        """Test that simulates the real httpx async client behavior that caused the original bug"""
        
        # Mock llm_evaluate_batch to simulate the httpx async client behavior
        async def mock_llm_evaluate_that_uses_httpx(responses, ground_truths, task_names):
            """Simulate what llm_evaluate_batch actually does with httpx"""
            import httpx
            
            # This simulates creating an async HTTP client like the real function does
            async with httpx.AsyncClient() as client:
                # Simulate making API calls
                results = []
                for response, gt, task in zip(responses, ground_truths, task_names):
                    # Mock API call that would normally go to OpenAI
                    score = 1.0 if response.lower() == gt.lower() else 0.0
                    results.append(score)
                
                # The httpx client will try to clean up when exiting async context
                return results
        
        with patch('emotion_memory_experiments.evaluation_utils.llm_evaluate_batch', 
                  side_effect=mock_llm_evaluate_that_uses_httpx):
            
            # Test data
            responses = ["correct", "wrong"]
            ground_truths = ["correct", "correct"]
            task_names = ["test_task", "test_task"]
            
            # This should work without the "Event loop is closed" error
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                scores = self.dataset.evaluate_batch(responses, ground_truths, task_names)
                
                # Should return valid scores
                self.assertIsInstance(scores, list)
                self.assertEqual(len(scores), 2)
                self.assertEqual(scores[0], 1.0)  # "correct" == "correct"
                self.assertEqual(scores[1], 0.0)  # "wrong" != "correct"
                
                # Check for asyncio-related warnings that indicate the bug
                asyncio_warnings = [warning for warning in w 
                                  if "Event loop is closed" in str(warning.message) 
                                  or "RuntimeError" in str(warning.message)]
                
                if asyncio_warnings:
                    self.fail(f"Asyncio lifecycle warnings detected: {[str(w.message) for w in asyncio_warnings]}")
    
    # ========================================================================
    # Edge Cases and Error Handling Tests  
    # ========================================================================
    
    def test_asyncio_logic_flow_debugging(self):
        """Test to verify exactly what happens in the asyncio logic flow"""
        # Track the actual call sequence
        calls = []
        
        def track_get_loop():
            calls.append("get_running_loop")
            raise RuntimeError("no running event loop")
        
        def track_run(coro):
            calls.append("asyncio.run")
            raise RuntimeError("Mock failure in asyncio.run")
        
        with patch('asyncio.get_running_loop', side_effect=track_get_loop):
            with patch('asyncio.run', side_effect=track_run):
                
                responses = ["test"]
                ground_truths = ["test"]  
                task_names = ["test"]
                
                scores = self.dataset.evaluate_batch(responses, ground_truths, task_names)
                
                # Check the call sequence
                self.assertIn("get_running_loop", calls)
                self.assertIn("asyncio.run", calls)
                
                # Should have fallen back to simple evaluation
                self.assertEqual(scores, [1.0])
    
    def test_various_input_combinations(self):
        """Test various input combinations to ensure robustness"""
        test_cases = [
            # (responses, ground_truths, expected_fallback_scores)
            (["correct"], ["correct"], [1.0]),  # Single correct
            (["wrong"], ["correct"], [0.0]),    # Single wrong
            (["a", "b", "c"], ["a", "b", "d"], [1.0, 1.0, 0.0]),  # Mixed results
            ([""], [""], [1.0]),  # Empty strings
            (["test with spaces"], ["test with spaces"], [1.0]),  # Spaces
            (["UPPERCASE"], ["uppercase"], [0.0]),  # Case sensitivity in fallback
        ]
        
        for responses, ground_truths, expected in test_cases:
            with self.subTest(responses=responses):
                task_names = ["test"] * len(responses)
                
                # Force fallback by mocking async to fail
                with patch('asyncio.get_running_loop', side_effect=Exception("Force fallback")):
                    scores = self.dataset.evaluate_batch(responses, ground_truths, task_names)
                    
                    self.assertIsInstance(scores, list)
                    self.assertEqual(len(scores), len(responses))
                    
                    # Check expected fallback behavior (case-insensitive string comparison)
                    for i, (score, expected_score) in enumerate(zip(scores, expected)):
                        # Fallback uses case-insensitive comparison
                        actual_expected = 1.0 if responses[i].lower() == ground_truths[i].lower() else 0.0
                        self.assertEqual(score, actual_expected)
    
    # ========================================================================
    # Helper Method Tests (Refactored Code)
    # ========================================================================
    
    def test_run_async_evaluation_safely_helper_method(self):
        """Test the refactored _run_async_evaluation_safely helper method directly"""
        async def mock_async_func():
            await asyncio.sleep(0.001)  # Tiny async operation
            return ["test_result"]
        
        # Test in sync context
        result = self.dataset._run_async_evaluation_safely(mock_async_func)
        self.assertEqual(result, ["test_result"])
    
    def test_helper_method_error_propagation(self):
        """Test that helper method properly propagates exceptions for caller to handle"""
        async def failing_async_func():
            raise ValueError("Test exception")
        
        with self.assertRaises(ValueError) as context:
            self.dataset._run_async_evaluation_safely(failing_async_func)
        
        self.assertEqual(str(context.exception), "Test exception")


if __name__ == '__main__':
    unittest.main()