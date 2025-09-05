#!/usr/bin/env python3
"""
Test file responsible for: memory_prompt_wrapper.py
Purpose: Test MemoryPromptWrapper class and its augment_context method, including edge cases
that cause AssertionError crashes.

This test suite validates that MemoryPromptWrapper handles edge cases gracefully without
crashing experiments due to assertion failures.
"""

import unittest
from unittest.mock import Mock, MagicMock
from pathlib import Path
import sys

# Add the project root to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from emotion_memory_experiments.memory_prompt_wrapper import (
    MemoryPromptWrapper,
    PasskeyPromptWrapper,
    ConversationalQAPromptWrapper,
    LongContextQAPromptWrapper,
    LongbenchRetrievalPromptWrapper,
    get_memory_prompt_wrapper
)


class TestMemoryPromptWrapper(unittest.TestCase):
    """Test suite for MemoryPromptWrapper and related classes"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock prompt format
        self.mock_prompt_format = Mock()
        self.mock_prompt_format.build.return_value = "formatted_prompt"
        
        # Create wrapper instance
        self.wrapper = MemoryPromptWrapper(self.mock_prompt_format)
    
    def test_system_prompt_with_context(self):
        """Test system prompt generation with context"""
        context = "This is test context"
        question = "What is this?"
        
        result = self.wrapper.system_prompt(context, question)
        
        expected = f"{self.wrapper.system_prompt_format}\n\nContext: {context}\n\nQuestion: {question}"
        self.assertEqual(result, expected)
    
    def test_system_prompt_without_context(self):
        """Test system prompt generation without context"""
        context = None
        question = "What is this?"
        
        result = self.wrapper.system_prompt(context, question)
        
        expected = f"{self.wrapper.system_prompt_format}\n\nQuestion: {question}"
        self.assertEqual(result, expected)
    
    def test_user_messages_string_input(self):
        """Test user_messages with string input"""
        result = self.wrapper.user_messages("test message")
        self.assertEqual(result, ["test message"])
    
    def test_user_messages_list_input(self):
        """Test user_messages with list input"""
        messages = ["message1", "message2"]
        result = self.wrapper.user_messages(messages)
        self.assertEqual(result, messages)
    
    # =============================================================================
    # AUGMENT_CONTEXT TESTS - REPRODUCING THE BUG
    # =============================================================================
    
    def test_augment_context_normal_case(self):
        """Test augment_context with normal valid inputs (should work)"""
        context = "The answer is 42. This is the meaning of life."
        answer = "42"
        augmentation_config = {
            "prefix": "**",
            "suffix": "**"
        }
        
        result = self.wrapper.augment_context(context, augmentation_config, answer)
        
        expected = "The answer is **42**. This is the meaning of life."
        self.assertEqual(result, expected)
    
    def test_augment_context_no_config_returns_original(self):
        """Test augment_context without configuration returns original context"""
        context = "Original context"
        answer = "some answer"
        
        result = self.wrapper.augment_context(context, None, answer)
        self.assertEqual(result, context)
        
        result = self.wrapper.augment_context(context, {}, answer)
        self.assertEqual(result, context)
    
    def test_augment_context_no_context_returns_none(self):
        """Test augment_context without context returns None"""
        augmentation_config = {"prefix": "**", "suffix": "**"}
        answer = "test"
        
        result = self.wrapper.augment_context(None, augmentation_config, answer)
        self.assertIsNone(result)
        
        result = self.wrapper.augment_context("", augmentation_config, answer)
        self.assertEqual(result, "")
    
    def test_augment_context_crashes_when_answer_is_none(self):
        """
        BUG REPRODUCTION: Test that augment_context crashes with AssertionError 
        when answer is None (this demonstrates the bug we need to fix)
        """
        context = "This is some context text"
        augmentation_config = {
            "prefix": "**",
            "suffix": "**"
        }
        answer = None  # This will trigger the assertion error
        
        # This should crash the experiment - demonstrating the bug
        with self.assertRaises(ValueError) as context_manager:
            self.wrapper.augment_context(context, augmentation_config, answer)
        
        # The ValueError should be raised for missing answer
        exception = context_manager.exception
        self.assertIsInstance(exception, ValueError)
    
    def test_augment_context_crashes_when_answer_not_in_context(self):
        """
        BUG REPRODUCTION: Test that augment_context crashes with AssertionError
        when answer is not found in context (this demonstrates the bug we need to fix)
        """
        context = "This is some context about cats and dogs"
        augmentation_config = {
            "prefix": "**", 
            "suffix": "**"
        }
        answer = "elephants"  # This answer is NOT in the context
        
        # This should crash the experiment - demonstrating the bug
        with self.assertRaises(ValueError) as context_manager:
            self.wrapper.augment_context(context, augmentation_config, answer)
        
        # The ValueError should be raised for answer not in context
        exception = context_manager.exception
        self.assertIsInstance(exception, ValueError)
    
    def test_augment_context_crashes_with_partial_match(self):
        """
        BUG REPRODUCTION: Test that augment_context crashes even with partial matches
        """
        context = "The temperature is 42 degrees"
        augmentation_config = {
            "prefix": "<<",
            "suffix": ">>"
        }
        answer = "42.0"  # Similar but not exact match - will fail
        
        with self.assertRaises(ValueError):
            self.wrapper.augment_context(context, augmentation_config, answer)
    
    def test_augment_context_crashes_with_case_sensitivity(self):
        """
        BUG REPRODUCTION: Test that augment_context crashes with case differences
        """
        context = "The answer is Python programming"
        augmentation_config = {
            "prefix": "[",
            "suffix": "]"
        }
        answer = "python programming"  # Different case - will fail
        
        with self.assertRaises(ValueError):
            self.wrapper.augment_context(context, augmentation_config, answer)

    # =============================================================================
    # INTEGRATION TESTS - DEMONSTRATING EXPERIMENT CRASH SCENARIOS
    # =============================================================================
    
    def test_call_method_crashes_when_augment_context_fails(self):
        """
        INTEGRATION BUG: Test that the __call__ method crashes when augment_context fails
        This demonstrates how the bug affects the entire experiment pipeline
        """
        context = "Some context"
        question = "What is the answer?"
        augmentation_config = {
            "prefix": "**",
            "suffix": "**"
        }
        answer = None  # Will cause assertion error
        
        # This simulates what happens during experiment execution
        with self.assertRaises(ValueError):
            self.wrapper(
                context=context,
                question=question,
                augmentation_config=augmentation_config,
                answer=answer
            )
    
    def test_experiment_pipeline_simulation_crash(self):
        """
        SIMULATION: Test that simulates the exact failure scenario in experiment pipeline
        """
        # Simulate data from a memory benchmark dataset that has missing or mismatched answers
        test_scenarios = [
            {
                "name": "Missing answer scenario",
                "context": "Long document with information about history...",
                "question": "What year was this written?",
                "answer": None,  # Dataset didn't provide answer
                "augmentation_config": {"prefix": "ANSWER:", "suffix": ""}
            },
            {
                "name": "Answer not found in context",
                "context": "The capital of France is Paris. The population is 2.2 million.",
                "question": "What is the capital?", 
                "answer": "paris",  # lowercase vs Paris - doesn't match exactly
                "augmentation_config": {"prefix": "**", "suffix": "**"}
            },
            {
                "name": "Whitespace/formatting difference",
                "context": "Result: 42\nExplanation: This is the answer",
                "question": "What is the result?",
                "answer": " 42 ",  # Extra whitespace - doesn't match exactly
                "augmentation_config": {"prefix": "[", "suffix": "]"}
            }
        ]
        
        for scenario in test_scenarios:
            with self.subTest(scenario=scenario["name"]):
                # Each of these should crash the experiment
                with self.assertRaises(ValueError):
                    self.wrapper(
                        context=scenario["context"],
                        question=scenario["question"],
                        augmentation_config=scenario["augmentation_config"],
                        answer=scenario["answer"]
                    )

    # =============================================================================
    # EXPECTED BEHAVIOR TESTS (THESE DEFINE WHAT WE WANT AFTER THE FIX)
    # =============================================================================
    
    def test_augment_context_should_handle_none_answer_gracefully(self):
        """
        EXPECTED BEHAVIOR: augment_context should handle None answer gracefully
        (This test will fail initially but should pass after we implement the fix)
        """
        context = "This is some context text"
        augmentation_config = {
            "prefix": "**",
            "suffix": "**"
        }
        answer = None
        
        # After fix: should return original context or handle gracefully
        # For now, this test documents the expected behavior
        # TODO: Uncomment after implementing fix
        # result = self.wrapper.augment_context(context, augmentation_config, answer)
        # self.assertEqual(result, context)  # Should return original context
    
    def test_augment_context_should_handle_missing_answer_gracefully(self):
        """
        EXPECTED BEHAVIOR: augment_context should handle answer not in context gracefully
        (This test will fail initially but should pass after we implement the fix)
        """
        context = "This is some context about cats"
        augmentation_config = {
            "prefix": "**",
            "suffix": "**"
        }
        answer = "dogs"  # Not in context
        
        # After fix: should return original context or handle gracefully
        # For now, this test documents the expected behavior
        # TODO: Uncomment after implementing fix
        # result = self.wrapper.augment_context(context, augmentation_config, answer)
        # self.assertEqual(result, context)  # Should return original context without crashing

    # =============================================================================
    # ADAPTIVE EMOTION AUGMENTATION TESTS
    # =============================================================================
    
    def test_adaptive_augmentation_happiness(self):
        """Test adaptive augmentation with happiness emotion"""
        context = "The answer is 42. This is the meaning of life."
        answer = "42"
        emotion = "happiness"
        augmentation_config = {"method": "adaptive"}
        
        result = self.wrapper.augment_context(context, augmentation_config, answer, emotion)
        
        # Should contain the happiness emotion suffix
        self.assertIn("absolutely thrilled", result)
        self.assertIn("overflowing with pure joy", result)
        self.assertIn("42", result)  # Original answer still there

    def test_adaptive_augmentation_sadness(self):
        """Test adaptive augmentation with sadness emotion"""
        context = "The result is Python. It's a programming language."
        answer = "Python"
        emotion = "sadness"
        augmentation_config = {"method": "adaptive"}
        
        result = self.wrapper.augment_context(context, augmentation_config, answer, emotion)
        
        # Should contain the sadness emotion suffix
        self.assertIn("heavy weight pressing down", result)
        self.assertIn("world seems gray", result)
        self.assertIn("Python", result)  # Original answer still there

    def test_adaptive_augmentation_all_emotions(self):
        """Test adaptive augmentation works for all 6 emotions"""
        context = "The answer is test. This is a test."
        answer = "test"
        augmentation_config = {"method": "adaptive"}
        
        emotion_keywords = {
            "happiness": "absolutely thrilled",
            "sadness": "heavy weight",
            "fear": "heart is pounding", 
            "anger": "absolutely furious",
            "disgust": "physically sick",
            "surprise": "completely stunned"
        }
        
        for emotion, keyword in emotion_keywords.items():
            with self.subTest(emotion=emotion):
                result = self.wrapper.augment_context(context, augmentation_config, answer, emotion)
                self.assertIn(keyword, result, f"Emotion '{emotion}' should contain keyword '{keyword}'")
                self.assertIn("test", result, f"Original answer should still be present for emotion '{emotion}'")

    def test_adaptive_augmentation_error_when_emotion_is_none(self):
        """Test that adaptive mode raises error when emotion is None"""
        context = "The answer is 42. This is the meaning."
        answer = "42"
        emotion = None
        augmentation_config = {"method": "adaptive"}
        
        with self.assertRaises(ValueError) as cm:
            self.wrapper.augment_context(context, augmentation_config, answer, emotion)
        
        self.assertIn("emotion is required", str(cm.exception).lower())

    def test_adaptive_augmentation_error_when_emotion_is_invalid(self):
        """Test that adaptive mode raises error when emotion is invalid"""
        context = "The answer is 42. This is the meaning."
        answer = "42"
        emotion = "invalid_emotion"
        augmentation_config = {"method": "adaptive"}
        
        with self.assertRaises(ValueError) as cm:
            self.wrapper.augment_context(context, augmentation_config, answer, emotion)
        
        self.assertIn("unsupported emotion", str(cm.exception).lower())

    def test_adaptive_mode_bypassed_when_method_not_adaptive(self):
        """Test that normal augmentation still works when method != 'adaptive'"""
        context = "The answer is 42. This is the meaning of life."
        answer = "42"
        emotion = "happiness"  # Even with emotion, should use manual prefix/suffix
        augmentation_config = {
            "method": "manual",  # Not adaptive
            "prefix": "**",
            "suffix": "**"
        }
        
        result = self.wrapper.augment_context(context, augmentation_config, answer, emotion)
        
        # Should use manual prefix/suffix, not emotion suffix
        self.assertIn("**42**", result)
        self.assertNotIn("absolutely thrilled", result)  # Should not contain emotion text

    def test_adaptive_augmentation_replaces_answer_in_context(self):
        """Test that adaptive augmentation properly replaces the answer in context"""
        context = "The capital of France is Paris. The population is large."
        answer = "Paris"
        emotion = "surprise"
        augmentation_config = {"method": "adaptive"}
        
        result = self.wrapper.augment_context(context, augmentation_config, answer, emotion)
        
        # Should replace "Paris" with " Paris [surprise_suffix]"
        self.assertNotEqual(result, context)  # Should be different from original
        self.assertIn("completely stunned", result)  # Should contain surprise emotion
        self.assertIn("Paris", result)  # Original answer should still be there
        
        # The original "Paris" should be replaced with augmented version
        original_paris_context = "The capital of France is Paris. The population is large."
        self.assertNotEqual(result, original_paris_context)


# Factory tests removed - now handled in test_benchmark_prompt_wrapper.py


class TestSpecializedWrappers(unittest.TestCase):
    """Test specialized prompt wrapper subclasses"""
    
    def test_passkey_prompt_wrapper_system_prompt(self):
        """Test PasskeyPromptWrapper system prompt format"""
        mock_format = Mock()
        wrapper = PasskeyPromptWrapper(mock_format)
        
        result = wrapper.system_prompt("test context", "find the passkey")
        self.assertIn("Find and return the passkey", result)
        self.assertIn("test context", result)
    
    def test_conversational_qa_prompt_wrapper_system_prompt(self):
        """Test ConversationalQAPromptWrapper system prompt format"""
        mock_format = Mock()
        wrapper = ConversationalQAPromptWrapper(mock_format)
        
        result = wrapper.system_prompt("conversation history", "what was said?")
        self.assertIn("conversation history", result)
        self.assertIn("Conversation History:", result)
    
    def test_long_context_qa_prompt_wrapper_system_prompt(self):
        """Test LongContextQAPromptWrapper system prompt format"""
        mock_format = Mock()
        wrapper = LongContextQAPromptWrapper(mock_format)
        
        result = wrapper.system_prompt("long document", "summarize this")
        self.assertIn("read the following document", result)
        self.assertIn("Document:", result)

    def test_passkey_wrapper_supports_adaptive_augmentation(self):
        """Test that PasskeyPromptWrapper supports adaptive augmentation"""
        mock_format = Mock()
        wrapper = PasskeyPromptWrapper(mock_format)
        
        context = "The pass key is 12345. Remember it. 12345 is the pass key."
        answer = "12345"  # Just the key, not the full formatted string
        emotion = "anger"
        augmentation_config = {"method": "adaptive"}
        
        result = wrapper.augment_context(context, augmentation_config, answer, emotion)
        
        # Should contain anger emotion suffix
        self.assertIn("absolutely furious", result)
        self.assertIn("12345", result)

    def test_longbench_retrieval_wrapper_supports_adaptive_augmentation(self):
        """Test that LongbenchRetrievalPromptWrapper supports adaptive augmentation"""
        mock_format = Mock()
        wrapper = LongbenchRetrievalPromptWrapper(mock_format)
        
        context = "Paragraph 1\nThis is content.\nParagraph 2\nMore content."
        answer = "Paragraph 1"
        emotion = "disgust"
        augmentation_config = {"method": "adaptive"}
        
        result = wrapper.augment_context(context, augmentation_config, answer, emotion)
        
        # Should contain disgust emotion suffix
        self.assertIn("physically sick", result)
        self.assertIn("Paragraph 1", result)


class TestAdaptiveAugmentationIntegration(unittest.TestCase):
    """Integration tests for adaptive emotion augmentation through the complete pipeline"""
    
    def test_yaml_to_prompt_wrapper_integration(self):
        """Test that YAML configuration flows correctly to prompt wrapper with adaptive augmentation"""
        from emotion_memory_experiments.data_models import BenchmarkConfig
        # Factory function removed - use benchmark_prompt_wrapper.get_benchmark_prompt_wrapper
        from unittest.mock import Mock
        
        # Simulate YAML configuration with adaptive augmentation
        benchmark_config = BenchmarkConfig(
            name="infinitebench",
            task_type="passkey",
            data_path=None,  # Not needed for this test
            sample_limit=10,
            augmentation_config={"method": "adaptive"},  # This is the key test point
            enable_auto_truncation=False,
            truncation_strategy="right", 
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        # Mock prompt format
        mock_prompt_format = Mock()
        mock_prompt_format.build.return_value = "formatted_prompt_output"
        
        # Create prompt wrapper via factory (as done in real pipeline)
        # Use PasskeyPromptWrapper directly since we're testing the specific wrapper
        wrapper = PasskeyPromptWrapper(mock_prompt_format)
        
        # Simulate data that would come from dataset
        context = "The pass key is SECRET123. Remember it. SECRET123 is the pass key."
        question = "What is the pass key?"
        answer = "SECRET123"
        emotion = "happiness"  # This would come from experiment loop
        
        # This is how the prompt wrapper is called in the real pipeline
        result = wrapper(
            context=context,
            question=question, 
            user_messages="Please provide your answer.",
            enable_thinking=False,
            augmentation_config=benchmark_config.augmentation_config,
            answer=answer,
            emotion=emotion
        )
        
        # Verify the flow worked - mock should have been called
        mock_prompt_format.build.assert_called_once()
        
        # Verify the call was made with system prompt that includes adaptive augmentation
        call_args = mock_prompt_format.build.call_args
        system_prompt = call_args[0][0]  # First positional argument
        
        # Should contain happiness emotion text from adaptive augmentation
        self.assertIn("absolutely thrilled", system_prompt)
        self.assertIn("SECRET123", system_prompt)
        
    def test_experiment_config_integration_mock(self):
        """Test ExperimentConfig integration with adaptive augmentation using mocks"""
        from emotion_memory_experiments.data_models import ExperimentConfig, BenchmarkConfig
        from emotion_memory_experiments.memory_prompt_wrapper import MemoryPromptWrapper
        from unittest.mock import Mock, patch
        
        # Create experiment config with adaptive augmentation
        benchmark_config = BenchmarkConfig(
            name="longbench",
            task_type="qa", 
            data_path=None,
            sample_limit=5,
            augmentation_config={"method": "adaptive"},
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        experiment_config = ExperimentConfig(
            model_path="/fake/model/path",
            emotions=["anger", "sadness"],
            intensities=[0.5, 1.0],
            benchmark=benchmark_config,
            output_dir="test_output",
            batch_size=2,
            generation_config={"temperature": 0.1},
            loading_config=None,
            repe_eng_config=None,
            max_evaluation_workers=1,
            pipeline_queue_size=2
        )
        
        # Mock prompt format
        mock_prompt_format = Mock()
        wrapper = MemoryPromptWrapper(mock_prompt_format)
        
        # Test all emotions from config
        test_cases = [
            ("anger", "absolutely furious"),
            ("sadness", "heavy weight")
        ]
        
        for emotion, expected_keyword in test_cases:
            with self.subTest(emotion=emotion):
                context = "The answer is Python. It's a programming language."
                answer = "Python"
                
                result = wrapper.augment_context(
                    context=context,
                    augmentation_config=experiment_config.benchmark.augmentation_config,
                    answer=answer,
                    emotion=emotion
                )
                
                # Verify adaptive augmentation worked
                self.assertIn(expected_keyword, result)
                self.assertIn("Python", result)
                self.assertNotEqual(result, context)  # Should be modified
                
    def test_dataset_prompt_wrapper_partial_integration(self):
        """Test the prompt wrapper partial function creation as done in experiments"""
        from functools import partial
        from emotion_memory_experiments.data_models import BenchmarkConfig
        # Factory function removed - use benchmark_prompt_wrapper.get_benchmark_prompt_wrapper
        from unittest.mock import Mock
        
        # Create benchmark config with adaptive augmentation
        benchmark_config = BenchmarkConfig(
            name="infinitebench",
            task_type="conversational",
            data_path=None,
            sample_limit=None,
            augmentation_config={"method": "adaptive"},
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        mock_prompt_format = Mock()
        mock_prompt_format.build.return_value = "test_output"
        
        # Create wrapper via factory
        # Use ConversationalQAPromptWrapper directly since we're testing the specific wrapper
        memory_prompt_wrapper = ConversationalQAPromptWrapper(mock_prompt_format)
        
        # Create partial as done in real experiment setup
        memory_prompt_wrapper_partial = partial(
            memory_prompt_wrapper.__call__,
            user_messages="Please provide your answer.",
            enable_thinking=False,
            augmentation_config=benchmark_config.augmentation_config,  # Contains method='adaptive'
        )
        
        # Test that partial works with different emotions
        for emotion in ["surprise", "disgust", "fear"]:
            with self.subTest(emotion=emotion):
                # Simulate dataset providing these parameters
                result = memory_prompt_wrapper_partial(
                    context="User: Hello there! Assistant: Hi! How are you doing today?",
                    question="What did the user say?",
                    answer="Hello there!",
                    emotion=emotion  # This comes from experiment loop
                )
                
                # Verify prompt wrapper was called
                self.assertEqual(result, "test_output")
                
                # Verify the build was called (meaning adaptive augmentation triggered)
                self.assertTrue(mock_prompt_format.build.called)
                
                # Reset for next iteration
                mock_prompt_format.build.reset_mock()

    def test_error_propagation_in_integration(self):
        """Test that errors in adaptive mode propagate correctly through the pipeline"""
        from emotion_memory_experiments.data_models import BenchmarkConfig
        # Factory function removed - use benchmark_prompt_wrapper.get_benchmark_prompt_wrapper
        from unittest.mock import Mock
        
        benchmark_config = BenchmarkConfig(
            name="infinitebench", 
            task_type="passkey",
            data_path=None,
            sample_limit=None,
            augmentation_config={"method": "adaptive"},
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        mock_prompt_format = Mock()
        # Use PasskeyPromptWrapper directly since we're testing the specific wrapper
        wrapper = PasskeyPromptWrapper(mock_prompt_format)
        
        # Test invalid emotion propagates error
        with self.assertRaises(ValueError) as cm:
            wrapper(
                context="The pass key is 123. Remember it. 123 is the pass key.",
                question="What is the pass key?",
                user_messages="Answer:",
                enable_thinking=False,
                augmentation_config=benchmark_config.augmentation_config,
                answer="123",
                emotion="invalid_emotion"
            )
        
        self.assertIn("unsupported emotion", str(cm.exception).lower())
        
        # Test missing emotion propagates error
        with self.assertRaises(ValueError) as cm:
            wrapper(
                context="The pass key is 123. Remember it. 123 is the pass key.",
                question="What is the pass key?",
                user_messages="Answer:",
                enable_thinking=False,
                augmentation_config=benchmark_config.augmentation_config,
                answer="123",
                emotion=None
            )
        
        self.assertIn("emotion is required", str(cm.exception).lower())


class TestLongbenchRetrievalBugReproduction(unittest.TestCase):
    """
    Test file responsible for: LongbenchRetrievalPromptWrapper.augment_context line 276 bug
    Purpose: RED PHASE - Reproduce the exact bug using real LongBench data
    
    The bug occurs in line 276: raise ValueError(f" {answer_prefix}{answer_num} or {answer_prefix}{answer_num + 1} not found in context")
    when the code tries to find paragraph boundaries but fails to find the next paragraph.
    """
    
    def setUp(self):
        """Set up test fixtures"""
        from unittest.mock import Mock
        self.mock_prompt_format = Mock()
        self.mock_prompt_format.build.return_value = "formatted_prompt_output"
        self.wrapper = LongbenchRetrievalPromptWrapper(self.mock_prompt_format)
        
        # Load real data case that triggers the bug
        import json
        self.real_bug_case = {
            "context": "Paragraph 1: The Lake Causeway (Calzada del Lago in Spanish) runs north from the lake shore to the city centre, where it continues as Via 5...\n\nParagraph 2: Mordashov is the son of Russian parents...\n\nParagraph 30: The embodiment of the Gopher mascot came to life in 1952 when University of Minnesota assistant bandmaster Jerome Glass bought a fuzzy wool gopher suit with a papier-mâché head and asked one of the band members to climb into it.  \"Goldy\" Gopher (the first name seems to have appeared sometime in the 1960s) became a fixture within the University of Minnesota Marching Band and Pep Band, as each year a band member was chosen to don the suit for that season. Wherever these two bands performed, Goldy was there to glad-hand with the crowd, hug the little kids, torment the cheerleaders and generally add a friendly Minnesota flavor to the event.",
            "answer": "Paragraph 30"
        }
    
    def test_real_longbench_last_paragraph_handles_gracefully(self):
        """
        GREEN PHASE: Test that last paragraph augmentation now works correctly
        
        This uses real data from longbench_passage_retrieval_en.jsonl line 9 where
        the answer is "Paragraph 30" (the last paragraph). The fix should handle
        this gracefully without crashing.
        """
        # Use real data that previously caused crash
        context = self.real_bug_case["context"]
        answer = self.real_bug_case["answer"]  # "Paragraph 30" - the LAST paragraph
        augmentation_config = {"method": "adaptive"}
        emotion = "anger"
        
        # This should now work without crashing
        result = self.wrapper.augment_context(context, augmentation_config, answer, emotion)
        
        # Verify the result is valid
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertNotEqual(result, context)  # Should be different from original
        
        # Should contain the original paragraph 30 content
        self.assertIn("Paragraph 30:", result)
        self.assertIn("Goldy", result)  # Content from paragraph 30
        
        # Should contain emotion augmentation (anger)
        self.assertIn("absolutely furious", result)  # Anger emotion text
    
    def test_experiment_pipeline_integration_works(self):
        """
        GREEN PHASE: Integration test showing experiment pipeline now works
        
        This simulates exactly what happens in memory_experiment_series_runner when it
        processes real LongBench data with last paragraphs - should now work.
        """
        # This is how the experiment pipeline calls the prompt wrapper
        result = self.wrapper(
            context=self.real_bug_case["context"],
            question="Which paragraph discusses the main topic?",
            user_messages="Please provide your answer.",
            enable_thinking=False,
            augmentation_config={"method": "adaptive"},
            answer=self.real_bug_case["answer"],  # "Paragraph 30"
            emotion="anger"
        )
        
        # Verify the pipeline produces a valid prompt
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        
        # Mock should have been called to build the prompt
        self.mock_prompt_format.build.assert_called_once()


if __name__ == '__main__':
    print("🧪 Running Memory Prompt Wrapper Tests")
    print("🔴 RED PHASE: These tests demonstrate the AssertionError bug")
    print("📍 Tests will show where assertions crash experiments")
    
    unittest.main()