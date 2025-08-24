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
        with self.assertRaises(AssertionError) as context_manager:
            self.wrapper.augment_context(context, augmentation_config, answer)
        
        # The assertion error should be raised at line 61: assert answer is not None
        exception = context_manager.exception
        self.assertIsInstance(exception, AssertionError)
    
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
        with self.assertRaises(AssertionError) as context_manager:
            self.wrapper.augment_context(context, augmentation_config, answer)
        
        # The assertion error should be raised at line 62: assert answer in context
        exception = context_manager.exception
        self.assertIsInstance(exception, AssertionError)
    
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
        
        with self.assertRaises(AssertionError):
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
        
        with self.assertRaises(AssertionError):
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
        with self.assertRaises(AssertionError):
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
                with self.assertRaises(AssertionError):
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


class TestPromptWrapperFactory(unittest.TestCase):
    """Test the factory function for creating appropriate prompt wrappers"""
    
    def test_get_memory_prompt_wrapper_passkey(self):
        """Test factory creates PasskeyPromptWrapper for passkey tasks"""
        mock_format = Mock()
        wrapper = get_memory_prompt_wrapper("passkey", mock_format)
        self.assertIsInstance(wrapper, PasskeyPromptWrapper)
    
    def test_get_memory_prompt_wrapper_conversational(self):
        """Test factory creates ConversationalQAPromptWrapper for conversational tasks"""
        mock_format = Mock()
        wrapper = get_memory_prompt_wrapper("conversational_qa", mock_format)
        self.assertIsInstance(wrapper, ConversationalQAPromptWrapper)
    
    def test_get_memory_prompt_wrapper_longbench(self):
        """Test factory creates LongContextQAPromptWrapper for long context tasks"""
        mock_format = Mock()
        wrapper = get_memory_prompt_wrapper("longbench", mock_format)
        self.assertIsInstance(wrapper, LongContextQAPromptWrapper)
    
    def test_get_memory_prompt_wrapper_default(self):
        """Test factory creates MemoryPromptWrapper for unknown tasks"""
        mock_format = Mock()
        wrapper = get_memory_prompt_wrapper("unknown_task", mock_format)
        self.assertIsInstance(wrapper, MemoryPromptWrapper)
        self.assertNotIsInstance(wrapper, (PasskeyPromptWrapper, ConversationalQAPromptWrapper, LongContextQAPromptWrapper))


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


if __name__ == '__main__':
    print("🧪 Running Memory Prompt Wrapper Tests")
    print("🔴 RED PHASE: These tests demonstrate the AssertionError bug")
    print("📍 Tests will show where assertions crash experiments")
    
    unittest.main()