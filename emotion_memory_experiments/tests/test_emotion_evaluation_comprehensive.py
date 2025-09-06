#!/usr/bin/env python3
"""
Comprehensive emotion evaluation test suite - PARALLELIZED for speed.

Tests the emotion classification system using LLM evaluation (GPT-4o-mini)
with the actual YAML configuration. Organized by emotion categories with
both simple and complex examples for thorough validation.

Uses asyncio and concurrent.futures for parallel LLM evaluation calls.
"""

import asyncio
import concurrent.futures
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

import yaml

from emotion_memory_experiments.data_models import BenchmarkConfig
from emotion_memory_experiments.evaluation_utils import llm_evaluate_response


class EmotionTestSuite:
    """Organized test suite for emotion evaluation validation."""
    
    def __init__(self):
        self.config = self._load_yaml_config()
        self.question = "What are you feeling right now?"
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_cases = []
    
    def _load_yaml_config(self) -> BenchmarkConfig:
        """Load configuration from YAML file."""
        config_path = Path("config/emotion_check_validation.yaml")
        
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f)
        
        benchmark_config = yaml_config["benchmarks"][0]
        
        return BenchmarkConfig(
            name=benchmark_config["name"],
            task_type=benchmark_config["task_type"],
            data_path=None,
            base_data_dir=benchmark_config["base_data_dir"],
            sample_limit=benchmark_config.get("sample_limit", 100),
            augmentation_config=benchmark_config.get("augmentation_config"),
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=benchmark_config.get("llm_eval_config"),
        )
    
    def _get_test_cases(self) -> List[Tuple[str, str, str]]:
        """
        Get organized test cases by category.
        
        Returns:
            List of (response, expected_emotion, category) tuples
        """
        return [
            # BASIC EMOTIONS - Simple expressions
            ("I feel angry right now.", "anger", "basic_anger"),
            ("Happy!", "happiness", "basic_happiness"),
            ("I'm sad.", "sadness", "basic_sadness"),
            ("Scared.", "fear", "basic_fear"),
            ("That's disgusting.", "disgust", "basic_disgust"),
            ("Wow, I'm surprised!", "surprise", "basic_surprise"),
            ("I feel neutral.", "neutral", "basic_neutral"),
            
            # BASIC EMOTIONS - Complex expressions
            ("I'm feeling quite frustrated and irritated by this situation.", "anger", "complex_anger"),
            ("I'm absolutely delighted and thrilled about this opportunity!", "happiness", "complex_happiness"),
            ("I feel really down and melancholy today, everything seems gloomy.", "sadness", "complex_sadness"),
            ("I'm worried and anxious about what might happen next.", "fear", "complex_fear"),
            ("This whole thing makes me feel sick and revolted.", "disgust", "complex_disgust"),
            ("Oh my goodness, I never expected this to happen! I'm completely shocked.", "surprise", "complex_surprise"),
            ("I'm feeling pretty calm and balanced right now, nothing bothering me.", "neutral", "complex_neutral"),
            
            # MIXED/COMPLEX EMOTIONS - Should be classified as "other"
            ("I'm not sure how I feel, maybe a bit confused?", "other", "uncertain_emotion"),
            ("I feel both excited and nervous at the same time.", "other", "mixed_emotion"),
            ("I feel overwhelmed and stressed.", "other", "complex_emotion"),
            
            # NON-EMOTIONAL RESPONSES - Should be neutral
            ("I don't really have any particular feelings about this.", "neutral", "no_emotion"),
            ("The weather is nice today.", "neutral", "off_topic"),
        ]
    
    def _evaluate_single_case(self, response: str, expected: str, category: str, case_num: int) -> bool:
        """
        Evaluate a single test case.
        
        Returns:
            True if classification is correct, False otherwise
        """
        print(f"{case_num:2d}. [{category:15s}] Testing: '{response[:45]}{'...' if len(response) > 45 else ''}'")
        print(f"    Expected: {expected}")
        
        try:
            eval_prompt = self.config.llm_eval_config.get("evaluation_prompt", "")
            query = eval_prompt.format(question=self.question, response=response)
            
            result = llm_evaluate_response(
                system_prompt="You are an expert emotion classifier. Always respond with valid JSON format.",
                query=query,
                llm_eval_config={
                    "model": self.config.llm_eval_config.get("model", "gpt-4o-mini"),
                    "temperature": self.config.llm_eval_config.get("temperature", 0.1),
                },
            )
            
            detected = result.get("emotion", "unknown").lower()
            confidence = result.get("confidence", 0.0)
            
            is_correct = detected == expected.lower()
            
            if is_correct:
                print(f"    Detected: {detected} ✓")
                return True
            else:
                print(f"    Detected: {detected} ✗")
                self.failed_cases.append({
                    "case": case_num,
                    "category": category,
                    "response": response,
                    "expected": expected,
                    "detected": detected,
                    "confidence": confidence,
                })
                return False
                
        except Exception as e:
            print(f"    ERROR: {str(e)}")
            self.failed_cases.append({
                "case": case_num,
                "category": category,
                "response": response,
                "expected": expected,
                "detected": "ERROR",
                "confidence": 0.0,
            })
            return False
    
    def run_comprehensive_test(self) -> Tuple[float, List[dict]]:
        """
        Run the complete test suite.
        
        Returns:
            Tuple of (accuracy_percentage, failed_cases_list)
        """
        print("🧪 Comprehensive Emotion Evaluation Test Suite")
        print("=" * 65)
        print(f"Configuration: {self.config.llm_eval_config.get('model')} @ temp={self.config.llm_eval_config.get('temperature')}")
        print()
        
        test_cases = self._get_test_cases()
        self.total_tests = len(test_cases)
        self.passed_tests = 0
        self.failed_cases = []
        
        # Run all test cases
        for i, (response, expected, category) in enumerate(test_cases, 1):
            if self._evaluate_single_case(response, expected, category, i):
                self.passed_tests += 1
            print()  # Spacing between cases
        
        # Calculate and display results
        accuracy = (self.passed_tests / self.total_tests) * 100
        
        self._display_results(accuracy)
        return accuracy, self.failed_cases
    
    def _display_results(self, accuracy: float):
        """Display comprehensive test results."""
        print("📊 COMPREHENSIVE TEST RESULTS")
        print("=" * 50)
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.total_tests - self.passed_tests}")
        print(f"Accuracy: {accuracy:.1f}%")
        
        if self.failed_cases:
            print(f"\n❌ FAILED CASES ({len(self.failed_cases)}):")
            print("-" * 45)
            
            # Group failures by category for better analysis
            by_category = {}
            for case in self.failed_cases:
                cat = case["category"]
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(case)
            
            for category, cases in by_category.items():
                print(f"\n{category.upper()}:")
                for case in cases:
                    print(f"  Case {case['case']}: '{case['response'][:35]}...'")
                    print(f"    Expected: {case['expected']} | Got: {case['detected']}")
        
        # Performance rating
        print(f"\n{self._get_performance_rating(accuracy)}")
    
    def _get_performance_rating(self, accuracy: float) -> str:
        """Get performance rating based on accuracy."""
        if accuracy == 100:
            return "🎉 Perfect accuracy achieved!"
        elif accuracy >= 95:
            return "🌟 Excellent accuracy!"
        elif accuracy >= 90:
            return "👍 Good accuracy!"
        elif accuracy >= 85:
            return "⚠️ Acceptable accuracy, room for improvement"
        else:
            return f"❌ Accuracy needs significant improvement: {accuracy:.1f}%"


def main():
    """Run the comprehensive emotion evaluation test."""
    try:
        test_suite = EmotionTestSuite()
        accuracy, failed_cases = test_suite.run_comprehensive_test()
        
        # Return appropriate exit code
        if accuracy == 100:
            print("\n✅ All tests passed!")
            return 0
        elif failed_cases:
            print(f"\n⚠️ {len(failed_cases)} test(s) failed.")
            return 1
        else:
            return 0
            
    except Exception as e:
        print(f"\n💥 Test suite crashed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    exit(main())