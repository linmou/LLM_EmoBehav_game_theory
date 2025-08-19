"""
Comprehensive test cases for all evaluation methods.
Tests both success and failure cases for InfiniteBench and LongBench tasks.
"""

import pytest
from typing import Dict, Any
from evaluation_utils import (
    get_score_one,
    get_success_test_cases,
    get_failure_test_cases,
    # InfiniteBench functions
    get_score_one_passkey,
    get_score_one_kv_retrieval,
    get_score_one_number_string,
    get_score_one_code_run,
    get_score_one_code_debug,
    get_score_one_math_find,
    get_score_one_math_calc,
    get_score_one_longbook_choice_eng,
    get_score_one_longbook_qa_eng,
    get_score_one_longbook_qa_chn,
    get_score_one_longbook_sum_eng,
    get_score_one_longdialogue_qa_eng,
    # LongBench functions
    longbench_qa_f1_score,
    longbench_qa_f1_zh_score,
    rouge_score,
    rouge_zh_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)


class TestInfiniteBenchEvaluations:
    """Test cases for InfiniteBench evaluation functions"""

    def test_passkey_success(self):
        """Test successful passkey evaluation"""
        prediction = "The passkey is 12345"
        ground_truth = "12345"
        result = get_score_one_passkey(prediction, ground_truth, "test_model")
        assert result == True, f"Expected True, got {result}"

    def test_passkey_failure(self):
        """Test failed passkey evaluation"""
        prediction = "I don't know the passkey"
        ground_truth = "12345"
        result = get_score_one_passkey(prediction, ground_truth, "test_model")
        assert result == False, f"Expected False, got {result}"

    def test_kv_retrieval_success(self):
        """Test successful KV retrieval evaluation"""
        prediction = "The answer is: apple"
        ground_truth = "apple"
        result = get_score_one_kv_retrieval(prediction, ground_truth, "test_model")
        assert result == True, f"Expected True, got {result}"

    def test_kv_retrieval_failure(self):
        """Test failed KV retrieval evaluation"""
        prediction = "The answer is banana"
        ground_truth = "apple"
        result = get_score_one_kv_retrieval(prediction, ground_truth, "test_model")
        assert result == False, f"Expected False, got {result}"

    def test_number_string_success(self):
        """Test successful number string evaluation"""
        prediction = "The number is 789"
        ground_truth = "789"
        result = get_score_one_number_string(prediction, ground_truth, "test_model")
        assert result == True, f"Expected True, got {result}"

    def test_number_string_failure(self):
        """Test failed number string evaluation"""
        prediction = "No numbers here"
        ground_truth = "789"
        result = get_score_one_number_string(prediction, ground_truth, "test_model")
        assert result == False, f"Expected False, got {result}"

    def test_code_run_success(self):
        """Test successful code run evaluation"""
        prediction = "The output is: 42"
        ground_truth = 42
        result = get_score_one_code_run(prediction, ground_truth, "test_model")
        assert result == True, f"Expected True, got {result}"

    def test_code_run_failure(self):
        """Test failed code run evaluation"""
        prediction = "Error occurred"
        ground_truth = 42
        result = get_score_one_code_run(prediction, ground_truth, "test_model")
        assert result == False, f"Expected False, got {result}"

    def test_code_debug_success(self):
        """Test successful code debug evaluation"""
        prediction = "The answer is B"
        ground_truth = ["function_name", "B"]
        result = get_score_one_code_debug(prediction, ground_truth, "test_model")
        assert result == True, f"Expected True, got {result}"

    def test_code_debug_failure(self):
        """Test failed code debug evaluation"""
        prediction = "I don't know"
        ground_truth = ["function_name", "B"]
        result = get_score_one_code_debug(prediction, ground_truth, "test_model")
        assert result == False, f"Expected False, got {result}"

    def test_math_find_success(self):
        """Test successful math find evaluation"""
        prediction = "The result is 3.14"
        ground_truth = 3.14
        result = get_score_one_math_find(prediction, ground_truth, "test_model")
        assert result == True, f"Expected True, got {result}"

    def test_math_find_failure(self):
        """Test failed math find evaluation"""
        prediction = "No solution found"
        ground_truth = 3.14
        result = get_score_one_math_find(prediction, ground_truth, "test_model")
        assert result == False, f"Expected False, got {result}"

    def test_math_calc_success(self):
        """Test successful math calc evaluation"""
        prediction = "1 2 3 4 5"
        ground_truth = [1, 2, 3, 4, 5]
        result = get_score_one_math_calc(prediction, ground_truth, "test_model")
        assert result == 1.0, f"Expected 1.0, got {result}"

    def test_math_calc_failure(self):
        """Test failed math calc evaluation"""
        prediction = "6 7 8 9 10"
        ground_truth = [1, 2, 3, 4, 5]
        result = get_score_one_math_calc(prediction, ground_truth, "test_model")
        assert result == 0.0, f"Expected 0.0, got {result}"

    def test_longbook_choice_success(self):
        """Test successful longbook choice evaluation"""
        prediction = "The answer is A"
        ground_truth = ["A"]
        result = get_score_one_longbook_choice_eng(prediction, ground_truth, "test_model")
        assert result == True, f"Expected True, got {result}"

    def test_longbook_choice_failure(self):
        """Test failed longbook choice evaluation"""
        prediction = "The answer is X"
        ground_truth = ["A"]
        result = get_score_one_longbook_choice_eng(prediction, ground_truth, "test_model")
        assert result == False, f"Expected False, got {result}"

    def test_longbook_qa_eng_success(self):
        """Test successful English longbook QA evaluation"""
        prediction = "The main character is Alice"
        ground_truth = ["Alice is the main character"]
        result = get_score_one_longbook_qa_eng(prediction, ground_truth, "test_model")
        assert result > 0.8, f"Expected high F1 score, got {result}"

    def test_longbook_qa_eng_failure(self):
        """Test failed English longbook QA evaluation"""
        prediction = "The story is about robots"
        ground_truth = ["Alice is the main character"]
        result = get_score_one_longbook_qa_eng(prediction, ground_truth, "test_model")
        assert result < 0.2, f"Expected low F1 score, got {result}"

    def test_longbook_qa_chn_success(self):
        """Test successful Chinese longbook QA evaluation"""
        prediction = "主角是爱丽丝"
        ground_truth = ["爱丽丝是主角"]
        result = get_score_one_longbook_qa_chn(prediction, ground_truth, "test_model")
        assert result > 0.8, f"Expected high F1 score, got {result}"

    def test_longbook_qa_chn_failure(self):
        """Test failed Chinese longbook QA evaluation"""
        prediction = "这是关于机器人的"
        ground_truth = ["爱丽丝是主角"]
        result = get_score_one_longbook_qa_chn(prediction, ground_truth, "test_model")
        assert result < 0.2, f"Expected low F1 score, got {result}"

    def test_longdialogue_qa_eng_success(self):
        """Test successful long dialogue QA evaluation"""
        prediction = "JOHN said hello"
        ground_truth = ["JOHN"]
        result = get_score_one_longdialogue_qa_eng(prediction, ground_truth, "test_model")
        assert result == True, f"Expected True, got {result}"

    def test_longdialogue_qa_eng_failure(self):
        """Test failed long dialogue QA evaluation"""
        prediction = "MARY said goodbye"
        ground_truth = ["JOHN"]
        result = get_score_one_longdialogue_qa_eng(prediction, ground_truth, "test_model")
        assert result == False, f"Expected False, got {result}"


class TestLongBenchEvaluations:
    """Test cases for LongBench evaluation functions"""

    def test_qa_f1_score_success(self):
        """Test successful QA F1 score evaluation"""
        prediction = "The story is about a young wizard"
        ground_truth = ["A young wizard's story"]
        result = longbench_qa_f1_score(prediction, ground_truth)
        assert result > 0.8, f"Expected high F1 score, got {result}"

    def test_qa_f1_score_failure(self):
        """Test failed QA F1 score evaluation"""
        prediction = "The story is about robots"
        ground_truth = ["A young wizard's story"]
        result = longbench_qa_f1_score(prediction, ground_truth)
        assert result < 0.2, f"Expected low F1 score, got {result}"

    def test_qa_f1_zh_score_success(self):
        """Test successful Chinese QA F1 score evaluation"""
        prediction = "这个故事是关于年轻巫师的"
        ground_truth = ["年轻巫师的故事"]
        result = longbench_qa_f1_zh_score(prediction, ground_truth)
        assert result > 0.8, f"Expected high F1 score, got {result}"

    def test_qa_f1_zh_score_failure(self):
        """Test failed Chinese QA F1 score evaluation"""
        prediction = "这是关于机器人的故事"
        ground_truth = ["年轻巫师的故事"]
        result = longbench_qa_f1_zh_score(prediction, ground_truth)
        assert result < 0.2, f"Expected low F1 score, got {result}"

    def test_classification_score_success(self):
        """Test successful classification evaluation"""
        prediction = "This is about location"
        ground_truth = "location"
        result = classification_score(prediction, ground_truth)
        assert result == 1.0, f"Expected 1.0, got {result}"

    def test_classification_score_failure(self):
        """Test failed classification evaluation"""
        prediction = "This is about animals"
        ground_truth = "location"
        result = classification_score(prediction, ground_truth)
        assert result == 0.0, f"Expected 0.0, got {result}"

    def test_retrieval_score_success(self):
        """Test successful retrieval evaluation"""
        prediction = "The document contains important information"
        ground_truth = "important information"
        result = retrieval_score(prediction, ground_truth)
        assert result == 1.0, f"Expected 1.0, got {result}"

    def test_retrieval_score_failure(self):
        """Test failed retrieval evaluation"""
        prediction = "The document contains irrelevant data"
        ground_truth = "important information"
        result = retrieval_score(prediction, ground_truth)
        assert result == 0.0, f"Expected 0.0, got {result}"

    def test_count_score_success(self):
        """Test successful count evaluation"""
        prediction = "There are 5 passages"
        ground_truth = "5"
        result = count_score(prediction, ground_truth)
        assert result == 1.0, f"Expected 1.0, got {result}"

    def test_count_score_failure(self):
        """Test failed count evaluation"""
        prediction = "Many passages exist"
        ground_truth = "5"
        result = count_score(prediction, ground_truth)
        assert result == 0.0, f"Expected 0.0, got {result}"

    def test_code_sim_score_success(self):
        """Test successful code similarity evaluation"""
        prediction = "def function(): return value"
        ground_truth = "def function(): return value"
        result = code_sim_score(prediction, ground_truth)
        assert result == 1.0, f"Expected 1.0, got {result}"

    def test_code_sim_score_failure(self):
        """Test failed code similarity evaluation"""
        prediction = "def different_function(): return other_value"
        ground_truth = "def function(): return value"
        result = code_sim_score(prediction, ground_truth)
        assert result < 0.5, f"Expected low similarity, got {result}"


class TestUnifiedEvaluation:
    """Test cases for the unified get_score_one function"""

    def test_infinitebench_tasks(self):
        """Test that all InfiniteBench tasks route correctly"""
        infinitebench_tasks = [
            "passkey", "kv_retrieval", "number_string", "code_run", "code_debug",
            "math_find", "math_calc", "longbook_choice_eng", "longbook_qa_eng",
            "longbook_qa_chn", "longbook_sum_eng", "longdialogue_qa_eng"
        ]
        
        for task in infinitebench_tasks:
            # Use simple test cases for each
            if task == "passkey":
                result = get_score_one("The passkey is 123", "123", task, "test_model")
            elif task == "kv_retrieval":
                result = get_score_one("The answer is apple", "apple", task, "test_model")
            elif task == "number_string":
                result = get_score_one("The number is 456", "456", task, "test_model")
            elif task == "code_run":
                result = get_score_one("Output: 42", 42, task, "test_model")
            elif task == "code_debug":
                result = get_score_one("Answer is B", ["func", "B"], task, "test_model")
            elif task == "math_find":
                result = get_score_one("Result is 3.14", 3.14, task, "test_model")
            elif task == "math_calc":
                result = get_score_one("1 2 3", [1, 2, 3], task, "test_model")
            elif task == "longbook_choice_eng":
                result = get_score_one("Answer is A", ["A"], task, "test_model")
            elif task == "longbook_qa_eng":
                result = get_score_one("Alice is here", ["Alice"], task, "test_model")
            elif task == "longbook_qa_chn":
                result = get_score_one("爱丽丝在这里", ["爱丽丝"], task, "test_model")
            elif task == "longbook_sum_eng":
                result = get_score_one("Summary text", "Summary", task, "test_model")
            elif task == "longdialogue_qa_eng":
                result = get_score_one("JOHN spoke", ["JOHN"], task, "test_model")
            
            assert isinstance(result, (int, float)), f"Task {task} should return numeric result, got {type(result)}"
            assert 0 <= result <= 1, f"Task {task} should return score 0-1, got {result}"

    def test_longbench_tasks(self):
        """Test that all LongBench tasks route correctly"""
        longbench_tasks = [
            "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
            "hotpotqa", "2wikimqa", "musique", "dureader", "gov_report",
            "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum",
            "lsht", "passage_retrieval_en", "passage_count", "passage_retrieval_zh",
            "lcc", "repobench-p"
        ]
        
        for task in longbench_tasks:
            # Use simple test cases for each
            if task in ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "triviaqa"]:
                result = get_score_one("The answer is correct", ["correct answer"], task, "test_model")
            elif task == "multifieldqa_zh":
                result = get_score_one("答案是正确的", ["正确答案"], task, "test_model")
            elif task in ["dureader", "vcsum"]:
                result = get_score_one("摘要文本", "摘要", task, "test_model")
            elif task in ["gov_report", "qmsum", "multi_news", "samsum"]:
                result = get_score_one("Summary text", "Summary", task, "test_model")
            elif task in ["trec", "lsht"]:
                result = get_score_one("This is location", "location", task, "test_model")
            elif task in ["passage_retrieval_en", "passage_retrieval_zh"]:
                result = get_score_one("Found passage", "passage", task, "test_model")
            elif task == "passage_count":
                result = get_score_one("There are 5 passages", "5", task, "test_model")
            elif task in ["lcc", "repobench-p"]:
                result = get_score_one("def func(): pass", "def func(): pass", task, "test_model")
            
            assert isinstance(result, (int, float)), f"Task {task} should return numeric result, got {type(result)}"
            assert 0 <= result <= 1, f"Task {task} should return score 0-1, got {result}"

    def test_unknown_task_fallback(self):
        """Test that unknown tasks fall back to exact match"""
        result = get_score_one("exact match", "exact match", "unknown_task", "test_model")
        assert result == 1.0, f"Expected exact match to return 1.0, got {result}"
        
        result = get_score_one("no match", "different", "unknown_task", "test_model")
        assert result == 0.0, f"Expected no match to return 0.0, got {result}"


class TestSuccessAndFailureCases:
    """Test comprehensive success and failure cases"""

    def test_all_success_cases(self):
        """Test all predefined success cases"""
        success_cases = get_success_test_cases()
        
        for task_name, test_case in success_cases.items():
            prediction = test_case["prediction"]
            ground_truth = test_case["ground_truth"]
            expected_score = test_case["expected_score"]
            
            actual_score = get_score_one(prediction, ground_truth, task_name, "test_model")
            
            # Allow some tolerance for F1 scores
            if expected_score == 1.0:
                assert actual_score >= 0.8, f"Task {task_name} success case failed: expected >= 0.8, got {actual_score}"
            else:
                assert abs(actual_score - expected_score) < 0.2, f"Task {task_name} success case failed: expected ~{expected_score}, got {actual_score}"

    def test_all_failure_cases(self):
        """Test all predefined failure cases"""
        failure_cases = get_failure_test_cases()
        
        for task_name, test_case in failure_cases.items():
            prediction = test_case["prediction"]
            ground_truth = test_case["ground_truth"]
            expected_score = test_case["expected_score"]
            
            actual_score = get_score_one(prediction, ground_truth, task_name, "test_model")
            
            # Allow some tolerance but should be low scores
            assert actual_score <= 0.2, f"Task {task_name} failure case failed: expected <= 0.2, got {actual_score}"


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_responses(self):
        """Test handling of empty responses"""
        tasks = ["passkey", "narrativeqa", "trec"]
        
        for task in tasks:
            result = get_score_one("", "some_ground_truth", task, "test_model")
            assert isinstance(result, (int, float)), f"Empty response for {task} should return numeric value"
            assert result == 0.0, f"Empty response for {task} should return 0.0, got {result}"

    def test_none_ground_truth(self):
        """Test handling of None ground truth"""
        result = get_score_one("some response", None, "passkey", "test_model")
        assert isinstance(result, (int, float)), "None ground truth should return numeric value"

    def test_list_vs_string_ground_truth(self):
        """Test consistent handling of list vs string ground truth"""
        # Test with string ground truth
        result1 = get_score_one("The answer is apple", "apple", "kv_retrieval", "test_model")
        
        # Test with list ground truth
        result2 = get_score_one("The answer is apple", ["apple"], "kv_retrieval", "test_model")
        
        assert result1 == result2, f"String and list ground truth should give same result: {result1} vs {result2}"

    def test_case_sensitivity(self):
        """Test case sensitivity handling"""
        # Test that case is handled appropriately for different tasks
        result = get_score_one("JOHN spoke", ["john"], "longdialogue_qa_eng", "test_model")
        assert result == 1.0, f"Case insensitive task should match: got {result}"


if __name__ == "__main__":
    # Run basic smoke tests if executed directly
    print("Running basic evaluation tests...")
    
    # Test a few key functions
    assert get_score_one_passkey("The key is 123", "123", "test") == True
    assert get_score_one_kv_retrieval("Answer: apple", "apple", "test") == True
    assert longbench_qa_f1_score("The wizard story", ["wizard story"]) > 0.5
    assert classification_score("about location", "location") == 1.0
    
    print("✅ All basic tests passed!")
    print("\nTo run comprehensive tests, use: pytest test_evaluations.py")