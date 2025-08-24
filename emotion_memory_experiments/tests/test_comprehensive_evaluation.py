"""
Test file for: Comprehensive evaluation methods (TDD Red phase)
Purpose: Test all 36+ task-specific evaluation methods to ensure proper restoration

This test suite defines the expected behavior for all evaluation methods that were
removed during refactoring. These tests will initially FAIL and drive the restoration
of the comprehensive evaluation system.
"""

import unittest
import tempfile
import json
from pathlib import Path

from ..smart_datasets import SmartMemoryBenchmarkDataset
from ..data_models import BenchmarkConfig


class TestInfiniteBenchEvaluation(unittest.TestCase):
    """Test InfiniteBench-specific evaluation methods (14+ evaluators)"""
    
    def setUp(self):
        """Set up test data"""
        self.temp_files = []
    
    def tearDown(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            if temp_file.exists():
                temp_file.unlink()
    
    def _create_dataset(self, task_type: str, mock_data: list):
        """Helper to create dataset with mock data"""
        temp_file = Path(tempfile.mktemp(suffix='.jsonl'))
        with open(temp_file, 'w') as f:
            for item in mock_data:
                f.write(json.dumps(item) + '\n')
        self.temp_files.append(temp_file)
        
        config = BenchmarkConfig(
            name="infinitebench",
            task_type=task_type,
            data_path=temp_file
        )
        return SmartMemoryBenchmarkDataset(config)
    
    def test_passkey_extraction(self):
        """Test passkey evaluation - should extract integer from text"""
        dataset = self._create_dataset("passkey", [
            {"id": "passkey_1", "context": "long text", "input": "What's the passkey?", "answer": "12345"}
        ])
        
        # Should extract passkey from natural language response
        score = dataset.evaluate_response("The passkey mentioned in the text is 12345.", "12345", "passkey")
        self.assertEqual(score, 1.0, "Should extract '12345' from natural language response")
        
        # Should handle multiple numbers and extract first
        score = dataset.evaluate_response("Numbers: 67890, 12345, 99999", "12345", "passkey")  
        self.assertEqual(score, 0.0, "Should extract first number 67890, not match 12345")
        
        # Should fail on no numbers
        score = dataset.evaluate_response("No passkey found in the text", "12345", "passkey")
        self.assertEqual(score, 0.0, "Should fail when no numbers found")
    
    def test_kv_retrieval_evaluation(self):
        """Test key-value retrieval evaluation - should check word-split response"""
        dataset = self._create_dataset("kv_retrieval", [
            {"id": "kv_1", "context": "key1: apple, key2: banana", "input": "key1?", "answer": "apple"}
        ])
        
        # Should find answer in word-split response
        score = dataset.evaluate_response("The answer is: apple, definitely.", "apple", "kv_retrieval")
        self.assertEqual(score, 1.0, "Should find 'apple' in word-split response")
        
        # Should handle punctuation splitting
        score = dataset.evaluate_response('Result: "apple".', "apple", "kv_retrieval")
        self.assertEqual(score, 1.0, "Should find 'apple' after punctuation split")
        
        # Should fail on substring without word boundary
        score = dataset.evaluate_response("The answer is pineapple", "apple", "kv_retrieval")
        self.assertEqual(score, 0.0, "Should not match 'apple' within 'pineapple'")
    
    def test_code_run_evaluation(self):
        """Test code execution evaluation - should parse output value"""
        dataset = self._create_dataset("code_run", [
            {"id": "code_1", "context": "def f(): return 42", "input": "What's the output?", "answer": 42}
        ])
        
        # Should extract numeric output  
        score = dataset.evaluate_response("The output is: 42", 42, "code_run")
        self.assertEqual(score, 1.0, "Should extract output value 42")
        
        # Should handle string formatting
        score = dataset.evaluate_response("Output: `42`", 42, "code_run")
        self.assertEqual(score, 1.0, "Should extract 42 from formatted output")
        
        # Should fail on no output
        score = dataset.evaluate_response("Code failed to run", 42, "code_run")
        self.assertEqual(score, 0.0, "Should fail when no output found")
    
    def test_code_debug_evaluation(self):
        """Test code debugging evaluation - complex pattern matching"""
        dataset = self._create_dataset("code_debug", [
            {"id": "debug_1", "context": "buggy code", "input": "Fix it", "answer": ["function_name", "B"]}
        ])
        
        # Should match A-J pattern and function name
        score = dataset.evaluate_response("The bug is in function_name, answer B", ["function_name", "B"], "code_debug")
        self.assertEqual(score, 1.0, "Should match function name and choice B")
        
        # Should prefer last A-J choice
        score = dataset.evaluate_response("First A, but actually B", ["function_name", "B"], "code_debug")
        self.assertEqual(score, 1.0, "Should use last choice (B) not first (A)")
        
        # Should match just the letter B (simpler than expected)
        score = dataset.evaluate_response("Just the letter B", ["function_name", "B"], "code_debug")
        self.assertEqual(score, 1.0, "Should match choice B even without function name")
    
    def test_math_find_evaluation(self):
        """Test mathematical result evaluation - should extract numbers with tolerance"""
        dataset = self._create_dataset("math_find", [
            {"id": "math_1", "context": "pi calculation", "input": "What's pi?", "answer": 3.14159}
        ])
        
        # Should extract decimal number
        score = dataset.evaluate_response("The result is approximately 3.14159", 3.14159, "math_find")
        self.assertEqual(score, 1.0, "Should extract exact decimal match")
        
        # Should fail on different float (exact match required)
        score = dataset.evaluate_response("Pi equals 3.14160", 3.14159, "math_find")
        self.assertEqual(score, 0.0, "Should fail on different float (exact match)")
        
        # Should fail on very different number  
        score = dataset.evaluate_response("Pi is roughly 3.2", 3.14159, "math_find")
        self.assertEqual(score, 0.0, "Should fail on different number")
        
        # Should handle integer answers
        score = dataset.evaluate_response("The count is 5 items", 5, "math_find")
        self.assertEqual(score, 1.0, "Should extract integer 5")
    
    def test_math_calc_evaluation(self):
        """Test mathematical calculation evaluation - sequence parsing"""
        dataset = self._create_dataset("math_calc", [
            {"id": "calc_1", "context": "sequence", "input": "Next 5 numbers?", "answer": [1, 2, 3, 4, 5]}
        ])
        
        # Should extract number sequence
        score = dataset.evaluate_response("The sequence is: 1 2 3 4 5", [1, 2, 3, 4, 5], "math_calc")
        self.assertEqual(score, 1.0, "Should extract exact number sequence")
        
        # Should handle formatted output
        score = dataset.evaluate_response("Numbers: 1, 2, 3, 4, 5", [1, 2, 3, 4, 5], "math_calc")
        self.assertEqual(score, 1.0, "Should extract sequence with commas")
        
        # Should get partial score on incomplete sequence (3/5 = 0.6)
        score = dataset.evaluate_response("Numbers: 1 2 3", [1, 2, 3, 4, 5], "math_calc")
        self.assertEqual(score, 0.6, "Should get partial score for partial sequence")
    
    def test_longbook_qa_eng_evaluation(self):
        """Test English long book QA evaluation - F1 scoring"""
        dataset = self._create_dataset("longbook_qa_eng", [
            {"id": "book_1", "context": "long book", "input": "Who is the protagonist?", "answer": ["Alice", "main character"]}
        ])
        
        # Should calculate F1 score with token overlap
        score = dataset.evaluate_response("Alice is the main character of the story", ["Alice", "main character"], "longbook_qa_eng")
        self.assertGreater(score, 0.4, "Should have decent F1 score for good overlap")
        
        # Should handle partial matches
        score = dataset.evaluate_response("The protagonist Alice appears frequently", ["Alice", "main character"], "longbook_qa_eng")
        self.assertGreater(score, 0.3, "Should have partial F1 score")
        
        # Should fail on no overlap
        score = dataset.evaluate_response("Bob is important", ["Alice", "main character"], "longbook_qa_eng")
        self.assertEqual(score, 0.0, "Should fail with no token overlap")
    
    def test_longbook_choice_eng_evaluation(self):
        """Test English long book choice evaluation - pattern matching"""
        dataset = self._create_dataset("longbook_choice_eng", [
            {"id": "choice_1", "context": "long book", "input": "Choose A, B, C, or D", "answer": ["B"]}
        ])
        
        # Should extract last A-D choice
        score = dataset.evaluate_response("First I thought A, but the answer is B", ["B"], "longbook_choice_eng")
        self.assertEqual(score, 1.0, "Should use last choice B")
        
        # Should fail on wrong choice
        score = dataset.evaluate_response("The answer is C", ["B"], "longbook_choice_eng")
        self.assertEqual(score, 0.0, "Should fail on choice C vs B")
        
        # Should fail without clear choice
        score = dataset.evaluate_response("I'm not sure about this", ["B"], "longbook_choice_eng")
        self.assertEqual(score, 0.0, "Should fail without A-D choice")
    
    def test_longdialogue_qa_eng_evaluation(self):
        """Test English long dialogue QA evaluation - substring matching"""
        dataset = self._create_dataset("longdialogue_qa_eng", [
            {"id": "dialogue_1", "context": "long dialogue", "input": "Who spoke?", "answer": ["JOHN"]}
        ])
        
        # Should match speaker name (case insensitive)
        score = dataset.evaluate_response("JOHN said hello to everyone", ["JOHN"], "longdialogue_qa_eng")
        self.assertEqual(score, 1.0, "Should match JOHN in response")
        
        # Should handle lowercase in response
        score = dataset.evaluate_response("john was the speaker", ["JOHN"], "longdialogue_qa_eng")
        self.assertEqual(score, 1.0, "Should match case-insensitive")
        
        # Should fail on wrong speaker
        score = dataset.evaluate_response("MARY was talking", ["JOHN"], "longdialogue_qa_eng")
        self.assertEqual(score, 0.0, "Should fail on wrong speaker")


class TestLongBenchEvaluation(unittest.TestCase):
    """Test LongBench-specific evaluation methods (22+ evaluators)"""
    
    def setUp(self):
        """Set up test data"""
        self.temp_files = []
    
    def tearDown(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            if temp_file.exists():
                temp_file.unlink()
    
    def _create_dataset(self, task_type: str, mock_data: list):
        """Helper to create dataset with mock data"""
        temp_file = Path(tempfile.mktemp(suffix='.jsonl'))
        with open(temp_file, 'w') as f:
            for item in mock_data:
                f.write(json.dumps(item) + '\n')
        self.temp_files.append(temp_file)
        
        config = BenchmarkConfig(
            name="longbench",
            task_type=task_type, 
            data_path=temp_file
        )
        return SmartMemoryBenchmarkDataset(config)
    
    def test_narrativeqa_evaluation(self):
        """Test narrative QA evaluation - F1 scoring with normalization"""
        dataset = self._create_dataset("narrativeqa", [
            {"id": "narrative_1", "context": "story", "input": "What happened?", "answer": "The wizard cast a spell"}
        ])
        
        # Should calculate F1 with token normalization
        score = dataset.evaluate_response("A wizard cast his spell", "The wizard cast a spell", "narrativeqa")
        self.assertGreater(score, 0.7, "Should have high F1 for similar tokens")
        
        # Should normalize articles and punctuation
        score = dataset.evaluate_response("wizard cast spell", "The wizard cast a spell", "narrativeqa")
        self.assertGreater(score, 0.8, "Should normalize articles")
        
        # Should fail on no overlap
        score = dataset.evaluate_response("A dragon appeared", "The wizard cast a spell", "narrativeqa")
        self.assertEqual(score, 0.0, "Should fail with no token overlap")
    
    def test_gov_report_evaluation(self):
        """Test government report evaluation - ROUGE scoring"""
        dataset = self._create_dataset("gov_report", [
            {"id": "gov_1", "context": "long report", "input": "Summarize", "answer": "Government spending increased"}
        ])
        
        # Should calculate ROUGE-L score
        score = dataset.evaluate_response("The government increased spending", "Government spending increased", "gov_report")
        self.assertGreater(score, 0.5, "Should have decent ROUGE score for similar content")
        
        # Should handle longer summaries
        score = dataset.evaluate_response("The federal government significantly increased spending", "Government spending increased", "gov_report")
        self.assertGreater(score, 0.3, "Should handle longer text")
        
        # Should fail on unrelated content
        score = dataset.evaluate_response("Weather was sunny", "Government spending increased", "gov_report")
        self.assertLess(score, 0.1, "Should have low ROUGE for unrelated content")
    
    def test_trec_classification_evaluation(self):
        """Test TREC classification evaluation - category matching"""
        dataset = self._create_dataset("trec", [
            {"id": "trec_1", "context": "question", "input": "Classify", "answer": "location"}
        ])
        
        # Should match category in response
        score = dataset.evaluate_response("This is about location", "location", "trec")
        self.assertEqual(score, 1.0, "Should match location category")
        
        # Should be case insensitive
        score = dataset.evaluate_response("LOCATION based question", "location", "trec")
        self.assertEqual(score, 1.0, "Should match case-insensitive")
        
        # Should fail on wrong category
        score = dataset.evaluate_response("This is about time", "location", "trec")
        self.assertEqual(score, 0.0, "Should fail on wrong category")
    
    def test_passage_count_evaluation(self):
        """Test passage counting evaluation - number extraction"""
        dataset = self._create_dataset("passage_count", [
            {"id": "count_1", "context": "passages", "input": "How many?", "answer": "5"}
        ])
        
        # Should extract first number from response
        score = dataset.evaluate_response("There are 5 passages in total", "5", "passage_count")
        self.assertEqual(score, 1.0, "Should extract number 5")
        
        # Should handle different formats
        score = dataset.evaluate_response("Count: 5", "5", "passage_count")
        self.assertEqual(score, 1.0, "Should extract formatted number")
        
        # Should fail on wrong count
        score = dataset.evaluate_response("I found 3 passages", "5", "passage_count")
        self.assertEqual(score, 0.0, "Should fail on wrong count")
        
        # Should fail on no numbers
        score = dataset.evaluate_response("Many passages exist", "5", "passage_count")
        self.assertEqual(score, 0.0, "Should fail when no number found")
    
    def test_passage_retrieval_evaluation(self):
        """Test passage retrieval evaluation - substring matching"""  
        dataset = self._create_dataset("passage_retrieval_en", [
            {"id": "retrieval_1", "context": "passages", "input": "Find passage", "answer": "passage_id_123"}
        ])
        
        # Should find answer in response
        score = dataset.evaluate_response("The relevant passage is passage_id_123", "passage_id_123", "passage_retrieval_en")
        self.assertEqual(score, 1.0, "Should find passage ID in response")
        
        # Should be case insensitive
        score = dataset.evaluate_response("PASSAGE_ID_123 contains the answer", "passage_id_123", "passage_retrieval_en")  
        self.assertEqual(score, 1.0, "Should match case-insensitive")
        
        # Should fail on wrong passage
        score = dataset.evaluate_response("passage_id_456 is relevant", "passage_id_123", "passage_retrieval_en")
        self.assertEqual(score, 0.0, "Should fail on wrong passage ID")
    
    def test_code_similarity_evaluation(self):
        """Test code similarity evaluation - token-based similarity"""
        dataset = self._create_dataset("lcc", [
            {"id": "code_1", "context": "code", "input": "Similar code?", "answer": "def func(x): return x + 1"}
        ])
        
        # Should calculate token-based similarity
        score = dataset.evaluate_response("def func(x): return x + 1", "def func(x): return x + 1", "lcc")
        self.assertEqual(score, 1.0, "Should have perfect similarity")
        
        # Should handle partial similarity  
        score = dataset.evaluate_response("def func(y): return y + 1", "def func(x): return x + 1", "lcc")
        self.assertGreater(score, 0.4, "Should have decent similarity with variable rename")
        
        # Should fail on very different code
        score = dataset.evaluate_response("print('hello')", "def func(x): return x + 1", "lcc")
        self.assertLess(score, 0.3, "Should have low similarity for different code")


class TestChineseEvaluation(unittest.TestCase):
    """Test Chinese-specific evaluation methods"""
    
    def setUp(self):
        """Set up test data"""
        self.temp_files = []
    
    def tearDown(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            if temp_file.exists():
                temp_file.unlink()
    
    def _create_dataset(self, task_type: str, mock_data: list):
        """Helper to create dataset with mock data"""
        temp_file = Path(tempfile.mktemp(suffix='.jsonl'))
        with open(temp_file, 'w') as f:
            for item in mock_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        self.temp_files.append(temp_file)
        
        config = BenchmarkConfig(
            name="longbench",
            task_type=task_type,
            data_path=temp_file
        )
        return SmartMemoryBenchmarkDataset(config)
    
    def test_chinese_qa_evaluation(self):
        """Test Chinese QA evaluation - character-level F1"""
        dataset = self._create_dataset("multifieldqa_zh", [
            {"id": "zh_1", "context": "中文文本", "input": "问题", "answer": "这是答案"}
        ])
        
        # Should calculate character-level F1
        score = dataset.evaluate_response("这是答案", "这是答案", "multifieldqa_zh")
        self.assertEqual(score, 1.0, "Should have perfect character match")
        
        # Should handle partial character overlap
        score = dataset.evaluate_response("这是部分答案", "这是答案", "multifieldqa_zh")
        self.assertGreater(score, 0.5, "Should have partial character overlap")
        
        # Should fail on no overlap
        score = dataset.evaluate_response("完全不同", "这是答案", "multifieldqa_zh")
        self.assertEqual(score, 0.0, "Should fail with no character overlap")
    
    def test_chinese_rouge_evaluation(self):
        """Test Chinese ROUGE evaluation - character-level overlap"""
        dataset = self._create_dataset("dureader", [
            {"id": "zh_rouge_1", "context": "中文文档", "input": "总结", "answer": "政府增加支出"}
        ])
        
        # Should calculate character-level overlap
        score = dataset.evaluate_response("政府增加了支出", "政府增加支出", "dureader")
        self.assertGreater(score, 0.8, "Should have high character overlap")
        
        # Should handle different text lengths
        score = dataset.evaluate_response("政府", "政府增加支出", "dureader")
        self.assertGreater(score, 0.2, "Should have some overlap")
        
        # Should fail on no overlap
        score = dataset.evaluate_response("天气晴朗", "政府增加支出", "dureader")
        self.assertEqual(score, 0.0, "Should fail with no character overlap")


class TestEvaluationRouting(unittest.TestCase):
    """Test that SmartDataset routes to correct evaluation methods"""
    
    def setUp(self):
        """Set up test data"""
        self.temp_files = []
    
    def tearDown(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            if temp_file.exists():
                temp_file.unlink()
    
    def _create_dataset(self, benchmark_name: str, task_type: str, mock_data: list):
        """Helper to create dataset with mock data"""
        temp_file = Path(tempfile.mktemp(suffix='.jsonl'))
        with open(temp_file, 'w') as f:
            for item in mock_data:
                f.write(json.dumps(item) + '\n')
        self.temp_files.append(temp_file)
        
        config = BenchmarkConfig(
            name=benchmark_name,
            task_type=task_type,
            data_path=temp_file
        )
        return SmartMemoryBenchmarkDataset(config)
    
    def test_task_routing_infinitebench(self):
        """Test that InfiniteBench tasks route to correct evaluators"""
        dataset = self._create_dataset("infinitebench", "passkey", [
            {"id": "passkey_1", "context": "text", "input": "passkey?", "answer": "12345"}
        ])
        
        # Should route passkey to passkey-specific evaluator (not exact match)
        score = dataset.evaluate_response("The passkey is 12345", "12345", "passkey")
        self.assertEqual(score, 1.0, "Should use passkey extraction, not exact match")
    
    def test_task_routing_longbench(self):
        """Test that LongBench tasks route to correct evaluators"""
        dataset = self._create_dataset("longbench", "narrativeqa", [
            {"id": "qa_1", "context": "story", "input": "what?", "answer": "The wizard cast a spell"}
        ])
        
        # Should route to F1 scoring, not exact match
        score = dataset.evaluate_response("wizard cast spell", "The wizard cast a spell", "narrativeqa")
        self.assertGreater(score, 0.0, "Should use F1 scoring, not exact match")
    
    def test_locomo_routing_preserved(self):
        """Test that LoCoMo routing is preserved (already working)"""
        dataset = self._create_dataset("locomo", "conversational_qa", [
            {"conversations": {"session_1": {"1": "hello", "2": "world"}}, "qa": [{"question": "test", "answer": "hello world"}]}
        ])
        
        # Should continue using LoCoMo F1 scoring
        score = dataset.evaluate_response("hello world", "hello world", "conversational_qa")
        self.assertEqual(score, 1.0, "Should preserve LoCoMo F1 scoring")


if __name__ == '__main__':
    unittest.main()