"""
RED PHASE - Failing test that reproduces the metadata bug in MTBench101Dataset.evaluate_response

This test demonstrates that MTBench101Dataset._parse_conversations creates BenchmarkItem
objects with metadata that doesn't contain the 'history' and 'ref_answer' fields 
expected by the evaluate_response method.

Bug: ground_truth must have metadata attribute with required fields
File: emotion_memory_experiments/datasets/mtbench101.py:374-375
"""

import json
import tempfile
import pytest
from pathlib import Path

from emotion_memory_experiments.data_models import BenchmarkConfig
from emotion_memory_experiments.datasets.mtbench101 import MTBench101Dataset


class TestMTBench101MetadataBug:
    """Test class that reproduces the exact metadata bug"""

    def test_evaluation_with_parsed_benchmark_item_fails(self):
        """
        RED PHASE - This test reproduces the exact bug.
        
        The bug occurs when:
        1. _parse_conversations creates BenchmarkItem with metadata containing:
           - task, conversation_id, user_messages, assistant_messages, full_conversation
        2. evaluate_response is called with that BenchmarkItem as ground_truth
        3. evaluate_response expects metadata to contain 'history' and 'ref_answer' fields
        4. ValueError is raised: "ground_truth must have metadata attribute with required fields"
        """
        # Create sample MTBench101 conversation data
        sample_data = [
            {
                "task": "CM",
                "id": 1145,
                "history": [
                    {"user": "I want to buy a new laptop", "bot": "What's your budget?"},
                    {"user": "Under $1500", "bot": "I recommend the Dell XPS 15"}
                ]
            }
        ]
        
        # Create temporary data file
        with tempfile.NamedTemporaryFile(mode='w', suffix='_CM.jsonl', delete=False) as f:
            for conversation in sample_data:
                json.dump(conversation, f)
                f.write('\n')
            temp_file_path = Path(f.name)
        
        try:
            # Create MTBench101Dataset with the sample data
            config = BenchmarkConfig(
                name="mtbench101",
                task_type="CM",
                data_path=temp_file_path,
                sample_limit=None,
                augmentation_config=None,
                enable_auto_truncation=False,
                truncation_strategy="right",
                preserve_ratio=0.8,
                llm_eval_config={"model": "gpt-4o", "temperature": 0.0}
            )
            
            dataset = MTBench101Dataset(config, prompt_wrapper=None)
            
            # Get the first parsed item
            benchmark_item = dataset.items[0]
            
            # Verify the metadata structure created by _parse_conversations
            assert benchmark_item.metadata is not None, "BenchmarkItem should have metadata"
            assert "task" in benchmark_item.metadata
            assert "conversation_id" in benchmark_item.metadata
            assert "user_messages" in benchmark_item.metadata
            assert "assistant_messages" in benchmark_item.metadata
            assert "full_conversation" in benchmark_item.metadata
            
            # Show what's missing - the fields expected by evaluate_response
            assert "history" not in benchmark_item.metadata, "metadata missing 'history' field expected by evaluate_response"
            assert "ref_answer" not in benchmark_item.metadata, "metadata missing 'ref_answer' field expected by evaluate_response"
            
            # This call should fail with "ground_truth must have metadata attribute with required fields"
            # because benchmark_item.ground_truth is a string, not an object with metadata
            test_response = "I can help you with laptop recommendations"
            
            with pytest.raises(ValueError, match="ground_truth must have metadata attribute with required fields"):
                dataset.evaluate_response(
                    response=test_response,
                    ground_truth=benchmark_item.ground_truth,  # This is just a string
                    task_name="CM"
                )
                
        finally:
            # Cleanup
            if temp_file_path.exists():
                temp_file_path.unlink()

    def test_evaluation_expects_ground_truth_object_with_metadata(self):
        """
        RED PHASE - This test shows what evaluate_response actually expects.
        
        The evaluate_response method expects ground_truth to be an object with:
        - .metadata attribute (not None)  
        - .metadata["history"] field
        - .metadata["ref_answer"] field (optional)
        
        But _parse_conversations creates BenchmarkItem where:
        - ground_truth is the string response (not an object with metadata)
        - The BenchmarkItem itself has metadata, but ground_truth doesn't
        """
        sample_data = [
            {
                "task": "AR",
                "id": 2001,
                "history": [
                    {"user": "What was the capital mentioned earlier?", "bot": "Paris was the capital we discussed"}
                ]
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='_AR.jsonl', delete=False) as f:
            for conversation in sample_data:
                json.dump(conversation, f)
                f.write('\n')
            temp_file_path = Path(f.name)
        
        try:
            config = BenchmarkConfig(
                name="mtbench101",
                task_type="AR", 
                data_path=temp_file_path,
                sample_limit=None,
                augmentation_config=None,
                enable_auto_truncation=False,
                truncation_strategy="right",
                preserve_ratio=0.8,
                llm_eval_config={"model": "gpt-4o", "temperature": 0.0}
            )
            
            dataset = MTBench101Dataset(config, prompt_wrapper=None)
            benchmark_item = dataset.items[0]
            
            # Show what _parse_conversations actually creates
            assert isinstance(benchmark_item.ground_truth, str), "ground_truth is a string, not an object with metadata"
            assert benchmark_item.ground_truth == "Paris was the capital we discussed"
            assert not hasattr(benchmark_item.ground_truth, 'metadata'), "String ground_truth has no metadata attribute"
            
            # Show what evaluate_response expects but doesn't get
            test_response = "The capital was Paris"
            
            with pytest.raises(ValueError, match="ground_truth must have metadata attribute with required fields"):
                dataset.evaluate_response(
                    response=test_response,
                    ground_truth=benchmark_item.ground_truth,  # String has no .metadata
                    task_name="AR"
                )
                
        finally:
            if temp_file_path.exists():
                temp_file_path.unlink()

    def test_evaluation_works_with_properly_structured_ground_truth(self):
        """
        This test shows what structure evaluate_response actually expects to work.
        
        This is a reference test to demonstrate the expected interface.
        When we fix the bug, the dataset should either:
        1. Modify ground_truth objects to have the expected metadata, or
        2. Modify evaluate_response to extract metadata from the BenchmarkItem differently
        """
        # Create a properly structured ground_truth object with expected metadata
        class ProperlyStructuredGroundTruth:
            def __init__(self, answer, history, ref_answer=None):
                self.answer = answer
                self.metadata = {
                    "history": history,
                    "ref_answer": ref_answer
                }
                
        # Create sample data
        sample_data = [
            {
                "task": "SI",
                "id": 3001,
                "history": [
                    {"user": "I need help", "bot": "What kind of help do you need?"},
                    {"user": "Please provide details", "bot": "Here are the details you requested"}
                ]
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='_SI.jsonl', delete=False) as f:
            for conversation in sample_data:
                json.dump(conversation, f)
                f.write('\n')
            temp_file_path = Path(f.name)
        
        try:
            config = BenchmarkConfig(
                name="mtbench101",
                task_type="SI",
                data_path=temp_file_path,
                sample_limit=None,
                augmentation_config=None,
                enable_auto_truncation=False,
                truncation_strategy="right",
                preserve_ratio=0.8,
                llm_eval_config={"model": "gpt-4o", "temperature": 0.0}
            )
            
            dataset = MTBench101Dataset(config, prompt_wrapper=None)
            
            # Create the history string that evaluate_response expects
            history_string = "User: I need help\nAssistant: What kind of help do you need?\nUser: Please provide details\nAssistant: Here are the details you requested"
            
            # Create properly structured ground truth
            proper_ground_truth = ProperlyStructuredGroundTruth(
                answer="Here are the details you requested",
                history=history_string,
                ref_answer=None
            )
            
            test_response = "I can provide you with the information you need"
            
            # This should NOT raise the metadata error (though it might fail for other reasons like LLM API calls)
            try:
                score = dataset.evaluate_response(
                    response=test_response,
                    ground_truth=proper_ground_truth,
                    task_name="SI"
                )
                assert isinstance(score, float)
                assert 0.0 <= score <= 10.0
            except Exception as e:
                # If it fails, it should NOT be the metadata error
                assert "ground_truth must have metadata attribute with required fields" not in str(e)
                
        finally:
            if temp_file_path.exists():
                temp_file_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])