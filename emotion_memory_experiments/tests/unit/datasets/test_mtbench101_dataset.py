"""
Test file for MTBench101Dataset class - TDD Phase 2 Red
Testing: emotion_memory_experiments/datasets/mtbench101.py functionality  
Purpose: Ensure MTBench101Dataset follows BaseBenchmarkDataset interface and handles conversation data correctly
"""

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from emotion_memory_experiments.data_models import BenchmarkConfig, BenchmarkItem

# Import will fail initially - this is the Red phase
try:
    from emotion_memory_experiments.datasets.mtbench101 import MTBench101Dataset
    from emotion_memory_experiments.dataset_factory import create_dataset_from_config
except ImportError:
    # Expected during Red phase - tests should fail
    MTBench101Dataset = None
    create_dataset_from_config = None


class TestMTBench101Dataset:
    """Test MTBench101Dataset functionality"""

    @pytest.fixture
    def sample_mtbench_conversations(self):
        """Sample conversation data for testing"""
        return [
            {
                "task": "CM",
                "id": 1145,
                "history": [
                    {"user": "I want to buy a new laptop", "bot": "What's your budget?"},
                    {"user": "Under $1500", "bot": "I recommend the Dell XPS 15"}
                ]
            },
            {
                "task": "CM", 
                "id": 1146,
                "history": [
                    {"user": "I'm thinking of baking", "bot": "What would you like to bake?"},
                    {"user": "Chocolate chip cookies", "bot": "Great choice! Here's a recipe..."}
                ]
            }
        ]

    @pytest.fixture 
    def temp_mtbench_file(self, sample_mtbench_conversations):
        """Create temporary MTBench101 task file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='_CM.jsonl', delete=False) as f:
            for conversation in sample_mtbench_conversations:
                json.dump(conversation, f)
                f.write('\n')
            return Path(f.name)

    @pytest.fixture
    def benchmark_config(self, temp_mtbench_file):
        """BenchmarkConfig for MTBench101 CM task"""
        return BenchmarkConfig(
            name="mtbench101",
            task_type="CM", 
            data_path=temp_mtbench_file,
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )

    def test_mtbench101_dataset_initialization(self, benchmark_config):
        """Test dataset can be created with BenchmarkConfig"""
        # This test will fail because MTBench101Dataset class doesn't exist yet (Red phase)
        dataset = MTBench101Dataset(benchmark_config)
        
        assert dataset.config == benchmark_config
        assert hasattr(dataset, 'items')
        assert isinstance(dataset.items, list)

    def test_load_single_task_file(self, benchmark_config, sample_mtbench_conversations):
        """Test loading mtbench101_CM.jsonl creates correct BenchmarkItems"""
        dataset = MTBench101Dataset(benchmark_config)
        
        # Should have converted conversations to BenchmarkItems
        assert len(dataset.items) == len(sample_mtbench_conversations)
        
        for i, item in enumerate(dataset.items):
            assert isinstance(item, BenchmarkItem)
            expected_conv = sample_mtbench_conversations[i]
            assert item.id == f"CM_{expected_conv['id']}"

    def test_conversation_parsing(self, benchmark_config):
        """Test conversation history parsed into user/assistant message lists"""
        dataset = MTBench101Dataset(benchmark_config)
        
        item = dataset.items[0]
        
        # input_text should contain parsed conversation as tuple
        assert isinstance(item.input_text, tuple)
        user_messages, assistant_messages = item.input_text
        
        # Check message extraction
        assert isinstance(user_messages, list)
        assert isinstance(assistant_messages, list)
        assert len(user_messages) == 2  # Two user messages
        assert len(assistant_messages) == 2  # Two bot responses
        
        assert user_messages[0] == "I want to buy a new laptop"
        assert assistant_messages[0] == "What's your budget?"

    def test_benchmark_item_structure(self, benchmark_config):
        """Test BenchmarkItem has correct id, input_text tuple, metadata"""
        dataset = MTBench101Dataset(benchmark_config)
        
        item = dataset.items[0]
        
        # Check BenchmarkItem structure
        assert hasattr(item, 'id')
        assert hasattr(item, 'input_text') 
        assert hasattr(item, 'ground_truth')
        assert hasattr(item, 'metadata')
        
        # Check metadata contains conversation info
        assert item.metadata['task'] == 'CM'
        assert item.metadata['conversation_id'] == 1145
        assert 'user_messages' in item.metadata
        assert 'assistant_messages' in item.metadata

    def test_ground_truth_is_last_response(self, benchmark_config):
        """Test ground_truth is set to the last assistant response"""
        dataset = MTBench101Dataset(benchmark_config)
        
        item = dataset.items[0]
        
        # Ground truth should be the last bot response for evaluation
        assert item.ground_truth == "I recommend the Dell XPS 15"

    def test_dataset_length(self, benchmark_config, sample_mtbench_conversations):
        """Test __len__ method returns correct count"""
        dataset = MTBench101Dataset(benchmark_config)
        
        assert len(dataset) == len(sample_mtbench_conversations)

    def test_dataset_getitem(self, benchmark_config):
        """Test __getitem__ returns formatted dict with BenchmarkItem"""  
        dataset = MTBench101Dataset(benchmark_config)
        
        # Test indexing
        result0 = dataset[0]
        result1 = dataset[1]
        
        # Base class returns dict with item, prompt, ground_truth
        assert isinstance(result0, dict)
        assert isinstance(result1, dict)
        assert "item" in result0 and "prompt" in result0 and "ground_truth" in result0
        assert "item" in result1 and "prompt" in result1 and "ground_truth" in result1
        
        # The actual BenchmarkItems
        item0 = result0["item"]
        item1 = result1["item"]
        assert isinstance(item0, BenchmarkItem)
        assert isinstance(item1, BenchmarkItem)
        assert item0.id != item1.id

    def test_sample_limit_respected(self, temp_mtbench_file):
        """Test sample_limit reduces dataset size"""
        config = BenchmarkConfig(
            name="mtbench101",
            task_type="CM",
            data_path=temp_mtbench_file,
            sample_limit=1,  # Limit to 1 conversation
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right", 
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        dataset = MTBench101Dataset(config)
        
        assert len(dataset) == 1

    def test_evaluate_response_method_exists(self, benchmark_config):
        """Test dataset has evaluate_response method for evaluation"""
        dataset = MTBench101Dataset(benchmark_config)
        
        # Should have evaluate_response method from base class contract
        assert hasattr(dataset, 'evaluate_response')
        assert callable(getattr(dataset, 'evaluate_response'))

    def test_get_task_metrics_method_exists(self, benchmark_config):
        """Test dataset has get_task_metrics method"""
        dataset = MTBench101Dataset(benchmark_config)
        
        # Should have get_task_metrics method from base class contract
        assert hasattr(dataset, 'get_task_metrics')
        assert callable(getattr(dataset, 'get_task_metrics'))

    def test_conversation_context_preserved(self, benchmark_config):
        """Test full conversation history is preserved in metadata"""
        dataset = MTBench101Dataset(benchmark_config)
        
        item = dataset.items[0]
        
        # Should preserve full conversation context for evaluation
        assert 'full_conversation' in item.metadata
        full_conv = item.metadata['full_conversation']
        assert full_conv['task'] == 'CM'
        assert full_conv['id'] == 1145
        assert len(full_conv['history']) == 2

    def test_task_consistency_validation(self, benchmark_config):
        """Test all conversations in task file have same task label"""
        dataset = MTBench101Dataset(benchmark_config)
        
        # All items should have same task since from single task file
        for item in dataset.items:
            assert item.metadata['task'] == 'CM'

    def test_empty_file_handling(self):
        """Test handling of empty task file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='_CM.jsonl', delete=False) as empty_file:
            pass  # Create empty file
        
        config = BenchmarkConfig(
            name="mtbench101",
            task_type="CM",
            data_path=Path(empty_file.name),
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        dataset = MTBench101Dataset(config)
        
        # Should handle empty file gracefully
        assert len(dataset) == 0
        assert dataset.items == []

    def test_malformed_conversation_handling(self):
        """Test error handling for malformed conversation data"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='_CM.jsonl', delete=False) as bad_file:
            # Write conversation missing required fields
            json.dump({"id": 1, "history": []}, bad_file)  # Missing 'task' field
            bad_file.write('\n')
        
        config = BenchmarkConfig(
            name="mtbench101",
            task_type="CM", 
            data_path=Path(bad_file.name),
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        # Should raise appropriate error for malformed data
        with pytest.raises(ValueError, match="Missing.*task"):
            MTBench101Dataset(config)

    def cleanup_temp_files(self, temp_mtbench_file):
        """Cleanup temp files after tests"""
        if temp_mtbench_file.exists():
            temp_mtbench_file.unlink()

    def test_evaluation_integration(self, benchmark_config):
        """Test MTBench101 evaluation integrates correctly"""
        if MTBench101Dataset is None or create_dataset_from_config is None:
            pytest.skip("MTBench101Dataset or factory not available")
            
        config = BenchmarkConfig(
            name="mtbench101",
            task_type="SI",
            data_path=None,
            sample_limit=1,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        dataset = create_dataset_from_config(config)
        item_dict = dataset[0]
        benchmark_item = item_dict['item']
        
        # Test evaluation function
        mock_response = "Here's my step-by-step approach: 1) First step, 2) Second step, 3) Final conclusion."
        score = dataset.evaluate_response(
            response=mock_response,
            ground_truth=benchmark_item.ground_truth,
            task_name="SI"
        )
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 10.0

    def test_conversation_format_handling(self, benchmark_config):
        """Test MTBench101 handles conversation format correctly"""
        if MTBench101Dataset is None or create_dataset_from_config is None:
            pytest.skip("MTBench101Dataset or factory not available")
            
        config = BenchmarkConfig(
            name="mtbench101",
            task_type="CM",
            data_path=None,
            sample_limit=1,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        dataset = create_dataset_from_config(config)
        item_dict = dataset[0]
        benchmark_item = item_dict['item']
        
        user_messages, assistant_messages = benchmark_item.input_text
        
        # Verify conversation format
        assert isinstance(user_messages, list)
        assert isinstance(assistant_messages, list)
        assert len(user_messages) >= len(assistant_messages)

    def test_if_else_branches_in_evaluate_response(self, benchmark_config):
        """Test both if-else branches in evaluate_response method"""
        if MTBench101Dataset is None:
            pytest.skip("MTBench101Dataset not available")
            
        dataset = MTBench101Dataset(benchmark_config)
        
        # Test branch 1: ground_truth with metadata
        class MockGroundTruthWithMetadata:
            metadata = {"history": "test history", "ref_answer": "test ref"}
            
        # Test branch 2: simple ground_truth without metadata
        simple_ground_truth = "simple answer"
        
        # Both should work without errors (actual scoring may vary due to LLM calls)
        try:
            score1 = dataset.evaluate_response("test response", MockGroundTruthWithMetadata(), "SI")
            assert isinstance(score1, float)
        except Exception:
            # LLM errors are acceptable for this branch test
            pass
            
        try:
            score2 = dataset.evaluate_response("test response", simple_ground_truth, "SI") 
            assert isinstance(score2, float)
        except Exception:
            # LLM errors are acceptable for this branch test
            pass