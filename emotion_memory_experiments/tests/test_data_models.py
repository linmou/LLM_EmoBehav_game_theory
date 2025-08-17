"""
Unit tests for data models.
"""
import unittest
from pathlib import Path

from ..data_models import (
    ResultRecord, BenchmarkConfig, ExperimentConfig, BenchmarkItem, DEFAULT_GENERATION_CONFIG
)


class TestDataModels(unittest.TestCase):
    """Test data model classes"""
    
    def test_result_record_creation(self):
        """Test ResultRecord creation and attributes"""
        result = ResultRecord(
            emotion="anger",
            intensity=1.0,
            item_id="test_item",
            task_name="passkey",
            prompt="What is the passkey?",
            response="12345",
            ground_truth="12345",
            score=1.0,
            metadata={"test": "data"}
        )
        
        self.assertEqual(result.emotion, "anger")
        self.assertEqual(result.intensity, 1.0)
        self.assertEqual(result.item_id, "test_item")
        self.assertEqual(result.task_name, "passkey")
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.metadata["test"], "data")
    
    def test_result_record_optional_metadata(self):
        """Test ResultRecord with no metadata"""
        result = ResultRecord(
            emotion="neutral",
            intensity=0.0,
            item_id=1,
            task_name="kv_retrieval",
            prompt="test prompt",
            response="test response",
            ground_truth="test ground truth",
            score=0.5
        )
        
        self.assertIsNone(result.metadata)
    
    def test_benchmark_config_creation(self):
        """Test BenchmarkConfig creation"""
        config = BenchmarkConfig(
            name="infinitebench",
            data_path=Path("test_data.jsonl"),
            task_type="passkey",
            evaluation_method="get_score_one_passkey",
            sample_limit=10
        )
        
        self.assertEqual(config.name, "infinitebench")
        self.assertEqual(config.task_type, "passkey")
        self.assertEqual(config.sample_limit, 10)
        self.assertIsInstance(config.data_path, Path)
    
    def test_benchmark_config_optional_sample_limit(self):
        """Test BenchmarkConfig with no sample limit"""
        config = BenchmarkConfig(
            name="locomo",
            data_path=Path("locomo.json"),
            task_type="conversational_qa",
            evaluation_method="custom_eval"
        )
        
        self.assertIsNone(config.sample_limit)
    
    def test_experiment_config_creation(self):
        """Test ExperimentConfig creation"""
        benchmark_config = BenchmarkConfig(
            name="infinitebench",
            data_path=Path("test.jsonl"),
            task_type="passkey",
            evaluation_method="test"
        )
        
        config = ExperimentConfig(
            model_path="/path/to/model",
            emotions=["anger", "happiness"],
            intensities=[0.5, 1.0],
            benchmark=benchmark_config,
            output_dir="test_output",
            batch_size=4,
            generation_config={"temperature": 0.7}
        )
        
        self.assertEqual(config.model_path, "/path/to/model")
        self.assertEqual(len(config.emotions), 2)
        self.assertEqual(len(config.intensities), 2)
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.generation_config["temperature"], 0.7)
    
    def test_experiment_config_defaults(self):
        """Test ExperimentConfig with default values"""
        benchmark_config = BenchmarkConfig(
            name="test",
            data_path=Path("test.jsonl"),
            task_type="test",
            evaluation_method="test"
        )
        
        config = ExperimentConfig(
            model_path="/path/to/model",
            emotions=["anger"],
            intensities=[1.0],
            benchmark=benchmark_config,
            output_dir="output"
        )
        
        self.assertEqual(config.batch_size, 4)  # Default
        self.assertIsNone(config.generation_config)
    
    def test_benchmark_item_creation(self):
        """Test BenchmarkItem creation"""
        item = BenchmarkItem(
            id="item_1",
            input_text="What is the answer?",
            context="Some context text",
            ground_truth="42",
            metadata={"source": "test"}
        )
        
        self.assertEqual(item.id, "item_1")
        self.assertEqual(item.input_text, "What is the answer?")
        self.assertEqual(item.context, "Some context text")
        self.assertEqual(item.ground_truth, "42")
        self.assertEqual(item.metadata["source"], "test")
    
    def test_benchmark_item_optional_fields(self):
        """Test BenchmarkItem with only required fields"""
        item = BenchmarkItem(
            id=123,
            input_text="Test question"
        )
        
        self.assertEqual(item.id, 123)
        self.assertEqual(item.input_text, "Test question")
        self.assertIsNone(item.context)
        self.assertIsNone(item.ground_truth)
        self.assertIsNone(item.metadata)
    
    def test_default_generation_config(self):
        """Test default generation config values"""
        self.assertIn("temperature", DEFAULT_GENERATION_CONFIG)
        self.assertIn("max_new_tokens", DEFAULT_GENERATION_CONFIG)
        self.assertIn("do_sample", DEFAULT_GENERATION_CONFIG)
        self.assertIn("top_p", DEFAULT_GENERATION_CONFIG)
        
        # Check reasonable defaults
        self.assertEqual(DEFAULT_GENERATION_CONFIG["temperature"], 0.1)
        self.assertEqual(DEFAULT_GENERATION_CONFIG["max_new_tokens"], 100)
        self.assertFalse(DEFAULT_GENERATION_CONFIG["do_sample"])


if __name__ == '__main__':
    unittest.main()