"""
Unit tests for the main experiment class.
"""
import unittest
from unittest.mock import patch, MagicMock, Mock
import pandas as pd
import tempfile
from pathlib import Path
import shutil

from ..experiment import EmotionMemoryExperiment
from ..data_models import ExperimentConfig, BenchmarkConfig
from .test_utils import (
    create_mock_experiment_config, MockRepControlPipeline,
    MockOutput, cleanup_temp_files
)


class TestEmotionMemoryExperiment(unittest.TestCase):
    """Test the main experiment class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dirs = []
        self.temp_files = []
        self.configs = []
    
    def tearDown(self):
        """Clean up temporary files and directories"""
        cleanup_temp_files(self.configs)
        
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except FileNotFoundError:
                pass
        
        for temp_file in self.temp_files:
            try:
                temp_file.unlink()
            except FileNotFoundError:
                pass
    
    @patch('emotion_memory_experiments.experiment.setup_model_and_tokenizer')
    @patch('emotion_memory_experiments.experiment.ModelLayerDetector')
    @patch('emotion_memory_experiments.experiment.load_emotion_readers')
    @patch('emotion_memory_experiments.experiment.get_pipeline')
    def test_experiment_initialization(self, mock_get_pipeline, mock_load_emotion_readers,
                                      mock_model_detector, mock_setup_model):
        """Test experiment initialization"""
        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_setup_model.return_value = (mock_model, mock_tokenizer, "chat")
        mock_model_detector.num_layers.return_value = 12
        mock_load_emotion_readers.return_value = {"anger": MagicMock(), "happiness": MagicMock()}
        mock_get_pipeline.return_value = MockRepControlPipeline(["test response"])
        
        # Create test config
        config = create_mock_experiment_config("passkey", 3)
        self.configs.append(config.benchmark)
        
        # Initialize experiment
        experiment = EmotionMemoryExperiment(config)
        
        # Verify initialization
        self.assertEqual(experiment.config, config)
        self.assertIsNotNone(experiment.logger)
        self.assertIsNotNone(experiment.benchmark_adapter)
        self.assertIsNotNone(experiment.emotion_rep_readers)
        self.assertIsNotNone(experiment.rep_control_pipeline)
        
        # Verify model setup was called correctly
        self.assertEqual(mock_setup_model.call_count, 2)  # Once for HF, once for vLLM
        mock_load_emotion_readers.assert_called_once()
        mock_get_pipeline.assert_called_once()
    
    @patch('emotion_memory_experiments.experiment.setup_model_and_tokenizer')
    @patch('emotion_memory_experiments.experiment.ModelLayerDetector')
    @patch('emotion_memory_experiments.experiment.load_emotion_readers')
    @patch('emotion_memory_experiments.experiment.get_pipeline')
    def test_process_emotion_condition_with_emotion(self, mock_get_pipeline, mock_load_emotion_readers,
                                                   mock_model_detector, mock_setup_model):
        """Test processing an emotion condition"""
        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_setup_model.return_value = (mock_model, mock_tokenizer, "chat")
        mock_model_detector.num_layers.return_value = 12
        
        # Mock emotion reader
        import numpy as np
        mock_rep_reader = MagicMock()
        # Create directions and signs for all hidden layers (-1 to -12)
        mock_rep_reader.directions = {layer: np.array([0.1, 0.2, 0.3]) for layer in range(-1, -13, -1)}
        mock_rep_reader.direction_signs = {layer: np.array([1, -1, 1]) for layer in range(-1, -13, -1)}
        mock_load_emotion_readers.return_value = {"anger": mock_rep_reader}
        
        # Mock pipeline with specific responses
        responses = ["12345", "67890", "11111"]
        mock_pipeline = MockRepControlPipeline(responses)
        mock_get_pipeline.return_value = mock_pipeline
        
        # Create test config
        config = create_mock_experiment_config("passkey", 3)
        self.configs.append(config.benchmark)
        
        # Initialize experiment
        experiment = EmotionMemoryExperiment(config)
        
        # Mock is_vllm to True since we're using MockOutput
        experiment.is_vllm = True
        
        # Get test data
        benchmark_data = experiment.benchmark_adapter.get_data()
        
        # Process emotion condition
        results = experiment._process_emotion_condition(
            benchmark_data, mock_rep_reader, "anger", 1.0
        )
        
        # Verify results
        self.assertEqual(len(results), 3)
        for i, result in enumerate(results):
            self.assertEqual(result.emotion, "anger")
            self.assertEqual(result.intensity, 1.0)
            self.assertEqual(result.task_name, "passkey")
            self.assertIn(responses[i], result.response)
            self.assertIsInstance(result.score, float)
    
    @patch('emotion_memory_experiments.experiment.setup_model_and_tokenizer')
    @patch('emotion_memory_experiments.experiment.ModelLayerDetector')
    @patch('emotion_memory_experiments.experiment.load_emotion_readers')
    @patch('emotion_memory_experiments.experiment.get_pipeline')
    def test_process_neutral_condition(self, mock_get_pipeline, mock_load_emotion_readers,
                                      mock_model_detector, mock_setup_model):
        """Test processing neutral baseline condition"""
        # Setup mocks (same as above)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_setup_model.return_value = (mock_model, mock_tokenizer, "chat")
        mock_model_detector.num_layers.return_value = 12
        mock_load_emotion_readers.return_value = {"anger": MagicMock()}
        
        responses = ["neutral_response"]
        mock_pipeline = MockRepControlPipeline(responses)
        mock_get_pipeline.return_value = mock_pipeline
        
        # Create test config
        config = create_mock_experiment_config("passkey", 1)
        self.configs.append(config.benchmark)
        
        # Initialize experiment
        experiment = EmotionMemoryExperiment(config)
        
        # Mock is_vllm to True since we're using MockOutput
        experiment.is_vllm = True
        
        # Get test data
        benchmark_data = experiment.benchmark_adapter.get_data()
        
        # Process neutral condition
        results = experiment._process_emotion_condition(
            benchmark_data, None, "neutral", 0.0
        )
        
        # Verify results
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.emotion, "neutral")
        self.assertEqual(result.intensity, 0.0)
        self.assertIn("neutral_response", result.response)
    
    @patch('emotion_memory_experiments.experiment.setup_model_and_tokenizer')
    @patch('emotion_memory_experiments.experiment.ModelLayerDetector')
    @patch('emotion_memory_experiments.experiment.load_emotion_readers')
    @patch('emotion_memory_experiments.experiment.get_pipeline')
    def test_save_results(self, mock_get_pipeline, mock_load_emotion_readers,
                         mock_model_detector, mock_setup_model):
        """Test saving experiment results"""
        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_setup_model.return_value = (mock_model, mock_tokenizer, "chat")
        mock_model_detector.num_layers.return_value = 12
        mock_load_emotion_readers.return_value = {"anger": MagicMock()}
        mock_get_pipeline.return_value = MockRepControlPipeline(["test"])
        
        # Create test config with temp output dir
        config = create_mock_experiment_config("passkey", 1)
        temp_dir = Path(tempfile.mkdtemp())
        config.output_dir = str(temp_dir)
        self.temp_dirs.append(temp_dir)
        self.configs.append(config.benchmark)
        
        # Initialize experiment
        experiment = EmotionMemoryExperiment(config)
        
        # Create mock results
        from ..data_models import ResultRecord
        results = [
            ResultRecord(
                emotion="anger",
                intensity=1.0,
                item_id="test_1",
                task_name="passkey",
                prompt="Test prompt",
                response="Test response",
                ground_truth="Test ground truth",
                score=0.8,
                metadata={"benchmark": "infinitebench"}
            ),
            ResultRecord(
                emotion="neutral",
                intensity=0.0,
                item_id="test_2",
                task_name="passkey",
                prompt="Test prompt 2",
                response="Test response 2",
                ground_truth="Test ground truth 2",
                score=0.6,
                metadata={"benchmark": "infinitebench"}
            )
        ]
        
        # Save results
        df = experiment._save_results(results)
        
        # Verify DataFrame
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertIn('emotion', df.columns)
        self.assertIn('score', df.columns)
        
        # Verify files were created
        self.assertTrue((experiment.output_dir / "detailed_results.csv").exists())
        self.assertTrue((experiment.output_dir / "summary_results.csv").exists())
        self.assertTrue((experiment.output_dir / "raw_results.json").exists())
    
    @patch('emotion_memory_experiments.experiment.setup_model_and_tokenizer')
    @patch('emotion_memory_experiments.experiment.ModelLayerDetector')
    @patch('emotion_memory_experiments.experiment.load_emotion_readers')
    @patch('emotion_memory_experiments.experiment.get_pipeline')
    def test_run_sanity_check(self, mock_get_pipeline, mock_load_emotion_readers,
                             mock_model_detector, mock_setup_model):
        """Test sanity check functionality"""
        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_setup_model.return_value = (mock_model, mock_tokenizer, "chat")
        mock_model_detector.num_layers.return_value = 12
        
        # Mock emotion reader
        import numpy as np
        mock_rep_reader = MagicMock()
        # Create directions and signs for all hidden layers (-1 to -12) 
        mock_rep_reader.directions = {layer: np.array([0.1, 0.2]) for layer in range(-1, -13, -1)}
        mock_rep_reader.direction_signs = {layer: np.array([1, -1]) for layer in range(-1, -13, -1)}
        mock_load_emotion_readers.return_value = {"anger": mock_rep_reader, "happiness": mock_rep_reader}
        
        mock_pipeline = MockRepControlPipeline(["response1", "response2", "response3"])
        mock_get_pipeline.return_value = mock_pipeline
        
        # Create test config
        config = create_mock_experiment_config("passkey", 10)  # Start with 10 items
        temp_dir = Path(tempfile.mkdtemp())
        config.output_dir = str(temp_dir)
        self.temp_dirs.append(temp_dir)
        self.configs.append(config.benchmark)
        
        # Initialize experiment
        experiment = EmotionMemoryExperiment(config)
        
        # Mock is_vllm to True since we're using MockOutput
        experiment.is_vllm = True
        
        # Run sanity check with 2 samples
        df = experiment.run_sanity_check(sample_limit=2)
        
        # Verify results
        self.assertIsInstance(df, pd.DataFrame)
        # Should have 2 items × 2 emotions × 2 intensities + 2 items × 1 neutral = 10 results
        expected_results = 2 * 2 * 2 + 2 * 1  
        self.assertEqual(len(df), expected_results)
        
        # Verify original config was restored
        self.assertEqual(config.benchmark.sample_limit, 10)
    
    @patch('emotion_memory_experiments.experiment.setup_model_and_tokenizer')
    @patch('emotion_memory_experiments.experiment.ModelLayerDetector')
    @patch('emotion_memory_experiments.experiment.load_emotion_readers')
    @patch('emotion_memory_experiments.experiment.get_pipeline')
    def test_evaluation_error_handling(self, mock_get_pipeline, mock_load_emotion_readers,
                                      mock_model_detector, mock_setup_model):
        """Test handling of evaluation errors"""
        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_setup_model.return_value = (mock_model, mock_tokenizer, "chat")
        mock_model_detector.num_layers.return_value = 12
        mock_load_emotion_readers.return_value = {"anger": MagicMock()}
        mock_get_pipeline.return_value = MockRepControlPipeline(["test"])
        
        # Create test config
        config = create_mock_experiment_config("passkey", 1)
        self.configs.append(config.benchmark)
        
        # Initialize experiment
        experiment = EmotionMemoryExperiment(config)
        
        # Mock is_vllm to True since we're using MockOutput
        experiment.is_vllm = True
        
        # Mock adapter to raise evaluation error
        with patch.object(experiment.benchmark_adapter, 'evaluate_response', 
                         side_effect=Exception("Evaluation failed")):
            
            benchmark_data = experiment.benchmark_adapter.get_data()
            results = experiment._process_emotion_condition(
                benchmark_data, None, "neutral", 0.0
            )
            
            # Should still return results with score 0.0
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].score, 0.0)


if __name__ == '__main__':
    unittest.main()