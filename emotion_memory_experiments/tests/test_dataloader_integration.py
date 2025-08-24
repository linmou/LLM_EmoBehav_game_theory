"""
This file tests the DataLoader integration in EmotionMemoryExperiment to expose collate_fn duplication bug.
Purpose: Test that build_dataloader() method works correctly without keyword argument conflicts.
"""
import unittest
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path
import shutil

from ..experiment import EmotionMemoryExperiment
from .test_utils import create_mock_experiment_config, cleanup_temp_files


class TestDataLoaderIntegration(unittest.TestCase):
    """Test DataLoader integration to expose collate_fn duplication bug"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dirs = []
        self.configs = []
    
    def tearDown(self):
        """Clean up temporary files and directories"""
        cleanup_temp_files(self.configs)
        
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except FileNotFoundError:
                pass
    
    @patch('emotion_memory_experiments.experiment.setup_model_and_tokenizer')
    @patch('emotion_memory_experiments.experiment.ModelLayerDetector')
    @patch('emotion_memory_experiments.experiment.load_emotion_readers')
    @patch('emotion_memory_experiments.experiment.get_pipeline')
    def test_build_dataloader_no_collate_fn_duplication(self, mock_get_pipeline, mock_load_emotion_readers,
                                                       mock_model_detector, mock_setup_model):
        """Test that build_dataloader() doesn't create duplicate collate_fn arguments"""
        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_processor = MagicMock()
        mock_setup_model.return_value = (mock_model, mock_tokenizer, "chat", mock_processor)
        mock_model_detector.num_layers.return_value = 12
        mock_load_emotion_readers.return_value = {"anger": MagicMock()}
        mock_get_pipeline.return_value = MagicMock()
        
        # Create test config
        config = create_mock_experiment_config("passkey", 3)
        temp_dir = Path(tempfile.mkdtemp())
        config.output_dir = str(temp_dir)
        self.temp_dirs.append(temp_dir)
        self.configs.append(config.benchmark)
        
        # Initialize experiment
        experiment = EmotionMemoryExperiment(config)
        
        # This should NOT raise a TypeError about duplicate collate_fn arguments
        try:
            dataloader = experiment.build_dataloader()
            self.assertIsNotNone(dataloader)
        except TypeError as e:
            if "got multiple values for keyword argument 'collate_fn'" in str(e):
                self.fail(f"DataLoader creation failed due to collate_fn duplication: {e}")
            else:
                # Re-raise if it's a different TypeError
                raise


if __name__ == '__main__':
    unittest.main()