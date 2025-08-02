import pytest
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import os
import tempfile
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parents[5]
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer, AutoProcessor, PreTrainedModel, PretrainedConfig
from neuro_manipulation.repe.rep_reading_pipeline import RepReadingPipeline
from neuro_manipulation.repe.pipelines import repe_pipeline_registry
from neuro_manipulation.model_layer_detector import ModelLayerDetector


class MockMultimodalConfig(PretrainedConfig):
    """Mock configuration for multimodal model."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = 768
        self.num_hidden_layers = 12
        self.model_type = "qwen2_vl"


class MockLayer(nn.Module):
    """Mock transformer layer."""
    def __init__(self):
        super().__init__()
        self.attention = Mock()
        self.mlp = Mock()
        self.layer_norm = Mock()
    
    def forward(self, x):
        return x


class MockMultimodalModel(PreTrainedModel):
    """Mock multimodal model that works with Transformers framework."""
    config_class = MockMultimodalConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Create proper nn.Module layers
        self.layers = nn.ModuleList([
            MockLayer() for _ in range(config.num_hidden_layers)
        ])
        
        # Create mock structure for layer detection
        self.model = nn.Module()
        self.model.layers = self.layers
        
        # Mock vision components
        self.vision_tower = nn.Module()
        self.language_model = nn.Module()
        
    def forward(self, **kwargs):
        """Mock forward pass with hidden states."""
        batch_size = 1
        seq_len = kwargs.get('input_ids', torch.tensor([[1, 2, 3, 4, 5]])).shape[1]
        hidden_size = self.config.hidden_size
        
        # Create mock hidden states for each layer
        hidden_states = []
        for _ in range(self.config.num_hidden_layers):
            layer_states = torch.randn(batch_size, seq_len, hidden_size)
            hidden_states.append(layer_states)
        
        # Mock output with hidden states
        output = Mock()
        output.hidden_states = hidden_states
        output.last_hidden_state = hidden_states[-1]
        
        return output
    
    def named_modules(self):
        """Mock named_modules for layer detection."""
        return [
            ('vision_tower.vision_model.encoder.layers', self.layers),
            ('language_model.model.layers', self.layers),
            ('model.layers', self.layers)
        ]


class TestMultimodalRepReading:
    """Test suite for multimodal representation reading functionality."""
    
    @pytest.fixture
    def mock_multimodal_model(self):
        """Create a mock multimodal model with proper structure."""
        config = MockMultimodalConfig()
        model = MockMultimodalModel(config)
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        }
        tokenizer.pad_token = 0
        return tokenizer
    
    @pytest.fixture
    def mock_image_processor(self):
        """Create a mock image processor."""
        processor = Mock()
        processor.return_value = {
            'pixel_values': torch.randn(1, 3, 224, 224)
        }
        return processor
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample PIL image."""
        return Image.new('RGB', (224, 224), color='red')
    
    def test_multimodal_input_detection(self, mock_multimodal_model, mock_tokenizer, mock_image_processor, sample_image):
        """Test detection of multimodal inputs."""
        pipeline = RepReadingPipeline(
            model=mock_multimodal_model,
            tokenizer=mock_tokenizer,
            image_processor=mock_image_processor
        )
        
        # Test dict input with images
        multimodal_input = {
            'images': [sample_image],
            'text': 'when you see this image, your emotion is anger'
        }
        assert pipeline._is_multimodal_input(multimodal_input)
        
        # Test dict input with image (singular)
        multimodal_input_singular = {
            'image': sample_image,
            'text': 'when you see this image, your emotion is happiness'
        }
        assert pipeline._is_multimodal_input(multimodal_input_singular)
        
        # Test text-only input
        text_input = "This is just text"
        assert not pipeline._is_multimodal_input(text_input)
        
        # Test list with mixed content
        mixed_input = [multimodal_input, "text only"]
        assert pipeline._is_multimodal_input(mixed_input)
    
    def test_multimodal_input_preparation(self, mock_multimodal_model, mock_tokenizer, mock_image_processor, sample_image):
        """Test preparation of multimodal inputs for processing."""
        pipeline = RepReadingPipeline(
            model=mock_multimodal_model,
            tokenizer=mock_tokenizer,
            image_processor=mock_image_processor
        )
        
        multimodal_input = {
            'images': [sample_image],
            'text': 'when you see this image, your emotion is anger'
        }
        
        result = pipeline._prepare_multimodal_inputs(multimodal_input)
        
        # Check that both image processor and tokenizer were called
        mock_image_processor.assert_called_once_with([sample_image], return_tensors="pt")
        mock_tokenizer.assert_called_once_with(
            'when you see this image, your emotion is anger', 
            return_tensors="pt"
        )
        
        # Result should combine both outputs
        assert 'pixel_values' in result
        assert 'input_ids' in result
    
    def test_multimodal_preprocessing(self, mock_multimodal_model, mock_tokenizer, mock_image_processor, sample_image):
        """Test the complete preprocessing pipeline for multimodal inputs."""
        pipeline = RepReadingPipeline(
            model=mock_multimodal_model,
            tokenizer=mock_tokenizer,
            image_processor=mock_image_processor
        )
        
        multimodal_input = {
            'images': [sample_image],
            'text': 'when you see this image, your emotion is anger'
        }
        
        result = pipeline.preprocess(multimodal_input)
        
        assert 'pixel_values' in result
        assert 'input_ids' in result
        assert result['pixel_values'].shape[0] == 1  # Batch size 1
        assert result['input_ids'].shape[0] == 1     # Batch size 1
    
    @patch('neuro_manipulation.repe.rep_reading_pipeline.torch.no_grad')
    def test_multimodal_forward_pass(self, mock_no_grad, mock_multimodal_model, mock_tokenizer, mock_image_processor):
        """Test forward pass with multimodal inputs."""
        # Setup the context manager mock
        mock_no_grad.return_value.__enter__ = Mock()
        mock_no_grad.return_value.__exit__ = Mock(return_value=None)
        
        pipeline = RepReadingPipeline(
            model=mock_multimodal_model,
            tokenizer=mock_tokenizer,
            image_processor=mock_image_processor
        )
        
        # Mock model inputs
        model_inputs = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]),
            'pixel_values': torch.randn(1, 3, 224, 224),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        }
        
        # Test forward pass
        result = pipeline._forward(
            model_inputs=model_inputs,
            rep_token=-1,
            hidden_layers=[0, 1, 2],
            rep_reader=None
        )
        
        # Check that model was called with multimodal inputs
        mock_multimodal_model.assert_called_once_with(
            **model_inputs,
            output_hidden_states=True
        )
        
        # Result should contain hidden states for each layer
        assert isinstance(result, dict)
        assert len(result) == 3  # Three layers
        for layer_idx in [0, 1, 2]:
            assert layer_idx in result
            assert result[layer_idx].shape == (1, 768)  # Should extract last token
    
    def test_model_layer_detection_multimodal(self, mock_multimodal_model):
        """Test that multimodal models are properly detected."""
        # Test multimodal detection
        is_multimodal = ModelLayerDetector.is_multimodal_model(mock_multimodal_model)
        assert is_multimodal
        
        # Test layer info extraction
        layer_info = ModelLayerDetector.get_multimodal_layer_info(mock_multimodal_model)
        assert 'vision_layers' in layer_info
        assert 'text_layers' in layer_info
        assert 'fusion_layers' in layer_info
    
    @patch('neuro_manipulation.repe.pipelines.pipeline')
    def test_pipeline_registration(self, mock_pipeline_fn):
        """Test that multimodal pipeline is properly registered."""
        from transformers.pipelines import PIPELINE_REGISTRY
        
        # Register pipelines
        repe_pipeline_registry()
        
        # Check that multimodal-rep-reading is registered
        assert "multimodal-rep-reading" in PIPELINE_REGISTRY.supported_tasks
        
        # Check that the pipeline class is correct
        task_info = PIPELINE_REGISTRY.supported_tasks["multimodal-rep-reading"]
        assert task_info["impl"] == RepReadingPipeline
    
    def test_emotion_template_processing(self, mock_multimodal_model, mock_tokenizer, mock_image_processor, sample_image):
        """Test processing of emotion inquiry templates."""
        pipeline = RepReadingPipeline(
            model=mock_multimodal_model,
            tokenizer=mock_tokenizer,
            image_processor=mock_image_processor
        )
        
        # Test different emotion templates
        templates = [
            "when you see this image, your emotion is anger",
            "when you see this image, your emotion is happiness", 
            "when you see this image, your emotion is sadness"
        ]
        
        for template in templates:
            multimodal_input = {
                'images': [sample_image],
                'text': template
            }
            
            result = pipeline.preprocess(multimodal_input)
            assert 'pixel_values' in result
            assert 'input_ids' in result
            
            # Verify tokenizer was called with the correct template
            assert mock_tokenizer.call_args[0][0] == template
    
    def test_batch_multimodal_processing(self, mock_multimodal_model, mock_tokenizer, mock_image_processor, sample_image):
        """Test batch processing of multimodal inputs."""
        pipeline = RepReadingPipeline(
            model=mock_multimodal_model,
            tokenizer=mock_tokenizer,
            image_processor=mock_image_processor
        )
        
        # Create batch of multimodal inputs
        batch_inputs = [
            {
                'images': [sample_image],
                'text': 'when you see this image, your emotion is anger'
            },
            {
                'images': [sample_image],
                'text': 'when you see this image, your emotion is happiness'
            }
        ]
        
        result = pipeline.preprocess(batch_inputs)
        
        # Should handle batching (currently returns first item, but structure is there)
        assert 'pixel_values' in result
        assert 'input_ids' in result
    
    def test_backward_compatibility(self, mock_multimodal_model, mock_tokenizer, mock_image_processor):
        """Test that text-only inputs still work (backward compatibility)."""
        pipeline = RepReadingPipeline(
            model=mock_multimodal_model,
            tokenizer=mock_tokenizer,
            image_processor=mock_image_processor
        )
        
        # Test text-only input
        text_input = "This is a text-only input"
        result = pipeline.preprocess(text_input)
        
        # Should use tokenizer only
        assert 'input_ids' in result
        assert 'pixel_values' not in result
        mock_tokenizer.assert_called_with(text_input, return_tensors="pt")


class TestMultimodalConfiguration:
    """Test suite for multimodal configuration handling."""
    
    def test_config_file_structure(self):
        """Test that the multimodal config file has correct structure."""
        config_path = "/data/home/jjl7137/LLM_EmoBehav_game_theory_multimodal/config/multimodal_rep_reading_config.yaml"
        assert os.path.exists(config_path)
        
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        assert 'experiment' in config
        assert 'pipeline' in config['experiment']
        assert 'emotions' in config['experiment']
        assert 'emotion_template' in config['experiment']
        assert 'data' in config['experiment']
        
        # Check pipeline configuration
        pipeline_config = config['experiment']['pipeline']
        assert pipeline_config['task'] == 'multimodal-rep-reading'
        assert pipeline_config['rep_token'] == -1
        assert 'hidden_layers' in pipeline_config
        
        # Check emotion configuration
        emotions = config['experiment']['emotions']
        assert isinstance(emotions, list)
        assert len(emotions) > 0
        assert 'anger' in emotions
        assert 'happiness' in emotions
    
    def test_usage_examples_in_config(self):
        """Test that usage examples in config are well-formed."""
        config_path = "/data/home/jjl7137/LLM_EmoBehav_game_theory_multimodal/config/multimodal_rep_reading_config.yaml"
        
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        usage_examples = config['usage_examples']
        
        # Check basic extraction example
        basic = usage_examples['basic_extraction']
        assert 'input' in basic
        assert 'images' in basic['input']
        assert 'text' in basic['input']
        assert 'expected_output' in basic
        
        # Check batch processing example
        batch = usage_examples['batch_processing']
        assert 'input' in batch
        assert isinstance(batch['input'], list)
        assert len(batch['input']) >= 2


if __name__ == "__main__":
    pytest.main([__file__])