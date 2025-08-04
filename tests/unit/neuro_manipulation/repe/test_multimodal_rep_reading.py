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


# MockMultimodalModel classes removed - using proper Mock objects instead


class TestMultimodalRepReading:
    """Test suite for multimodal representation reading functionality."""
    
    @pytest.fixture
    def mock_multimodal_model(self):
        """Create a proper mock multimodal model that inherits from PreTrainedModel."""
        # Create a mock that inherits from PreTrainedModel for pipeline compatibility
        class MockMultimodalModel(PreTrainedModel):
            def __init__(self):
                # Initialize the nn.Module part properly
                nn.Module.__init__(self)
                self.config = Mock()
                self.config.model_type = "qwen2_vl"
                
            def forward(self, **kwargs):
                # Mock forward pass with hidden states
                hidden_states = []
                for i in range(12):  # 12 layers
                    layer_hidden = torch.randn(1, 10, 768)  # batch_size=1, seq_len=10, hidden_size=768
                    hidden_states.append(layer_hidden)
                
                # Create a more realistic output object that supports subscripting
                class MockOutput:
                    def __init__(self, hidden_states):
                        self.hidden_states = hidden_states
                        self.last_hidden_state = hidden_states[-1]
                        
                    def __getitem__(self, key):
                        return getattr(self, key, None)
                        
                    def __setitem__(self, key, value):
                        setattr(self, key, value)
                        
                    def __contains__(self, key):
                        return hasattr(self, key)
                
                return MockOutput(hidden_states)
                
            def parameters(self, recurse=True):
                # Return a mock parameter for device detection
                mock_param = Mock()
                mock_param.device = torch.device('cpu')
                yield mock_param
                
            def named_modules(self, memo=None, prefix='', remove_duplicate=True):
                return [
                    ('vision_tower.vision_model.encoder.layers', Mock()),
                    ('language_model.model.layers', Mock()),
                    ('model.layers', Mock())
                ]
        
        mock_model = MockMultimodalModel()
        
        # Wrap in a spy to enable assertion checking
        mock_model.forward = Mock(side_effect=mock_model.forward)
        
        return mock_model
    
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
        
        # Check that the unified processor was called (modern implementation)
        # The actual implementation uses AutoProcessor with text+images together
        mock_image_processor.assert_called_once()  # Unified processor call
        mock_tokenizer.apply_chat_template.assert_called_once()  # Chat template formatting
        
        # Result should contain multimodal data
        # Note: exact keys depend on processor implementation
        assert result is not None
        assert isinstance(result, dict)
    
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
        
        # Check that preprocessing returns valid data structure
        assert result is not None
        assert isinstance(result, dict)
        # Note: exact keys depend on processor implementation
    
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
        
        # Check that model.forward was called with multimodal inputs
        mock_multimodal_model.forward.assert_called_once_with(
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
            # Check that preprocessing works for emotion templates
            assert result is not None
            assert isinstance(result, dict)
    
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
        
        # Should handle batching (processor implementation dependent)
        assert result is not None
        assert isinstance(result, dict)
    
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
        # Use project-relative path instead of hardcoded path
        project_root = Path(__file__).parents[4]  # Go up 4 levels to reach project root
        config_path = project_root / "config" / "multimodal_rep_reading_config.yaml"
        assert config_path.exists(), f"Config file not found at {config_path}"
        
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
        # Use project-relative path instead of hardcoded path
        project_root = Path(__file__).parents[4]  # Go up 4 levels to reach project root
        config_path = project_root / "config" / "multimodal_rep_reading_config.yaml"
        
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


    def test_multimodal_get_directions_integration(self, mock_multimodal_model, mock_tokenizer, mock_image_processor, sample_image):
        """Integration test for get_directions() with multimodal input format."""
        pipeline = RepReadingPipeline(
            model=mock_multimodal_model,
            tokenizer=mock_tokenizer,
            image_processor=mock_image_processor
        )
        
        # Create multimodal training data in the exact format used by primary_emotions_concept_dataset
        train_inputs = [
            {"images": [sample_image], "text": "when you see this image, your emotion is anger"},
            {"images": [sample_image], "text": "when you see this image, your emotion is happiness"},
            {"images": [sample_image], "text": "when you see this image, your emotion is anger"},
            {"images": [sample_image], "text": "when you see this image, your emotion is happiness"},
        ]
        
        # Test parameters matching all_emotion_rep_reader usage
        hidden_layers = [0, 1, 2]
        rep_token = -1
        n_difference = 1
        train_labels = [0, 1, 0, 1]  # Binary labels for PCA
        direction_method = 'pca'
        
        try:
            # This is the critical method that wasn't tested before
            rep_reader = pipeline.get_directions(
                train_inputs=train_inputs,
                rep_token=rep_token,
                hidden_layers=hidden_layers,
                n_difference=n_difference,
                train_labels=train_labels,
                direction_method=direction_method,
                batch_size=2
            )
            
            # Verify RepReader structure
            assert rep_reader is not None
            assert hasattr(rep_reader, 'directions')
            assert hasattr(rep_reader, 'direction_signs')
            
            # Verify directions for each layer
            for layer in hidden_layers:
                assert layer in rep_reader.directions
                assert layer in rep_reader.direction_signs
                assert rep_reader.directions[layer] is not None
                
        except Exception as e:
            pytest.fail(f"get_directions() failed with multimodal input: {e}")
    
    def test_multimodal_batched_string_to_hiddens_integration(self, mock_multimodal_model, mock_tokenizer, mock_image_processor, sample_image):
        """Integration test for _batched_string_to_hiddens() with multimodal data."""
        pipeline = RepReadingPipeline(
            model=mock_multimodal_model,
            tokenizer=mock_tokenizer,
            image_processor=mock_image_processor
        )
        
        # Multimodal input batch (List[Dict] format)
        train_inputs = [
            {"images": [sample_image], "text": "emotion: anger"},
            {"images": [sample_image], "text": "emotion: happiness"},
            {"images": [sample_image], "text": "emotion: sadness"}
        ]
        
        hidden_layers = [0, 1, 2]
        rep_token = -1
        batch_size = 2
        
        try:
            # Test the critical batching method
            hidden_states = pipeline._batched_string_to_hiddens(
                train_inputs=train_inputs,
                rep_token=rep_token,
                hidden_layers=hidden_layers,
                batch_size=batch_size,
                which_hidden_states=None
            )
            
            # Verify structure
            assert isinstance(hidden_states, dict)
            assert len(hidden_states) == len(hidden_layers)
            
            # Verify each layer has correct dimensions
            for layer in hidden_layers:
                assert layer in hidden_states
                layer_hidden = hidden_states[layer]
                assert layer_hidden.shape[0] == len(train_inputs)  # Batch dimension
                assert layer_hidden.shape[1] == 768  # Hidden dimension (from mock)
                
        except Exception as e:
            pytest.fail(f"_batched_string_to_hiddens() failed with multimodal input: {e}")
    
    def test_end_to_end_multimodal_repe_workflow(self, mock_multimodal_model, mock_tokenizer, mock_image_processor, sample_image):
        """End-to-end integration test for complete multimodal RepE workflow."""
        pipeline = RepReadingPipeline(
            model=mock_multimodal_model,
            tokenizer=mock_tokenizer,
            image_processor=mock_image_processor
        )
        
        # Simulate realistic emotion data structure from primary_emotions_concept_dataset
        emotion_data = {
            "anger": {
                "train": {
                    "data": [
                        {"images": [sample_image], "text": "Consider the emotion of the following scenario:\nScenario: [IMAGE]\nAnswer:"},
                        {"images": [sample_image], "text": "Consider the emotion of the following scenario:\nScenario: [IMAGE]\nAnswer:"},
                        {"images": [sample_image], "text": "Consider the emotion of the following scenario:\nScenario: [IMAGE]\nAnswer:"},
                        {"images": [sample_image], "text": "Consider the emotion of the following scenario:\nScenario: [IMAGE]\nAnswer:"},
                    ],
                    "labels": [[True, False], [False, True], [True, False], [False, True]]
                },
                "test": {
                    "data": [
                        {"images": [sample_image], "text": "Consider the emotion of the following scenario:\nScenario: [IMAGE]\nAnswer:"},
                        {"images": [sample_image], "text": "Consider the emotion of the following scenario:\nScenario: [IMAGE]\nAnswer:"},
                    ],
                    "labels": [[True, False], [False, True]]
                }
            }
        }
        
        try:
            # Test the complete workflow that all_emotion_rep_reader would use
            train_data = emotion_data["anger"]["train"]
            test_data = emotion_data["anger"]["test"]
            
            # Step 1: Extract directions (main RepE operation)
            rep_reader = pipeline.get_directions(
                train_inputs=train_data["data"],
                rep_token=-1,
                hidden_layers=[0, 1, 2],
                n_difference=1,
                train_labels=[0, 1, 0, 1],  # Binary labels from labels structure
                direction_method='pca',
                batch_size=2
            )
            
            # Step 2: Test directions (validation step)
            test_results = pipeline(
                test_data["data"],
                rep_token=-1,
                hidden_layers=[0, 1, 2],
                rep_reader=rep_reader,
                batch_size=2
            )
            
            # Verify end-to-end results
            assert rep_reader is not None
            assert hasattr(rep_reader, 'directions')
            assert len(test_results) == len(test_data["data"])
            
            # Verify each test result has the expected structure
            for result in test_results:
                assert isinstance(result, dict)
                for layer in [0, 1, 2]:
                    assert layer in result
                    
        except Exception as e:
            pytest.fail(f"End-to-end multimodal RepE workflow failed: {e}")
    
    @patch('neuro_manipulation.utils.get_rep_reader')
    def test_all_emotion_rep_reader_multimodal_integration(self, mock_get_rep_reader, mock_multimodal_model, mock_tokenizer, mock_image_processor, sample_image):
        """Integration test with all_emotion_rep_reader using multimodal data."""
        from neuro_manipulation.utils import all_emotion_rep_reader
        
        # Mock get_rep_reader to avoid complex RepReader logic
        mock_rep_reader = Mock()
        mock_rep_reader.directions = {0: torch.randn(768), 1: torch.randn(768)}
        mock_rep_reader.direction_signs = {0: 1, 1: -1}
        mock_get_rep_reader.return_value = (mock_rep_reader, {0: 0.75, 1: 0.82})
        
        pipeline = RepReadingPipeline(
            model=mock_multimodal_model,
            tokenizer=mock_tokenizer,
            image_processor=mock_image_processor
        )
        
        # Multimodal data structure exactly as primary_emotions_concept_dataset produces
        data = {
            "anger": {
                "train": {
                    "data": [
                        {"images": [sample_image], "text": "formatted_prompt_anger_1"},
                        {"images": [sample_image], "text": "formatted_prompt_anger_2"},
                    ],
                    "labels": [[True, False], [False, True]]
                },
                "test": {
                    "data": [
                        {"images": [sample_image], "text": "formatted_prompt_anger_test_1"},
                    ],
                    "labels": [[True, False]]
                }
            },
            "happiness": {
                "train": {
                    "data": [
                        {"images": [sample_image], "text": "formatted_prompt_happiness_1"},
                        {"images": [sample_image], "text": "formatted_prompt_happiness_2"},
                    ],
                    "labels": [[False, True], [True, False]]
                },
                "test": {
                    "data": [
                        {"images": [sample_image], "text": "formatted_prompt_happiness_test_1"},
                    ],
                    "labels": [[False, True]]
                }
            }
        }
        
        try:
            # Test the actual integration point
            emotion_rep_readers = all_emotion_rep_reader(
                data=data,
                emotions=["anger", "happiness"],
                rep_reading_pipeline=pipeline,
                hidden_layers=[0, 1],
                rep_token=-1,
                n_difference=1,
                direction_method='pca',
                save_path=None
            )
            
            # Verify results structure
            assert isinstance(emotion_rep_readers, dict)
            assert "anger" in emotion_rep_readers
            assert "happiness" in emotion_rep_readers
            assert "layer_acc" in emotion_rep_readers
            
            # Verify each emotion has a RepReader
            for emotion in ["anger", "happiness"]:
                assert emotion_rep_readers[emotion] is not None
                assert emotion in emotion_rep_readers["layer_acc"]
                
            # Verify get_rep_reader was called with multimodal data
            assert mock_get_rep_reader.call_count == 2  # Once per emotion
            
            # Verify the calls had the correct multimodal data structure
            for call in mock_get_rep_reader.call_args_list:
                args, kwargs = call
                train_data = args[1]  # Second argument is train_data
                assert "data" in train_data
                assert isinstance(train_data["data"], list)
                if len(train_data["data"]) > 0:
                    sample_item = train_data["data"][0]
                    assert isinstance(sample_item, dict)
                    assert "images" in sample_item
                    assert "text" in sample_item
                    
        except Exception as e:
            pytest.fail(f"all_emotion_rep_reader integration with multimodal data failed: {e}")
    
    def test_multimodal_pipeline_call_integration(self, mock_multimodal_model, mock_tokenizer, mock_image_processor, sample_image):
        """Integration test for pipeline __call__ method with multimodal input."""
        pipeline = RepReadingPipeline(
            model=mock_multimodal_model,
            tokenizer=mock_tokenizer,
            image_processor=mock_image_processor
        )
        
        # Mock RepReader for testing
        mock_rep_reader = Mock()
        mock_rep_reader.directions = {0: torch.randn(768), 1: torch.randn(768)}
        mock_rep_reader.direction_signs = {0: 1, 1: -1}
        
        multimodal_inputs = [
            {"images": [sample_image], "text": "emotion test 1"},
            {"images": [sample_image], "text": "emotion test 2"}
        ]
        
        try:
            # Test pipeline call (this is what test_direction() uses)
            results = pipeline(
                multimodal_inputs,
                rep_token=-1,
                hidden_layers=[0, 1],
                rep_reader=mock_rep_reader,
                batch_size=1
            )
            
            # Verify results
            assert isinstance(results, list)
            assert len(results) == len(multimodal_inputs)
            
            for result in results:
                assert isinstance(result, dict)
                for layer in [0, 1]:
                    assert layer in result
                    assert isinstance(result[layer], torch.Tensor)
                    
        except Exception as e:
            pytest.fail(f"Pipeline __call__ with multimodal input failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])