#!/usr/bin/env python3
"""
Comprehensive Tests for Multimodal Data Generation

Tests the enhanced data generation functions that support both text-only 
and multimodal (text+image) emotion datasets.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import json
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np
import sys
import os

# Add project root to path for imports
project_root = Path(__file__).parents[4]
sys.path.insert(0, str(project_root))

from neuro_manipulation.utils import get_emotion_images, primary_emotions_concept_dataset


class TestGetEmotionImages:
    """Test suite for get_emotion_images() function."""
    
    @pytest.fixture
    def temp_image_dir(self):
        """Create temporary directory with sample emotion images."""
        temp_dir = Path(tempfile.mkdtemp())
        
        emotions = ["anger", "happiness", "sadness"]
        
        for emotion in emotions:
            emotion_dir = temp_dir / emotion
            emotion_dir.mkdir(parents=True)
            
            # Create sample images for each emotion
            for i in range(3):
                img = Image.new('RGB', (100, 100), color='red' if emotion == 'anger' else 'blue')
                img.save(emotion_dir / f"{emotion}_{i}.jpg")
                
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_load_basic_emotion_images(self, temp_image_dir):
        """Test basic emotion image loading."""
        emotion_images = get_emotion_images(temp_image_dir, ["anger", "happiness", "sadness"])
        
        # Check that images were loaded for each emotion
        assert len(emotion_images) == 3
        assert "anger" in emotion_images
        assert "happiness" in emotion_images
        assert "sadness" in emotion_images
        
        # Check that each emotion has images
        for emotion in ["anger", "happiness", "sadness"]:
            assert len(emotion_images[emotion]) == 3
            assert all(isinstance(img, Image.Image) for img in emotion_images[emotion])
    
    def test_load_default_emotions(self, temp_image_dir):
        """Test loading with default emotion list."""
        # Add more emotion directories
        for emotion in ["fear", "disgust", "surprise"]:
            emotion_dir = temp_image_dir / emotion
            emotion_dir.mkdir(parents=True)
            img = Image.new('RGB', (100, 100), color='green')
            img.save(emotion_dir / f"{emotion}_1.jpg")
        
        emotion_images = get_emotion_images(temp_image_dir)  # No emotions specified
        
        # Should load all 6 standard emotions
        expected_emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
        assert len(emotion_images) == 6
        for emotion in expected_emotions:
            assert emotion in emotion_images
    
    def test_missing_emotion_directory(self, temp_image_dir):
        """Test handling of missing emotion directories."""
        emotion_images = get_emotion_images(temp_image_dir, ["anger", "nonexistent", "happiness"])
        
        # Should load existing emotions
        assert "anger" in emotion_images
        assert "happiness" in emotion_images
        assert len(emotion_images["anger"]) > 0
        assert len(emotion_images["happiness"]) > 0
        
        # Missing emotion should have empty list
        assert "nonexistent" in emotion_images
        assert len(emotion_images["nonexistent"]) == 0
    
    def test_multiple_image_formats(self, temp_image_dir):
        """Test loading different image formats."""
        anger_dir = temp_image_dir / "anger"
        
        # Add different image formats
        img = Image.new('RGB', (100, 100), color='red')
        img.save(anger_dir / "test.png")
        img.save(anger_dir / "test.bmp")
        
        emotion_images = get_emotion_images(temp_image_dir, ["anger"])
        
        # Should load all image formats (jpg + png + bmp = 5 total)
        assert len(emotion_images["anger"]) == 5


class TestPrimaryEmotionsConceptDatasetEnhanced:
    """Test suite for enhanced primary_emotions_concept_dataset() function."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with emotion scenario JSON files."""
        temp_dir = Path(tempfile.mkdtemp())
        
        emotions = ["anger", "happiness", "sadness"]
        
        for emotion in emotions:
            scenarios = [
                f"Sample {emotion} scenario 1",
                f"Sample {emotion} scenario 2", 
                f"Sample {emotion} scenario 3"
            ]
            
            with open(temp_dir / f"{emotion}.json", 'w') as f:
                json.dump(scenarios, f)
                
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def temp_image_dir(self):
        """Create temporary directory with emotion images."""
        temp_dir = Path(tempfile.mkdtemp())
        
        emotions = ["anger", "happiness", "sadness"]
        
        for emotion in emotions:
            emotion_dir = temp_dir / emotion
            emotion_dir.mkdir(parents=True)
            
            # Create sample images
            for i in range(2):
                img = Image.new('RGB', (100, 100), color='red')
                img.save(emotion_dir / f"{emotion}_{i}.jpg")
                
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.name_or_path = "test-model"
        return tokenizer
    
    def test_text_only_mode_backward_compatibility(self, temp_data_dir, mock_tokenizer):
        """Test that text-only mode works exactly as before."""
        with patch('neuro_manipulation.utils.PromptFormat') as mock_prompt_format:
            mock_format_instance = Mock()
            mock_format_instance.build.return_value = "formatted_prompt"
            mock_prompt_format.return_value = mock_format_instance
            
            # Call without multimodal parameters (backward compatibility)
            data = primary_emotions_concept_dataset(
                temp_data_dir, 
                model_name="test-model",
                tokenizer=mock_tokenizer
            )
            
            # Should return text-only format
            assert isinstance(data, dict)
            assert "anger" in data
            assert "train" in data["anger"]
            assert "test" in data["anger"]
            
            # Data should be strings (text-only)
            train_data = data["anger"]["train"]["data"]
            assert len(train_data) > 0
            assert all(isinstance(item, str) for item in train_data)
    
    def test_multimodal_mode_enabled(self, temp_data_dir, temp_image_dir, mock_tokenizer):
        """Test multimodal data generation."""
        with patch('neuro_manipulation.utils.PromptFormat') as mock_prompt_format:
            mock_format_instance = Mock()
            mock_format_instance.build.return_value = "formatted_prompt"
            mock_prompt_format.return_value = mock_format_instance
            
            # Call with multimodal parameters
            data = primary_emotions_concept_dataset(
                temp_data_dir,
                model_name="test-model", 
                tokenizer=mock_tokenizer,
                image_dir=temp_image_dir,
                multimodal=True
            )
            
            # Should return multimodal format
            assert isinstance(data, dict)
            assert "anger" in data
            
            # Data should be dictionaries with 'images' and 'text' keys
            train_data = data["anger"]["train"]["data"]
            assert len(train_data) > 0
            
            for item in train_data:
                assert isinstance(item, dict)
                assert "images" in item
                assert "text" in item
                assert isinstance(item["images"], list)
                assert len(item["images"]) == 1
                assert isinstance(item["images"][0], Image.Image)
                assert isinstance(item["text"], str)
                assert "Consider the emotion" in item["text"]
    
    def test_multimodal_disabled_with_images_available(self, temp_data_dir, temp_image_dir, mock_tokenizer):
        """Test that multimodal=False returns text-only even when images are available."""
        with patch('neuro_manipulation.utils.PromptFormat') as mock_prompt_format:
            mock_format_instance = Mock()
            mock_format_instance.build.return_value = "formatted_prompt"
            mock_prompt_format.return_value = mock_format_instance
            
            # Call with images available but multimodal=False
            data = primary_emotions_concept_dataset(
                temp_data_dir,
                model_name="test-model",
                tokenizer=mock_tokenizer, 
                image_dir=temp_image_dir,
                multimodal=False  # Explicitly disabled
            )
            
            # Should still return text-only format
            train_data = data["anger"]["train"]["data"]
            assert all(isinstance(item, str) for item in train_data)
    
    def test_multimodal_enabled_no_images(self, temp_data_dir, mock_tokenizer):
        """Test multimodal=True but no images available falls back to text-only."""
        with patch('neuro_manipulation.utils.PromptFormat') as mock_prompt_format:
            mock_format_instance = Mock()
            mock_format_instance.build.return_value = "formatted_prompt"
            mock_prompt_format.return_value = mock_format_instance
            
            # Call with multimodal=True but no image_dir
            data = primary_emotions_concept_dataset(
                temp_data_dir,
                model_name="test-model",
                tokenizer=mock_tokenizer,
                multimodal=True,
                image_dir=None  # No images
            )
            
            # Should fall back to text-only
            train_data = data["anger"]["train"]["data"]
            assert all(isinstance(item, str) for item in train_data)
    
    def test_prompt_template_consistency(self, temp_data_dir, temp_image_dir, mock_tokenizer):
        """Test that prompt templates are consistent between text-only and multimodal."""
        with patch('neuro_manipulation.utils.PromptFormat') as mock_prompt_format:
            mock_format_instance = Mock()
            mock_format_instance.build.return_value = "Consider the emotion of the following scenario:\nScenario: test\nAnswer:"
            mock_prompt_format.return_value = mock_format_instance
            
            # Get text-only data
            text_data = primary_emotions_concept_dataset(
                temp_data_dir,
                model_name="test-model",
                tokenizer=mock_tokenizer,
                multimodal=False
            )
            
            # Get multimodal data
            multimodal_data = primary_emotions_concept_dataset(
                temp_data_dir,
                model_name="test-model", 
                tokenizer=mock_tokenizer,
                image_dir=temp_image_dir,
                multimodal=True
            )
            
            # Text content should be identical (only difference is image addition)
            text_prompt = text_data["anger"]["train"]["data"][0]
            multimodal_prompt = multimodal_data["anger"]["train"]["data"][0]["text"]
            
            assert text_prompt == multimodal_prompt


class TestLoadEmotionReadersIntegration:
    """Integration tests for load_emotion_readers() with multimodal support."""
    
    @pytest.fixture
    def mock_config_text_only(self):
        """Mock configuration for text-only mode."""
        return {
            'data_dir': '/fake/data',
            'model_name_or_path': 'test-model',
            'rep_token': -1,
            'n_difference': 1,
            'direction_method': 'pca',
            'emotions': ['anger', 'happiness'],
            'rebuild': True,
            'multimodal': False
        }
    
    @pytest.fixture  
    def mock_config_multimodal(self):
        """Mock configuration for multimodal mode."""
        return {
            'data_dir': '/fake/data',
            'model_name_or_path': 'Qwen/Qwen2.5-VL-3B-Instruct',
            'rep_token': -1,
            'n_difference': 1,
            'direction_method': 'pca',
            'emotions': ['anger', 'happiness'],
            'rebuild': True,
            'multimodal': True,
            'image_dir': '/fake/images'
        }
    
    @patch('neuro_manipulation.model_utils.primary_emotions_concept_dataset')
    @patch('neuro_manipulation.model_utils.pipeline')
    @patch('neuro_manipulation.model_utils.all_emotion_rep_reader')
    def test_text_only_pipeline_creation(self, mock_all_emotion, mock_pipeline, mock_dataset, mock_config_text_only):
        """Test that text-only pipeline is created correctly."""
        from neuro_manipulation.model_utils import load_emotion_readers
        
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_dataset.return_value = {"anger": {"train": {"data": ["text"], "labels": []}}}
        mock_pipeline.return_value = Mock()
        mock_all_emotion.return_value = {"anger": Mock()}
        
        result = load_emotion_readers(
            mock_config_text_only, mock_model, mock_tokenizer, [-1]
        )
        
        # Should call pipeline with "rep-reading" task
        mock_pipeline.assert_called_once_with("rep-reading", model=mock_model, tokenizer=mock_tokenizer)
        
        # Should call dataset function without multimodal parameters
        mock_dataset.assert_called_once_with(
            '/fake/data',
            model_name='test-model',
            tokenizer=mock_tokenizer,
            image_dir=None,
            multimodal=False
        )
    
    @patch('neuro_manipulation.model_utils.primary_emotions_concept_dataset')
    @patch('neuro_manipulation.model_utils.pipeline')
    @patch('neuro_manipulation.model_utils.all_emotion_rep_reader')
    def test_multimodal_pipeline_creation(self, mock_all_emotion, mock_pipeline, mock_dataset, mock_config_multimodal):
        """Test that multimodal pipeline is created correctly."""
        from neuro_manipulation.model_utils import load_emotion_readers
        
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_processor = Mock()
        mock_dataset.return_value = {"anger": {"train": {"data": [{"images": [], "text": ""}], "labels": []}}}
        mock_pipeline.return_value = Mock()
        mock_all_emotion.return_value = {"anger": Mock()}
        
        result = load_emotion_readers(
            mock_config_multimodal, mock_model, mock_tokenizer, [-1], processor=mock_processor
        )
        
        # Should call pipeline with "multimodal-rep-reading" task
        mock_pipeline.assert_called_once_with(
            "multimodal-rep-reading", 
            model=mock_model, 
            tokenizer=mock_tokenizer,
            image_processor=mock_processor
        )
        
        # Should call dataset function with multimodal parameters
        mock_dataset.assert_called_once_with(
            '/fake/data',
            model_name='Qwen/Qwen2.5-VL-3B-Instruct',
            tokenizer=mock_tokenizer,
            image_dir='/fake/images',
            multimodal=True
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])