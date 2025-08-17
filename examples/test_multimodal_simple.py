#!/usr/bin/env python3
"""
Simple test script for multimodal emotion extraction functionality.
Tests the core multimodal preprocessing without full pipeline overhead.
"""

import torch
from PIL import Image
import numpy as np
from unittest.mock import Mock

# Add project root to path
import sys
import os
sys.path.append('/data/home/jjl7137/LLM_EmoBehav_game_theory_multimodal')

from neuro_manipulation.repe.rep_reading_pipeline import RepReadingPipeline
from neuro_manipulation.model_layer_detector import ModelLayerDetector


def test_multimodal_input_detection():
    """Test multimodal input detection functionality."""
    print("Testing multimodal input detection...")
    
    # Create a minimal pipeline instance for testing detection methods
    pipeline = RepReadingPipeline.__new__(RepReadingPipeline)
    
    # Create sample image
    sample_image = Image.new('RGB', (224, 224), color='red')
    
    # Test multimodal input detection
    multimodal_input = {
        'images': [sample_image],
        'text': 'when you see this image, your emotion is anger'
    }
    
    is_multimodal = pipeline._is_multimodal_input(multimodal_input)
    assert is_multimodal, "Should detect multimodal input"
    print("✓ Multimodal input detection works")
    
    # Test text-only input
    text_input = "This is just text"
    is_text_only = pipeline._is_multimodal_input(text_input)
    assert not is_text_only, "Should not detect text-only as multimodal"
    print("✓ Text-only input detection works")
    
    # Test singular image input
    singular_input = {
        'image': sample_image,
        'text': 'when you see this image, your emotion is happiness'
    }
    is_singular = pipeline._is_multimodal_input(singular_input)
    assert is_singular, "Should detect singular image input as multimodal"
    print("✓ Singular image input detection works")


def test_multimodal_input_preparation():
    """Test multimodal input preparation."""
    print("\nTesting multimodal input preparation...")
    
    # Create pipeline instance
    pipeline = RepReadingPipeline.__new__(RepReadingPipeline)
    
    # Mock processors
    mock_image_processor = Mock()
    mock_image_processor.return_value = {'pixel_values': torch.randn(1, 3, 224, 224)}
    
    mock_tokenizer = Mock()
    mock_tokenizer.return_value = {
        'input_ids': torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]])
    }
    
    # Assign to pipeline
    pipeline.image_processor = mock_image_processor
    pipeline.tokenizer = mock_tokenizer
    
    # Test preparation
    sample_image = Image.new('RGB', (224, 224), color='red')
    multimodal_input = {
        'images': [sample_image],
        'text': 'when you see this image, your emotion is anger'
    }
    
    result = pipeline._prepare_multimodal_inputs(multimodal_input)
    
    # Check results
    assert 'pixel_values' in result, "Should contain image features"
    assert 'input_ids' in result, "Should contain text tokens"
    assert result['pixel_values'].shape == torch.Size([1, 3, 224, 224])
    assert result['input_ids'].shape == torch.Size([1, 8])
    
    print("✓ Multimodal input preparation works")
    print(f"  - Image features shape: {result['pixel_values'].shape}")
    print(f"  - Text tokens shape: {result['input_ids'].shape}")


def test_model_layer_detection():
    """Test multimodal model layer detection."""
    print("\nTesting multimodal model detection...")
    
    # Create mock multimodal model
    mock_model = Mock()
    mock_model.__class__.__name__ = "Qwen2VLForConditionalGeneration"
    
    # Add vision-related modules
    mock_model.named_modules.return_value = [
        ('vision_tower.vision_model.encoder.layers', Mock()),
        ('language_model.model.layers', Mock()),
        ('cross_attention_layers', Mock())
    ]
    
    # Test detection
    is_multimodal = ModelLayerDetector.is_multimodal_model(mock_model)
    assert is_multimodal, "Should detect Qwen model as multimodal"
    print("✓ Multimodal model detection works")
    
    # Test layer info extraction
    layer_info = ModelLayerDetector.get_multimodal_layer_info(mock_model)
    assert 'vision_layers' in layer_info
    assert 'text_layers' in layer_info
    assert 'fusion_layers' in layer_info
    print("✓ Layer info extraction works")
    print(f"  - Available layer types: {list(layer_info.keys())}")


def test_emotion_templates():
    """Test emotion template processing."""
    print("\nTesting emotion template processing...")
    
    templates = [
        "when you see this image, your emotion is anger",
        "when you see this image, your emotion is happiness", 
        "when you see this image, your emotion is sadness",
        "when you see this image, your emotion is disgust",
        "when you see this image, your emotion is fear",
        "when you see this image, your emotion is surprise"
    ]
    
    sample_image = Image.new('RGB', (224, 224), color='blue')
    
    for template in templates:
        multimodal_input = {
            'images': [sample_image],
            'text': template
        }
        
        # Test that input is recognized as multimodal
        pipeline = RepReadingPipeline.__new__(RepReadingPipeline)
        is_multimodal = pipeline._is_multimodal_input(multimodal_input)
        assert is_multimodal, f"Template should be recognized as multimodal: {template}"
    
    print("✓ All emotion templates work correctly")
    print(f"  - Tested {len(templates)} emotion templates")


def test_config_validation():
    """Test multimodal configuration file."""
    print("\nTesting configuration validation...")
    
    import yaml
    config_path = "/data/home/jjl7137/LLM_EmoBehav_game_theory_multimodal/config/multimodal_rep_reading_config.yaml"
    
    assert os.path.exists(config_path), "Config file should exist"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate structure
    assert 'experiment' in config
    assert 'pipeline' in config['experiment']
    assert config['experiment']['pipeline']['task'] == 'multimodal-rep-reading'
    assert config['experiment']['pipeline']['rep_token'] == -1
    
    # Validate emotions
    emotions = config['experiment']['emotions']
    expected_emotions = ['anger', 'happiness', 'sadness', 'disgust', 'fear', 'surprise']
    for emotion in expected_emotions:
        assert emotion in emotions, f"Missing emotion: {emotion}"
    
    # Validate usage examples
    assert 'usage_examples' in config
    assert 'basic_extraction' in config['usage_examples']
    
    print("✓ Configuration file validation passed")
    print(f"  - Task: {config['experiment']['pipeline']['task']}")
    print(f"  - Rep token: {config['experiment']['pipeline']['rep_token']}")
    print(f"  - Emotions: {emotions}")


def run_all_tests():
    """Run all simple multimodal tests."""
    print("=" * 60)
    print("MULTIMODAL REPE FUNCTIONALITY TESTS")
    print("=" * 60)
    
    try:
        test_multimodal_input_detection()
        test_multimodal_input_preparation() 
        test_model_layer_detection()
        test_emotion_templates()
        test_config_validation()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✅")
        print("Multimodal RepE functionality is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)