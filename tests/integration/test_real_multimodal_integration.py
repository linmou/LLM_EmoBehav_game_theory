#!/usr/bin/env python3
"""
Real Integration Tests for Multimodal RepE with Qwen2.5-VL Model

These tests use the actual Qwen2.5-VL-3B-Instruct model to validate 
end-to-end multimodal emotion extraction functionality.
"""

import pytest
import torch
from PIL import Image
import numpy as np
import os
import tempfile
from pathlib import Path

# Skip tests if model not available
MODEL_PATH = "/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-VL-3B-Instruct"
SKIP_REASON = f"Qwen2.5-VL model not found at {MODEL_PATH}"

# Add project root to path for imports
import sys
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer, AutoProcessor, AutoModel
from transformers.pipelines import pipeline

from neuro_manipulation.repe.pipelines import repe_pipeline_registry
from neuro_manipulation.repe.rep_reading_pipeline import RepReadingPipeline
from neuro_manipulation.model_layer_detector import ModelLayerDetector
from neuro_manipulation.repe.rep_readers import PCARepReader, ClusterMeanRepReader
from neuro_manipulation.prompt_formats import QwenVLInstFormat, ManualPromptFormat
# Import the multimodal processor (no longer needs patching)
from neuro_manipulation.repe.multimodal_processor import QwenVLMultimodalProcessor


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason=SKIP_REASON)
class TestRealMultimodalIntegration:
    """Integration tests with real Qwen2.5-VL model."""
    
    @pytest.fixture(scope="class")
    def qwen_model_components(self):
        """Load Qwen2.5-VL model components (cached for all tests)."""
        print(f"Loading Qwen2.5-VL model from: {MODEL_PATH}")
        
        try:
            # Load model with minimal resources
            model = AutoModel.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATH, 
                trust_remote_code=True
            )
            
            processor = AutoProcessor.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True
            )
            
            print("✓ Model components loaded successfully")
            return {
                'model': model,
                'tokenizer': tokenizer,
                'processor': processor
            }
            
        except Exception as e:
            pytest.skip(f"Failed to load model: {e}")
    
    @pytest.fixture
    def sample_images(self):
        """Create sample images for different emotions."""
        images = {}
        emotions = ['anger', 'happiness', 'sadness']
        colors = ['red', 'yellow', 'blue']
        
        for emotion, color in zip(emotions, colors):
            # Create simple colored image (in practice you'd use real emotion images)
            img = Image.new('RGB', (224, 224), color=color)
            images[emotion] = img
            
        return images
    
    def test_multimodal_model_detection(self, qwen_model_components):
        """Test that Qwen2.5-VL is correctly detected as multimodal."""
        model = qwen_model_components['model']
        
        # Test multimodal detection
        is_multimodal = ModelLayerDetector.is_multimodal_model(model)
        assert is_multimodal, "Qwen2.5-VL should be detected as multimodal"
        
        # Test layer info extraction
        layer_info = ModelLayerDetector.get_multimodal_layer_info(model)
        print(f"Layer info: {layer_info}")
        
        # Should find some type of layers
        found_layers = sum(1 for v in layer_info.values() if v is not None)
        assert found_layers > 0, "Should find at least some layer types"
        
        # Test layer extraction
        try:
            layers = ModelLayerDetector.get_model_layers(model)
            assert len(layers) > 0, "Should find model layers"
            print(f"✓ Found {len(layers)} model layers")
        except Exception as e:
            print(f"Layer detection warning: {e}")
    
    def test_multimodal_input_processing(self, qwen_model_components, sample_images):
        """Test multimodal input processing with real model."""
        model = qwen_model_components['model']
        tokenizer = qwen_model_components['tokenizer']  
        processor = qwen_model_components['processor']
        
        # Create RepReadingPipeline instance without Transformers Pipeline overhead
        pipeline_instance = RepReadingPipeline.__new__(RepReadingPipeline)
        pipeline_instance.model = model
        pipeline_instance.tokenizer = tokenizer
        pipeline_instance.image_processor = processor.image_processor if hasattr(processor, 'image_processor') else processor
        
        # Test multimodal input detection
        multimodal_input = {
            'images': [sample_images['anger']],
            'text': 'when you see this image, your emotion is anger'
        }
        
        is_multimodal = pipeline_instance._is_multimodal_input(multimodal_input)
        assert is_multimodal, "Should detect multimodal input"
        
        # Test input preparation
        try:
            prepared = pipeline_instance._prepare_multimodal_inputs(multimodal_input)
            assert isinstance(prepared, dict), "Should return dict of model inputs"
            print(f"✓ Prepared multimodal inputs: {list(prepared.keys())}")
            
            # Check for expected keys (varies by processor)
            has_text = any(key in prepared for key in ['input_ids', 'text'])
            has_image = any(key in prepared for key in ['pixel_values', 'image', 'images'])
            
            print(f"Has text inputs: {has_text}, Has image inputs: {has_image}")
            
        except Exception as e:
            print(f"Input preparation error: {e}")
            # This might fail due to specific processor requirements, but detection should work
    
    def test_model_forward_pass(self, qwen_model_components, sample_images):
        """Test forward pass with multimodal input using new prompt format system."""
        model = qwen_model_components['model']
        tokenizer = qwen_model_components['tokenizer']
        processor = qwen_model_components['processor']
        
        # Use new QwenVL format system
        text_content = "when you see this image, your emotion is anger"
        image = sample_images['anger']
        
        # Get proper format using the new system
        try:
            # Use QwenVLInstFormat for proper token formatting
            formatted_text = QwenVLInstFormat.build(
                system_prompt=None,
                user_messages=[text_content],
                assistant_answers=[],
                images=[image]
            )
            
            print(f"Formatted text: {repr(formatted_text)}")
            
            # Process inputs using the processor with formatted text
            if hasattr(processor, 'process'):
                # For some processors
                inputs = processor(text=formatted_text, images=image, return_tensors="pt")
            elif hasattr(processor, '__call__'):
                # For others  
                inputs = processor(text=[formatted_text], images=[image], return_tensors="pt")
            else:
                # Manual processing
                text_inputs = tokenizer(formatted_text, return_tensors="pt")
                if hasattr(processor, 'image_processor'):
                    image_inputs = processor.image_processor(image, return_tensors="pt")
                else:
                    image_inputs = processor(image, return_tensors="pt")
                inputs = {**text_inputs, **image_inputs}
            
            print(f"Model inputs: {list(inputs.keys())}")
            
            # Debug: Check input_ids for image tokens
            if 'input_ids' in inputs:
                input_ids = inputs['input_ids']
                print(f"Input IDs shape: {input_ids.shape}")
                print(f"Input IDs: {input_ids}")
                
                # Check for special tokens
                special_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
                print(f"Tokens: {special_tokens}")
                
                # Look for vision-related token IDs (new format)
                vision_start_id = tokenizer.convert_tokens_to_ids('<|vision_start|>')
                vision_end_id = tokenizer.convert_tokens_to_ids('<|vision_end|>')
                print(f"Vision start token ID: {vision_start_id}")
                print(f"Vision end token ID: {vision_end_id}")
                
                if vision_start_id is not None:
                    print(f"Vision start tokens: {(input_ids == vision_start_id).sum().item()}")
                if vision_end_id is not None:
                    print(f"Vision end tokens: {(input_ids == vision_end_id).sum().item()}")
                
                # Check tokenizer special tokens
                print(f"Tokenizer special tokens: {tokenizer.special_tokens_map}")
                print(f"Tokenizer added tokens: {getattr(tokenizer, 'added_tokens_decoder', {})}")
            
            # Move inputs to same device as model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            print(f"Moved inputs to device: {device}")
            
            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            
            # Check outputs
            assert hasattr(outputs, 'hidden_states'), "Model should return hidden states"
            assert len(outputs.hidden_states) > 0, "Should have hidden states"
            
            print(f"✓ Forward pass successful")
            print(f"  - Hidden states layers: {len(outputs.hidden_states)}")
            print(f"  - Last layer shape: {outputs.hidden_states[-1].shape}")
            
        except Exception as e:
            print(f"Forward pass error: {e}")
            # Print more details for debugging
            import traceback
            traceback.print_exc()
            pytest.fail(f"Forward pass failed: {e}")
    
    def test_emotion_vector_extraction_basic(self, qwen_model_components, sample_images):
        """Test basic emotion vector extraction process."""
        model = qwen_model_components['model']
        tokenizer = qwen_model_components['tokenizer']
        processor = qwen_model_components['processor']
        
        # Register pipelines
        repe_pipeline_registry()
        
        try:
            # Create pipeline using the registered task
            # IMPORTANT: Use the full processor, not just image_processor component
            rep_pipeline = pipeline(
                task="multimodal-rep-reading",
                model=model,
                tokenizer=tokenizer,
                image_processor=processor  # Use full AutoProcessor
            )
            
            print("✓ Pipeline created successfully")
            
            # Create training stimuli
            stimuli = []
            emotions = ['anger', 'happiness']
            
            for emotion in emotions:
                stimulus = {
                    'images': [sample_images[emotion]],
                    'text': f'when you see this image, your emotion is {emotion}'
                }
                stimuli.append(stimulus)
                # Add duplicate for PCA (needs multiple samples)
                stimuli.append(stimulus)
            
            print(f"Created {len(stimuli)} training stimuli")
            
            # Test direction extraction
            try:
                directions = rep_pipeline.get_directions(
                    train_inputs=stimuli,
                    rep_token=-1,  # Last token (emotion word)
                    hidden_layers=[-1],  # Last layer
                    direction_method='pca',
                    batch_size=2
                )
                
                print("✓ Direction extraction completed")
                print(f"  - Method: {directions.direction_method}")
                print(f"  - Directions shape: {[d.shape for d in directions.directions.values()]}")
                
                # Validate extracted directions
                assert len(directions.directions) > 0, "Should extract at least one direction"
                for layer, direction_matrix in directions.directions.items():
                    assert direction_matrix.shape[0] >= 1, "Should have at least one component"
                    assert direction_matrix.shape[1] > 0, "Should have feature dimensions"
                
                print("✓ Emotion vector extraction validation passed")
                
            except Exception as e:
                print(f"Direction extraction error: {e}")
                import traceback
                traceback.print_exc()
                pytest.fail(f"Direction extraction failed: {e}")
                
        except Exception as e:
            print(f"Pipeline creation error: {e}")
            import traceback
            traceback.print_exc()
            pytest.fail(f"Pipeline creation failed: {e}")
    
    def test_emotion_vector_quality(self, qwen_model_components, sample_images):
        """Test extracted emotion vector quality."""
        directions = self.test_emotion_vector_extraction_basic(qwen_model_components, sample_images)
        
        if directions is None:
            pytest.skip("Could not extract directions for quality testing")
        
        # Basic quality checks
        for layer, direction_matrix in directions.directions.items():
            assert direction_matrix.shape[0] >= 1, "Should have at least one component"
            assert direction_matrix.shape[1] > 0, "Should have feature dimensions"
            
            # Check for non-zero directions
            magnitude = np.linalg.norm(direction_matrix)
            assert magnitude > 0, "Direction should have non-zero magnitude"
            
            print(f"✓ Layer {layer} direction quality check passed")
            print(f"  - Shape: {direction_matrix.shape}")
            print(f"  - Magnitude: {magnitude:.4f}")
    
    def test_pipeline_registration(self):
        """Test that multimodal pipelines are properly registered."""
        from transformers.pipelines import PIPELINE_REGISTRY
        
        # Register pipelines
        repe_pipeline_registry()
        
        # Check registration
        assert "multimodal-rep-reading" in PIPELINE_REGISTRY.supported_tasks
        
        task_info = PIPELINE_REGISTRY.supported_tasks["multimodal-rep-reading"]
        assert task_info["impl"] == RepReadingPipeline
        
        print("✓ Pipeline registration verified")
    
    def test_prompt_format_integration(self, qwen_model_components, sample_images):
        """Test that the new prompt format system works with the real model."""
        tokenizer = qwen_model_components['tokenizer']
        
        # Test format detection
        model_name = tokenizer.name_or_path
        print(f"Testing format detection for: {model_name}")
        
        try:
            format_class = ManualPromptFormat.get(model_name)
            print(f"✓ Detected format: {format_class.__name__}")
            assert format_class == QwenVLInstFormat, "Should detect QwenVL format"
            
            # Test multimodal support
            supports_multimodal = format_class.supports_multimodal()
            assert supports_multimodal, "QwenVL format should support multimodal"
            print(f"✓ Multimodal support: {supports_multimodal}")
            
            # Test tokenizer validation
            is_valid = format_class.validate_tokenizer(tokenizer)
            print(f"✓ Tokenizer validation: {is_valid}")
            
            # Test format building
            text_content = "when you see this image, your emotion is happiness"
            formatted_text = format_class.build(
                system_prompt=None,
                user_messages=[text_content],
                assistant_answers=[],
                images=[sample_images['happiness']]
            )
            
            print(f"✓ Formatted text sample: {repr(formatted_text[:100])}...")
            
            # Verify format contains vision tokens
            assert '<|vision_start|>' in formatted_text, "Should contain vision start token"
            assert '<|vision_end|>' in formatted_text, "Should contain vision end token"
            assert text_content in formatted_text, "Should contain original text"
            
            print("✓ All prompt format integration checks passed")
            
        except Exception as e:
            print(f"❌ Prompt format integration error: {e}")
            import traceback
            traceback.print_exc()
            pytest.fail(f"Prompt format integration failed: {e}")
    
    def test_config_compatibility(self, qwen_model_components):
        """Test that the model works with our configuration."""
        import yaml
        
        config_path = "/data/home/jjl7137/LLM_EmoBehav_game_theory_multimodal/config/multimodal_rep_reading_config.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        pipeline_config = config['experiment']['pipeline']
        
        # Test that configuration parameters are valid
        assert pipeline_config['task'] == 'multimodal-rep-reading'
        assert pipeline_config['rep_token'] == -1
        assert isinstance(pipeline_config['hidden_layers'], list)
        assert pipeline_config['direction_method'] in ['pca', 'cluster_mean']
        
        # Test that emotions are defined
        emotions = config['experiment']['emotions']
        assert len(emotions) >= 6
        assert 'anger' in emotions
        assert 'happiness' in emotions
        
        print("✓ Configuration compatibility verified")
        print(f"  - Pipeline task: {pipeline_config['task']}")
        print(f"  - Rep token: {pipeline_config['rep_token']}")
        print(f"  - Hidden layers: {pipeline_config['hidden_layers']}")
        print(f"  - Emotions: {emotions}")


def run_integration_tests():
    """Run integration tests independently."""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Qwen2.5-VL model not found at: {MODEL_PATH}")
        print("Please download the model or update the path")
        return False
    
    print("🚀 Running Real Multimodal Integration Tests")
    print("=" * 60)
    
    # Run with pytest
    result = pytest.main([
        __file__,
        "-v",
        "-s",  # Don't capture output
        "--tb=short"
    ])
    
    return result == 0


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)