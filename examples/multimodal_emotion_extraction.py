#!/usr/bin/env python3
"""
Multimodal Emotion Vector Extraction Example

This script demonstrates how to use the enhanced RepE pipeline to extract emotion vectors
from multimodal inputs (image + text prompts like "when you see this image, your emotion is anger").

The extracted emotion vectors can then be used to control text generation in game theory scenarios.
"""

import torch
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, AutoProcessor, AutoModel
from transformers.pipelines import pipeline
import yaml
from pathlib import Path

# Add project root to path
import sys
import os
sys.path.append('/data/home/jjl7137/LLM_EmoBehav_game_theory_multimodal')

from neuro_manipulation.repe.pipelines import repe_pipeline_registry
from neuro_manipulation.repe.rep_readers import PCARepReader, ClusterMeanRepReader


class MultimodalEmotionExtractor:
    """
    Extract emotion vectors from multimodal inputs using the RepE pipeline.
    """
    
    def __init__(self, model_path: str, config_path: str = None):
        """
        Initialize the multimodal emotion extractor.
        
        Args:
            model_path: Path to the multimodal model (e.g., Qwen2.5-VL)
            config_path: Path to configuration file (optional)
        """
        self.model_path = model_path
        self.config = self._load_config(config_path)
        self.pipeline = None
        self.emotion_vectors = {}
        
    def _load_config(self, config_path: str = None) -> dict:
        """Load configuration from file or use defaults."""
        if config_path is None:
            config_path = "/data/home/jjl7137/LLM_EmoBehav_game_theory_multimodal/config/multimodal_rep_reading_config.yaml"
            
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'experiment': {
                    'pipeline': {
                        'task': 'multimodal-rep-reading',
                        'rep_token': -1,
                        'hidden_layers': [-1, -2, -3],
                        'direction_method': 'pca',
                        'batch_size': 4
                    },
                    'emotions': ['anger', 'happiness', 'sadness', 'disgust', 'fear', 'surprise'],
                    'emotion_template': 'when you see this image, your emotion is'
                }
            }
    
    def setup_pipeline(self):
        """Initialize the multimodal RepE pipeline."""
        try:
            # Register custom pipelines
            repe_pipeline_registry()
            
            # Load model and tokenizer
            print(f"Loading model from: {self.model_path}")
            model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            
            # Create pipeline
            self.pipeline = pipeline(
                task=self.config['experiment']['pipeline']['task'],
                model=model,
                tokenizer=tokenizer,
                image_processor=processor.image_processor if hasattr(processor, 'image_processor') else processor
            )
            
            print("✓ Multimodal RepE pipeline initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup pipeline: {e}")
            return False
    
    def create_emotion_stimulus(self, image_path: str, emotion: str) -> dict:
        """
        Create a multimodal stimulus for emotion extraction.
        
        Args:
            image_path: Path to the image file
            emotion: Target emotion (e.g., 'anger', 'happiness')
            
        Returns:
            dict: Formatted multimodal input
        """
        try:
            image = Image.open(image_path).convert('RGB')
            template = self.config['experiment']['emotion_template']
            text = f"{template} {emotion}"
            
            return {
                'images': [image],
                'text': text
            }
        except Exception as e:
            print(f"❌ Failed to create stimulus for {image_path}: {e}")
            return None
    
    def extract_emotion_vectors(self, stimuli: list, emotion_labels: list = None):
        """
        Extract emotion vectors from a list of multimodal stimuli.
        
        Args:
            stimuli: List of multimodal inputs (image + text)
            emotion_labels: Optional labels for supervised extraction
            
        Returns:
            dict: Extracted emotion vectors by emotion type
        """
        if self.pipeline is None:
            print("❌ Pipeline not initialized. Call setup_pipeline() first.")
            return None
            
        try:
            pipeline_config = self.config['experiment']['pipeline']
            
            # Extract directions using the pipeline
            print("Extracting emotion directions from multimodal stimuli...")
            direction_finder = self.pipeline.get_directions(
                train_inputs=stimuli,
                rep_token=pipeline_config['rep_token'],
                hidden_layers=pipeline_config['hidden_layers'],
                direction_method=pipeline_config['direction_method'],
                batch_size=pipeline_config['batch_size'],
                train_labels=emotion_labels
            )
            
            # Store the extracted vectors
            self.emotion_vectors = {
                'direction_finder': direction_finder,
                'layers': pipeline_config['hidden_layers'],
                'method': pipeline_config['direction_method']
            }
            
            print("✓ Emotion vectors extracted successfully")
            print(f"  - Method: {pipeline_config['direction_method']}")
            print(f"  - Layers: {pipeline_config['hidden_layers']}")
            print(f"  - Stimuli processed: {len(stimuli)}")
            
            return self.emotion_vectors
            
        except Exception as e:
            print(f"❌ Failed to extract emotion vectors: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_emotion_vectors(self, output_path: str):
        """Save extracted emotion vectors to file."""
        if not self.emotion_vectors:
            print("❌ No emotion vectors to save")
            return False
            
        try:
            # Save directions as PyTorch tensors
            save_data = {
                'directions': {},
                'signs': getattr(self.emotion_vectors['direction_finder'], 'direction_signs', None),
                'config': self.config,
                'method': self.emotion_vectors['method'],
                'layers': self.emotion_vectors['layers']
            }
            
            # Convert numpy arrays to tensors for saving
            for layer, direction in self.emotion_vectors['direction_finder'].directions.items():
                if isinstance(direction, np.ndarray):
                    save_data['directions'][layer] = torch.tensor(direction)
                else:
                    save_data['directions'][layer] = direction
            
            torch.save(save_data, output_path)
            print(f"✓ Emotion vectors saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save emotion vectors: {e}")
            return False
    
    def load_emotion_vectors(self, input_path: str):
        """Load previously saved emotion vectors."""
        try:
            saved_data = torch.load(input_path)
            
            # Reconstruct direction finder
            if saved_data['method'] == 'pca':
                direction_finder = PCARepReader()
            elif saved_data['method'] == 'cluster_mean':
                direction_finder = ClusterMeanRepReader()
            else:
                print(f"❌ Unknown method: {saved_data['method']}")
                return False
            
            direction_finder.directions = saved_data['directions']
            direction_finder.direction_signs = saved_data.get('signs')
            
            self.emotion_vectors = {
                'direction_finder': direction_finder,
                'layers': saved_data['layers'],
                'method': saved_data['method']
            }
            
            print(f"✓ Emotion vectors loaded from: {input_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load emotion vectors: {e}")
            return False


def example_basic_extraction():
    """Example: Basic emotion vector extraction from sample images."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Emotion Vector Extraction")
    print("="*60)
    
    # Note: This example assumes you have a Qwen2.5-VL model available
    # You can modify the model path as needed
    model_path = "/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-VL-3B-Instruct"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("Please download a multimodal model or update the path")
        return False
    
    # Initialize extractor
    extractor = MultimodalEmotionExtractor(model_path)
    
    # Setup pipeline
    if not extractor.setup_pipeline():
        return False
    
    # Create sample stimuli (you would replace these with actual emotion-inducing images)
    stimuli = []
    emotions = ['anger', 'happiness', 'sadness']
    
    print("\nCreating sample stimuli...")
    for emotion in emotions:
        # Create a sample colored image for demonstration
        # In practice, you'd use real emotion-inducing images
        sample_image = Image.new('RGB', (224, 224), color='red' if emotion == 'anger' else 'yellow' if emotion == 'happiness' else 'blue')
        
        stimulus = {
            'images': [sample_image],
            'text': f'when you see this image, your emotion is {emotion}'
        }
        stimuli.append(stimulus)
        print(f"  ✓ Created stimulus for {emotion}")
    
    # Extract emotion vectors
    emotion_vectors = extractor.extract_emotion_vectors(stimuli)
    
    if emotion_vectors:
        # Save the vectors
        output_path = "results/sample_emotion_vectors.pt"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        extractor.save_emotion_vectors(output_path)
        
        print(f"\n✅ Example completed successfully!")
        print(f"Emotion vectors saved to: {output_path}")
        return True
    else:
        print("❌ Example failed")
        return False


def example_with_real_images():
    """Example: Extract emotion vectors from real image dataset."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Extraction from Real Images")
    print("="*60)
    
    # This example assumes you have a directory with emotion images
    image_dir = "data/emotion_images"
    
    if not os.path.exists(image_dir):
        print(f"❌ Image directory not found: {image_dir}")
        print("This example requires a collection of emotion-inducing images")
        print("Skipping real image example...")
        return True  # Not a failure, just skip
    
    model_path = "/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-VL-3B-Instruct"
    extractor = MultimodalEmotionExtractor(model_path)
    
    if not extractor.setup_pipeline():
        return False
    
    # Process images from directory
    stimuli = []
    emotions = ['anger', 'happiness', 'sadness', 'disgust', 'fear', 'surprise']
    
    for emotion in emotions:
        emotion_dir = os.path.join(image_dir, emotion)
        if os.path.exists(emotion_dir):
            for img_file in os.listdir(emotion_dir)[:5]:  # Max 5 images per emotion
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    img_path = os.path.join(emotion_dir, img_file)
                    stimulus = extractor.create_emotion_stimulus(img_path, emotion)
                    if stimulus:
                        stimuli.append(stimulus)
    
    if stimuli:
        print(f"Processing {len(stimuli)} real images...")
        emotion_vectors = extractor.extract_emotion_vectors(stimuli)
        
        if emotion_vectors:
            output_path = "results/real_emotion_vectors.pt"
            extractor.save_emotion_vectors(output_path)
            print(f"✅ Real image processing completed!")
            return True
    
    print("❌ No suitable images found")
    return False


def example_integration_with_game_theory():
    """Example: How to integrate extracted vectors with game theory experiments."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Integration with Game Theory")
    print("="*60)
    
    # This shows how you would use extracted emotion vectors in game experiments
    print("Integration workflow:")
    print("1. Extract emotion vectors using MultimodalEmotionExtractor")
    print("2. Load vectors in your game experiment")
    print("3. Apply vectors using RepControlVLLMHook during text generation")
    print("4. Measure behavioral changes in game responses")
    
    # Example code structure (pseudo-code)
    integration_example = """
    # In your game experiment:
    from neuro_manipulation.repe.rep_control_vllm_hook import RepControlVLLMHook
    
    # Load previously extracted emotion vectors
    extractor = MultimodalEmotionExtractor(model_path)
    extractor.load_emotion_vectors("results/emotion_vectors.pt")
    
    # Setup vLLM hook for game text generation
    hook = RepControlVLLMHook(
        model=vllm_model,
        tokenizer=tokenizer,
        layers=extractor.emotion_vectors['layers'],
        block_name="decoder_block",
        control_method="reading_vec"
    )
    
    # Apply emotion vector during game scenario generation
    hook.set_controller(
        direction=extractor.emotion_vectors['direction_finder'].directions,
        intensity=1.5,
        emotion="anger"  # or other emotions
    )
    
    # Generate game responses with emotional influence
    responses = model.generate(game_prompts, hooks=[hook])
    """
    
    print("\nExample integration code:")
    print(integration_example)
    print("\n✅ Integration example complete!")
    return True


def main():
    """Run all multimodal emotion extraction examples."""
    print("MULTIMODAL EMOTION EXTRACTION EXAMPLES")
    print("="*60)
    print("These examples demonstrate how to extract emotion vectors from")
    print("multimodal inputs (image + text) for game theory experiments.")
    print()
    
    # Run examples
    examples = [
        ("Basic Extraction", example_basic_extraction),
        ("Real Images", example_with_real_images),
        ("Game Integration", example_integration_with_game_theory)
    ]
    
    results = []
    for name, example_func in examples:
        try:
            success = example_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("EXAMPLE SUMMARY")
    print("="*60)
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name:20} {status}")
    
    print(f"\nOverall: {sum(r[1] for r in results)}/{len(results)} examples completed successfully")
    
    if all(r[1] for r in results):
        print("\n🎉 All examples completed! The multimodal RepE system is ready to use.")
    else:
        print("\n⚠️  Some examples had issues. Check model paths and dependencies.")


if __name__ == "__main__":
    main()