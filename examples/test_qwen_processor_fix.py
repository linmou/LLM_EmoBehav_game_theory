#!/usr/bin/env python3
"""
Test script to demonstrate the correct Qwen2.5-VL AutoProcessor usage
and fix the image token expansion issue.

This script provides both the official approach (with qwen_vl_utils) 
and a fallback approach (without qwen_vl_utils).
"""

import torch
from PIL import Image
import sys
import os

# Add project root to path
sys.path.append('/data/home/jjl7137/LLM_EmoBehav_game_theory_multimodal')

from transformers import AutoModel, AutoProcessor, AutoTokenizer


class QwenVLProcessorHelper:
    """Helper class to correctly use Qwen2.5-VL AutoProcessor."""
    
    def __init__(self, model_path: str):
        """Initialize with model components."""
        self.model_path = model_path
        self.load_components()
    
    def load_components(self):
        """Load model, tokenizer, and processor."""
        print(f"Loading Qwen2.5-VL components from: {self.model_path}")
        
        try:
            # Load the complete AutoProcessor (includes tokenizer + image processor)
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            # Access tokenizer from processor for compatibility
            self.tokenizer = self.processor.tokenizer
            
            print("✓ Components loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load components: {e}")
            raise
    
    def create_messages_format(self, text: str, images: list = None) -> list:
        """
        Create the correct message format for Qwen2.5-VL.
        
        Args:
            text: Text content
            images: List of PIL Images or image paths
            
        Returns:
            List of messages in the correct format
        """
        content = []
        
        # Add images first
        if images:
            for image in images:
                content.append({
                    "type": "image",
                    "image": image
                })
        
        # Add text
        content.append({
            "type": "text",
            "text": text
        })
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        return messages
    
    def process_with_official_utils(self, messages: list) -> dict:
        """
        Process using the official qwen_vl_utils (preferred method).
        
        Args:
            messages: Messages in the correct format
            
        Returns:
            Processed inputs ready for model
        """
        try:
            from qwen_vl_utils import process_vision_info
            
            # Step 1: Apply chat template
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Step 2: Extract vision information
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Step 3: Process everything together
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            print("✓ Processed using official qwen_vl_utils")
            return inputs
            
        except ImportError:
            raise ImportError("qwen_vl_utils not available. Use fallback method.")
    
    def process_with_fallback(self, messages: list) -> dict:
        """
        Fallback processing method without qwen_vl_utils.
        
        This method manually extracts images and uses the processor's
        components directly, but still leverages the chat template.
        
        Args:
            messages: Messages in the correct format
            
        Returns:
            Processed inputs ready for model
        """
        try:
            # Extract images from messages
            images = []
            text_parts = []
            
            for message in messages:
                if message.get("role") == "user":
                    for content_item in message.get("content", []):
                        if content_item.get("type") == "image":
                            images.append(content_item.get("image"))
                        elif content_item.get("type") == "text":
                            text_parts.append(content_item.get("text"))
            
            # Apply chat template (this handles the token formatting correctly)
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            print(f"Chat template output: {repr(text[:200])}...")
            
            # Process text and images together using the unified processor
            # This is the key: use the processor as a unified component
            inputs = self.processor(
                text=[text],
                images=images if images else None,
                padding=True,
                return_tensors="pt"
            )
            
            print("✓ Processed using fallback method")
            return inputs
            
        except Exception as e:
            print(f"❌ Fallback processing failed: {e}")
            raise
    
    def process_multimodal_input(self, text: str, images: list = None) -> dict:
        """
        Main processing method that tries official method first, then fallback.
        
        Args:
            text: Text content
            images: List of PIL Images or image paths
            
        Returns:
            Processed inputs ready for model
        """
        # Create proper message format
        messages = self.create_messages_format(text, images)
        
        print(f"Created messages: {messages}")
        
        # Try official method first
        try:
            return self.process_with_official_utils(messages)
        except ImportError:
            print("qwen_vl_utils not available, using fallback method")
            return self.process_with_fallback(messages)
    
    def test_forward_pass(self, inputs: dict) -> bool:
        """
        Test forward pass to verify no token/feature mismatches.
        
        Args:
            inputs: Processed inputs from process_multimodal_input
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Move inputs to model device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            print(f"Input keys: {list(inputs.keys())}")
            
            # Debug input shapes
            if 'input_ids' in inputs:
                print(f"Input IDs shape: {inputs['input_ids'].shape}")
                print(f"Input IDs: {inputs['input_ids']}")
                
                # Decode tokens to see the formatted text
                tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                print(f"Tokens: {tokens[:20]}...")  # Show first 20 tokens
            
            if 'pixel_values' in inputs:
                print(f"Pixel values shape: {inputs['pixel_values'].shape}")
            elif 'image_grid_thw' in inputs:
                print(f"Image grid shape: {inputs['image_grid_thw'].shape}")
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            print("✓ Forward pass successful - no token/feature mismatch!")
            print(f"  - Output keys: {list(outputs.keys()) if hasattr(outputs, 'keys') else 'ModelOutput'}")
            print(f"  - Hidden states layers: {len(outputs.hidden_states) if hasattr(outputs, 'hidden_states') else 'N/A'}")
            
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                print(f"  - Last layer shape: {outputs.hidden_states[-1].shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_emotion_extraction_scenario():
    """Test the emotion extraction scenario that was causing issues."""
    print("\n" + "="*80)
    print("TESTING EMOTION EXTRACTION SCENARIO")
    print("="*80)
    
    model_path = "/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-VL-3B-Instruct"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        return False
    
    # Initialize helper
    helper = QwenVLProcessorHelper(model_path)
    
    # Create test image
    test_image = Image.new('RGB', (224, 224), color='red')
    
    # Test the exact scenario from your use case
    text = "when you see this image, your emotion is anger"
    images = [test_image]
    
    print(f"Testing with text: '{text}'")
    print(f"Testing with {len(images)} image(s)")
    
    # Process the input
    try:
        inputs = helper.process_multimodal_input(text, images)
        
        # Test forward pass
        success = helper.test_forward_pass(inputs)
        
        if success:
            print("\n🎉 SUCCESS: The image token expansion issue has been resolved!")
            print("The processor correctly aligns text tokens with image features.")
            return True
        else:
            print("\n❌ FAILED: Forward pass still has issues")
            return False
            
    except Exception as e:
        print(f"\n❌ FAILED: Processing error: {e}")
        import traceback
        traceback.print_exc()
        return False


def demonstrate_token_alignment():
    """Demonstrate how the correct approach aligns tokens and features."""
    print("\n" + "="*80)
    print("DEMONSTRATING TOKEN-FEATURE ALIGNMENT")
    print("="*80)
    
    model_path = "/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-VL-3B-Instruct"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        return
    
    helper = QwenVLProcessorHelper(model_path)
    
    # Create test case
    test_image = Image.new('RGB', (224, 224), color='blue')
    text = "what emotion does this image evoke?"
    
    # Show the correct approach
    messages = helper.create_messages_format(text, [test_image])
    
    print("1. Correct message format:")
    print(f"   {messages}")
    
    print("\n2. Processing with unified AutoProcessor:")
    inputs = helper.process_multimodal_input(text, [test_image])
    
    print("\n3. Key insight:")
    print("   - The processor automatically expands image placeholders")
    print("   - Text tokens and image features are correctly aligned")
    print("   - No manual <|image_pad|> token management needed")
    
    if 'input_ids' in inputs:
        num_tokens = inputs['input_ids'].shape[1]
        print(f"   - Total tokens: {num_tokens}")
    
    if 'pixel_values' in inputs:
        image_features = inputs['pixel_values'].shape
        print(f"   - Image feature shape: {image_features}")
    elif 'image_grid_thw' in inputs:
        print(f"   - Image grid info: {inputs['image_grid_thw']}")


def main():
    """Run all tests and demonstrations."""
    print("QWEN2.5-VL AUTOPROCESSOR CORRECT USAGE DEMONSTRATION")
    print("="*80)
    print("This script demonstrates the correct way to use Qwen2.5-VL")
    print("AutoProcessor to avoid image token expansion issues.")
    print()
    
    # Test the specific scenario
    success = test_emotion_extraction_scenario()
    
    # Demonstrate the concepts
    demonstrate_token_alignment()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if success:
        print("✅ SOLUTION VERIFIED: The AutoProcessor approach resolves the token mismatch issue")
        print()
        print("Key points:")
        print("1. Use the unified AutoProcessor instead of separate tokenizer + image_processor")
        print("2. Format inputs using the proper message structure")
        print("3. Let the processor handle token expansion automatically")
        print("4. Apply chat template before processing")
        print()
        print("Next steps:")
        print("- Update your RepReadingPipeline to use this approach")
        print("- Install qwen_vl_utils for optimal performance: pip install qwen-vl-utils")
        print("- Test with your actual emotion extraction scenarios")
        
    else:
        print("❌ Issues detected. Check:")
        print("1. Model path is correct")
        print("2. Dependencies are installed")
        print("3. Model is compatible with AutoProcessor")
    
    print(f"\nFor detailed implementation guidance, see:")
    print(f"/data/home/jjl7137/LLM_EmoBehav_game_theory_multimodal/docs/QWEN_VL_PROCESSOR_USAGE_GUIDE.md")


if __name__ == "__main__":
    main()