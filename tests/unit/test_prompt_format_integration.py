#!/usr/bin/env python3
"""
Test the integrated prompt format approach for multimodal RepE.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

from neuro_manipulation.prompt_formats import QwenVLInstFormat, ManualPromptFormat
from PIL import Image

def test_qwen_format_basic():
    """Test basic QwenVL format functionality."""
    print("=" * 60)
    print("Testing QwenVL Format Basic Functionality")
    print("=" * 60)
    
    # Test name pattern matching
    test_models = [
        "Qwen2.5-VL-3B-Instruct",
        "Qwen2.5-VL-7B-Instruct", 
        "/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-VL-3B-Instruct",
        "llama-3.1-instruct",
        "mistral-7b-instruct"
    ]
    
    for model in test_models:
        matches = QwenVLInstFormat.name_pattern(model)
        print(f"Model: {model} -> Matches QwenVL: {matches}")
    
    print("\n" + "-" * 40)
    
    # Test text-only formatting
    text_only = QwenVLInstFormat.build(
        system_prompt="You are a helpful assistant",
        user_messages=["What is the weather today?"],
        assistant_answers=[]
    )
    print("Text-only format:")
    print(repr(text_only))
    print()
    
    # Test multimodal formatting  
    multimodal = QwenVLInstFormat.build(
        system_prompt="You are a helpful assistant",
        user_messages=["when you see this image, your emotion is anger"],
        assistant_answers=[],
        images=[Image.new('RGB', (224, 224), 'red')]
    )
    print("Multimodal format:")
    print(repr(multimodal))
    print()
    
    return True

def test_manual_format_detection():
    """Test that ManualPromptFormat can find QwenVL format."""
    print("=" * 60)
    print("Testing Manual Format Detection")
    print("=" * 60)
    
    model_name = "Qwen2.5-VL-3B-Instruct"
    
    try:
        format_class = ManualPromptFormat.get(model_name)
        print(f"Found format: {format_class.__name__}")
        print(f"Supports multimodal: {format_class.supports_multimodal()}")
        
        # Test building with the detected format
        result = format_class.build(
            system_prompt=None,
            user_messages=["when you see this image, your emotion is happiness"],
            assistant_answers=[],
            images=[Image.new('RGB', (224, 224), 'yellow')]
        )
        print("Generated prompt:")
        print(repr(result))
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_tokenizer_validation():
    """Test tokenizer validation with real Qwen-VL tokenizer."""
    print("=" * 60)
    print("Testing Tokenizer Validation")
    print("=" * 60)
    
    model_path = "/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-VL-3B-Instruct"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        return False
    
    try:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"Loaded tokenizer: {tokenizer.__class__.__name__}")
        
        # Test validation
        is_valid = QwenVLInstFormat.validate_tokenizer(tokenizer)
        print(f"Tokenizer validation result: {is_valid}")
        
        # Check specific tokens
        required_tokens = ['<|vision_start|>', '<|vision_end|>', '<|image_pad|>']
        for token in required_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            print(f"Token '{token}' -> ID: {token_id}")
        
        return is_valid
        
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return False

def test_integrated_emotion_prompt():
    """Test the integrated approach with emotion extraction prompts."""
    print("=" * 60)
    print("Testing Integrated Emotion Prompt Generation")
    print("=" * 60)
    
    emotions = ['anger', 'happiness', 'sadness', 'disgust', 'fear', 'surprise']
    
    for emotion in emotions:
        # Create multimodal emotion stimulus
        result = QwenVLInstFormat.build(
            system_prompt=None,
            user_messages=[f"when you see this image, your emotion is {emotion}"],
            assistant_answers=[],
            images=[Image.new('RGB', (224, 224), 'blue')]
        )
        
        print(f"Emotion: {emotion}")
        print(f"Prompt: {repr(result)}")
        print()
    
    return True

def run_all_tests():
    """Run all prompt format integration tests."""
    print("🚀 PROMPT FORMAT INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        ("Basic QwenVL Format", test_qwen_format_basic),
        ("Manual Format Detection", test_manual_format_detection),
        ("Tokenizer Validation", test_tokenizer_validation),
        ("Integrated Emotion Prompts", test_integrated_emotion_prompt)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{status}: {name}\n")
        except Exception as e:
            print(f"❌ FAILED: {name} - {e}\n")
            results.append((name, False))
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name:30} {status}")
    
    total_passed = sum(r[1] for r in results)
    print(f"\nOverall: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("\n🎉 All tests passed! Prompt format integration is working.")
    else:
        print(f"\n⚠️  {len(results) - total_passed} tests failed. Check implementation.")
    
    return total_passed == len(results)

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)