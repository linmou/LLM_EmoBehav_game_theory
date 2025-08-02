# Qwen2.5-VL AutoProcessor Correct Usage Guide

## Current Issue Analysis

Based on your codebase analysis, the current issue stems from:

1. **Separate Processing**: Text and images are being processed separately using different components (tokenizer for text, image_processor for images)
2. **Token Mismatch**: The formatted text contains `<|image_pad|>` tokens (2 tokens) but the image processor produces 128 image features, causing a mismatch
3. **Missing Integration**: The AutoProcessor's unified processing pipeline is not being used correctly

## Correct AutoProcessor Usage for Qwen2.5-VL

### 1. Required Dependencies

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
```

**Note**: The `qwen_vl_utils` package is essential for proper vision processing.

### 2. Correct Message Format

Qwen2.5-VL expects a specific message structure:

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path_or_pil_image  # Can be URL, file path, or PIL Image
            },
            {
                "type": "text", 
                "text": "when you see this image, your emotion is anger"
            }
        ]
    }
]
```

### 3. Complete Processing Pipeline

```python
# Step 1: Apply chat template to convert messages to text
text = processor.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

# Step 2: Extract vision information
image_inputs, video_inputs = process_vision_info(messages)

# Step 3: Process everything together
inputs = processor(
    text=[text],  # Note: text must be wrapped in a list
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
)
```

### 4. Key Differences from Current Implementation

| Current Approach | Correct Approach |
|------------------|------------------|
| `tokenizer(formatted_text)` + `image_processor(images)` | `processor(text=[text], images=image_inputs)` |
| Manual token formatting with `<|image_pad|>` | Let processor handle token expansion automatically |
| Separate processing and manual combination | Unified processing with automatic alignment |

## Implementation Fix for Your Codebase

### 1. Update RepReadingPipeline

Modify `/data/home/jjl7137/LLM_EmoBehav_game_theory_multimodal/neuro_manipulation/repe/rep_reading_pipeline.py`:

```python
def _prepare_multimodal_inputs(self, inputs: Union[Dict, List], **tokenizer_kwargs) -> Dict[str, Any]:
    """Prepare multimodal inputs using the correct Qwen2.5-VL processor format."""
    
    if isinstance(inputs, dict):
        images = inputs.get('images', inputs.get('image'))
        text = inputs.get('text', inputs.get('prompt', ''))
        
        if not isinstance(images, list) and images is not None:
            images = [images]
        
        try:
            # Import qwen_vl_utils if available
            from qwen_vl_utils import process_vision_info
            
            # Create proper message format for Qwen2.5-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img
                        } for img in (images or [])
                    ] + [
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
                }
            ]
            
            # Apply chat template
            formatted_text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process vision info
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Use unified processor
            if hasattr(self.image_processor, 'apply_chat_template'):
                # This is the full AutoProcessor
                model_inputs = self.image_processor(
                    text=[formatted_text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    **tokenizer_kwargs
                )
            else:
                # Fallback for separate components
                model_inputs = self.tokenizer(
                    formatted_text, 
                    return_tensors="pt", 
                    **tokenizer_kwargs
                )
                if image_inputs:
                    img_inputs = self.image_processor(
                        image_inputs, 
                        return_tensors="pt"
                    )
                    model_inputs.update(img_inputs)
            
            return model_inputs
            
        except ImportError:
            print("qwen_vl_utils not available, falling back to manual processing")
            # Your current fallback logic here
            
        except Exception as e:
            print(f"Unified processing failed: {e}")
            # Your current fallback logic here
    
    return {}
```

### 2. Update QwenVLInstFormat

The QwenVLInstFormat should not manually add `<|image_pad|>` tokens when using the proper processor:

```python
@staticmethod
def build_for_processor(system_prompt, user_messages: list, assistant_answers: list = [], images: list = None):
    """
    Build messages in the format expected by Qwen2.5-VL AutoProcessor.
    This bypasses manual token formatting and lets the processor handle it.
    """
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    for i, msg in enumerate(user_messages):
        content = []
        
        # Add images if available for this message
        if images and i < len(images):
            content.append({
                "type": "image",
                "image": images[i]
            })
        
        # Add text content
        content.append({
            "type": "text",
            "text": msg
        })
        
        messages.append({
            "role": "user",
            "content": content
        })
        
        # Add assistant response if available
        if i < len(assistant_answers):
            messages.append({
                "role": "assistant",
                "content": assistant_answers[i]
            })
    
    return messages
```

### 3. Update Model Loading

Ensure you're loading the complete AutoProcessor:

```python
# In your model loading code
from transformers import AutoModel, AutoProcessor, AutoTokenizer

# Load the full processor (not just tokenizer + image_processor)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# The processor contains both tokenizer and image_processor
# But should be used as a unified component
tokenizer = processor.tokenizer  # For backward compatibility
image_processor = processor      # Use the full processor
```

## Why This Approach Works

1. **Automatic Token Expansion**: The processor automatically expands image placeholders to match the actual number of image features
2. **Proper Alignment**: Text and image tokens are correctly aligned during processing
3. **No Manual Token Management**: Eliminates the need to manually handle `<|image_pad|>` tokens
4. **Follows Official API**: Uses the intended Qwen2.5-VL processing pipeline

## Testing the Fix

```python
# Test the corrected approach
def test_correct_processing():
    from PIL import Image
    
    # Create test data
    image = Image.new('RGB', (224, 224), color='red')
    text = "when you see this image, your emotion is anger"
    
    # Create proper message format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text}
            ]
        }
    ]
    
    # Process correctly
    formatted_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)  
    inputs = processor(
        text=[formatted_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    
    # Check that tokens and features align
    print(f"Input shape: {inputs['input_ids'].shape}")
    if 'pixel_values' in inputs:
        print(f"Image features: {inputs['pixel_values'].shape}")
    
    # Forward pass should work without token/feature mismatch
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    print("✓ Processing successful - no token/feature mismatch")
```

## Summary

The key insight is that Qwen2.5-VL's AutoProcessor must be used as a unified component that handles both text and image processing together. Manual token formatting and separate processing leads to misalignment between text tokens and image features. The correct approach uses the official message format and lets the processor handle token expansion automatically.