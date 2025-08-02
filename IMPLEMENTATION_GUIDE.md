# Qwen2.5-VL AutoProcessor Implementation Guide

## Problem Summary

Your current implementation has a token/feature mismatch issue:
- **Current**: Text formatting adds 2 `<|image_pad|>` tokens, but image processor produces 128 image features
- **Root Cause**: Processing text and images separately instead of using the unified AutoProcessor
- **Solution**: Use Qwen2.5-VL's proper message format and unified processing pipeline

## Required Changes

### 1. Install Dependencies (if not already installed)

```bash
# Optional but recommended for optimal performance
pip install qwen-vl-utils
```

### 2. Update Model Loading Code

**Current approach:**
```python
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# Using separate components
pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    image_processor=processor.image_processor  # ❌ This causes the issue
)
```

**Fixed approach:**
```python
# Load the complete AutoProcessor
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
tokenizer = processor.tokenizer  # For backward compatibility

# Use the full processor, not just the image_processor component
pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    image_processor=processor  # ✅ Use the complete processor
)
```

### 3. Update RepReadingPipeline (Quick Fix)

Add this to your code before using the pipeline:

```python
# Quick fix - apply the patch
from neuro_manipulation.repe.multimodal_processor_fix import patch_rep_reading_pipeline
patch_rep_reading_pipeline()

# Now use your pipeline normally
# It will automatically use the correct processing method
```

### 4. Update RepReadingPipeline (Permanent Fix)

**File:** `/data/home/jjl7137/LLM_EmoBehav_game_theory_multimodal/neuro_manipulation/repe/rep_reading_pipeline.py`

Replace the `_prepare_multimodal_inputs` method:

```python
def _prepare_multimodal_inputs(self, inputs: Union[Dict, List], **tokenizer_kwargs) -> Dict[str, Any]:
    """Prepare multimodal inputs using the correct Qwen2.5-VL processor format."""
    
    if isinstance(inputs, dict):
        images = inputs.get('images', inputs.get('image'))
        text = inputs.get('text', inputs.get('prompt', ''))
        
        if not isinstance(images, list) and images is not None:
            images = [images]
        
        try:
            # Create proper message format for Qwen2.5-VL
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
            
            # Apply chat template (this handles token formatting correctly)
            formatted_text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Try with qwen_vl_utils first (preferred)
            try:
                from qwen_vl_utils import process_vision_info
                
                # Extract vision information
                image_inputs, video_inputs = process_vision_info(messages)
                
                # Use unified processor
                model_inputs = self.image_processor(  # This should be the full AutoProcessor
                    text=[formatted_text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    **tokenizer_kwargs
                )
                
            except ImportError:
                # Fallback without qwen_vl_utils
                model_inputs = self.image_processor(  # This should be the full AutoProcessor
                    text=[formatted_text],
                    images=images,
                    padding=True,
                    return_tensors="pt",
                    **tokenizer_kwargs
                )
            
            return model_inputs
            
        except Exception as e:
            print(f"Unified processing failed: {e}")
            # Keep your existing fallback logic here
            
    elif isinstance(inputs, list):
        # Handle batch processing
        if inputs:
            # Process first item for now (TODO: proper batching)
            return self._prepare_multimodal_inputs(inputs[0], **tokenizer_kwargs)
    
    return {}
```

### 5. Update QwenVLInstFormat (Optional)

**File:** `/data/home/jjl7137/LLM_EmoBehav_game_theory_multimodal/neuro_manipulation/prompt_formats.py`

Add a new method to QwenVLInstFormat:

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

## Testing the Fix

### Quick Test Script

```python
# Test the fix
from neuro_manipulation.repe.multimodal_processor_fix import patch_rep_reading_pipeline
from transformers import AutoModel, AutoProcessor, pipeline
from PIL import Image

# Apply the fix
patch_rep_reading_pipeline()

# Load model components correctly
model_path = "/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")

# Create pipeline with the full processor
rep_pipeline = pipeline(
    task="multimodal-rep-reading",
    model=model,
    tokenizer=processor.tokenizer,
    image_processor=processor  # ✅ Use full processor
)

# Test with your exact use case
test_image = Image.new('RGB', (224, 224), color='red')
multimodal_input = {
    'images': [test_image],
    'text': 'when you see this image, your emotion is anger'
}

# This should now work without token/feature mismatch
try:
    result = rep_pipeline._prepare_multimodal_inputs(multimodal_input)
    print("✅ Success! No token/feature mismatch")
    print(f"Input keys: {list(result.keys())}")
    
    # Test forward pass
    with torch.no_grad():
        outputs = model(**result, output_hidden_states=True)
    print("✅ Forward pass successful!")
    
except Exception as e:
    print(f"❌ Still has issues: {e}")
```

## Key Points to Remember

1. **Use Full AutoProcessor**: Don't split into tokenizer + image_processor components
2. **Proper Message Format**: Use the role-based content structure with type specifications
3. **Apply Chat Template**: Let the processor handle token formatting, don't manually add `<|image_pad|>`
4. **Unified Processing**: Process text and images together, not separately
5. **Optional qwen_vl_utils**: Install for optimal performance, but fallback works without it

## Expected Results

After implementing this fix:
- ✅ No more "tokens: 2, features: 128" mismatches
- ✅ Automatic token expansion to match image features
- ✅ Proper alignment between text and image modalities
- ✅ Successful forward passes without tensor size errors
- ✅ Your emotion extraction pipeline should work correctly

## Next Steps

1. Apply the quick fix using `patch_rep_reading_pipeline()`
2. Test with your existing emotion extraction scenarios
3. If working well, implement the permanent fix in your pipeline
4. Consider installing `qwen_vl_utils` for optimal performance
5. Update your documentation and examples to use the correct approach

The core issue was treating text and image processing as separate steps when Qwen2.5-VL requires unified processing through its AutoProcessor. This fix aligns with the official Qwen2.5-VL usage patterns and should resolve your token/feature mismatch completely.