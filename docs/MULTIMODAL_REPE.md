# Multimodal Representation Engineering (RepE) Documentation

## Overview

The multimodal RepE extension enables extraction of emotion vectors from combined image+text inputs, allowing for sophisticated emotion manipulation in language model generation. Instead of using pure text or pure images, this approach captures authentic emotional responses through multimodal stimuli like `"[IMAGE] when you see this image, your emotion is anger"`.

## Key Features

- **Multimodal Input Processing**: Handle combined image+text inputs seamlessly
- **Existing Infrastructure Reuse**: Leverages current PCA and cluster-mean RepReaders
- **Multimodal Model Support**: Enhanced detection and layer selection for vision-language models
- **Flexible Configuration**: YAML-based configuration system for different experiments
- **vLLM Integration**: Compatible with existing vLLM hooks for text generation control

## Architecture

```
Image + Emotion Inquiry Text → Multimodal Model → 
Hidden States at Emotion Token → Existing RepReaders → 
Emotion Vectors → vLLM Hook Control
```

## Quick Start

### 1. Basic Emotion Vector Extraction

```python
from examples.multimodal_emotion_extraction import MultimodalEmotionExtractor
from PIL import Image

# Initialize extractor with Qwen-VL model
extractor = MultimodalEmotionExtractor(
    model_path="/path/to/Qwen2.5-VL-3B-Instruct"
)
extractor.setup_pipeline()

# Create multimodal stimulus
image = Image.open("angry_face.jpg")
stimulus = {
    'images': [image],
    'text': 'when you see this image, your emotion is anger'
}

# Extract emotion vectors
emotion_vectors = extractor.extract_emotion_vectors([stimulus])

# Save for later use
extractor.save_emotion_vectors("anger_vectors.pt")
```

### 2. Use in Game Theory Experiments

```python
from neuro_manipulation.repe.rep_control_vllm_hook import RepControlVLLMHook

# Load previously extracted vectors
extractor.load_emotion_vectors("anger_vectors.pt")

# Apply to vLLM model during game generation
hook = RepControlVLLMHook(vllm_model, tokenizer, layers, "decoder_block", "reading_vec")
hook.set_controller(
    direction=extractor.emotion_vectors['direction_finder'].directions,
    intensity=1.5
)

# Generate game responses with emotional influence
responses = model.generate(game_prompts, hooks=[hook])
```

## Configuration

### Multimodal Pipeline Configuration

```yaml
experiment:
  name: "Multimodal_Emotion_Extraction"
  
  pipeline:
    task: "multimodal-rep-reading"
    rep_token: -1  # Extract from emotion word (last token)
    hidden_layers: [-1, -2, -3]  # Last 3 layers
    direction_method: "pca"  # or "cluster_mean"
    batch_size: 4
    
  emotions:
    - "anger"
    - "happiness"
    - "sadness"
    - "disgust"
    - "fear" 
    - "surprise"
    
  emotion_template: "when you see this image, your emotion is"
```

### Model Configuration

```yaml
model_config:
  device_map: "auto"
  torch_dtype: "bfloat16"
  trust_remote_code: true

vllm_config:
  task: "multimodal-rep-reading-vllm"
  tensor_parallel_size: 1
  max_model_len: 2048
  gpu_memory_utilization: 0.8
```

## Supported Models

### Tested Multimodal Architectures
- **Qwen2.5-VL series** (3B, 7B, 72B variants)
- **LLaVA family** (experimental)
- **BLIP-2** (experimental)

### Model Detection

The system automatically detects multimodal models and selects appropriate layers:

```python
from neuro_manipulation.model_layer_detector import ModelLayerDetector

# Automatic detection
is_multimodal = ModelLayerDetector.is_multimodal_model(model)
layer_info = ModelLayerDetector.get_multimodal_layer_info(model)

# Prioritizes text/language layers for emotion extraction
# Falls back to fusion layers if text layers unavailable
```

## Input Formats

### Multimodal Dictionary Format
```python
multimodal_input = {
    'images': [PIL_image1, PIL_image2],  # List of PIL Images
    'text': 'when you see this image, your emotion is anger'
}

# Or singular image
multimodal_input = {
    'image': PIL_image,  # Single PIL Image  
    'text': 'when you see this image, your emotion is happiness'
}
```

### Batch Processing
```python
batch_inputs = [
    {'images': [angry_image], 'text': 'when you see this image, your emotion is anger'},
    {'images': [happy_image], 'text': 'when you see this image, your emotion is happiness'},
    {'images': [sad_image], 'text': 'when you see this image, your emotion is sadness'}
]
```

## RepE Stimulus Datasets (On Disk)

The RepE stimulus loader (`neuro_manipulation/utils.py::primary_emotions_concept_dataset`) supports multimodal datasets stored as JSON files (one per emotion) under `repe_eng_config.data_dir`.

### File Layout

```
data/stimulus/repeng_generated_images_with_text/
  anger.json
  happiness.json
  anger/...
  happiness/...
```

### JSON Entry Schemas

Each emotion JSON contains a list of entries. For multimodal (image-based) RepE, entries can be:

1) **Image-only** (string path)
```json
[
  "anger/0001.png",
  "anger/0002.png"
]
```

2) **Image + extra text** (object)
```json
[
  {"image": "anger/0001.png", "text": "Someone is yelling at you."},
  {"image_path": "anger/0002.png", "caption": "A threatening message on screen."}
]
```

Recognized keys:
- Image path: `image`, `image_path`, or `images[0]`
- Extra text: `text`, `prompt`, or `caption`

### Path Resolution (No `image_base_path` needed)

If the image path is **relative**, it is resolved relative to `repe_eng_config.data_dir` by default.

### Generated Prompt

For each multimodal entry, RepE uses:

```
Consider the emotion of the following scenario:
[IMAGE]
{extra_text (optional)}
```

## Running (Qwen2.5-VL image+text smoke config)

Example config: `config/qwen25_vl_mm_stimulus_image_text_smoke.yaml`

```
bash -c "source /usr/local/anaconda3/etc/profile.d/conda.sh && conda activate llm_fresh && CUDA_VISIBLE_DEVICES=0,1 python -m emotion_experiment_engine.emotion_experiment_series_runner --config config/qwen25_vl_mm_stimulus_image_text_smoke.yaml"
```

## vLLM Notes / Troubleshooting

- **No HF fallback**: the codepath is vLLM-only; vLLM init failures should raise.
- **vLLM v1 serialization**: RepE hook registration uses `collective_rpc` with callables; vLLM v1 requires `VLLM_ALLOW_INSECURE_SERIALIZATION=1` (the runner sets this automatically when `repe_eng_config` is present).
- **ViT FlashAttention head-dim constraint**: if your setup hits `headdim not being a multiple of 32`, force the vision encoder attention backend (e.g. `mm_encoder_attn_backend: TORCH_SDPA`) and avoid FlashAttention for ViT.

## Emotion Templates

### Standard Template
```
"when you see this image, your emotion is {emotion}"
```

### Supported Emotions
- **anger**: Aggressive, hostile, furious responses
- **happiness**: Joyful, positive, optimistic responses  
- **sadness**: Melancholic, depressed, sorrowful responses
- **disgust**: Repulsed, revolted, contemptuous responses
- **fear**: Anxious, scared, worried responses
- **surprise**: Shocked, amazed, astonished responses

## Pipeline Workflow

### 1. Input Processing
```python
# Automatic detection of multimodal vs text-only inputs
if pipeline._is_multimodal_input(inputs):
    # Process images + text together
    processed = pipeline._prepare_multimodal_inputs(inputs)
else:
    # Standard text-only processing
    processed = tokenizer(inputs)
```

### 2. Model Forward Pass
```python
# Both image and text features sent to model
outputs = model(
    input_ids=text_tokens,
    pixel_values=image_features,
    output_hidden_states=True
)

# Extract hidden states at emotion token position (default: -1)
emotion_representations = outputs.hidden_states[layer][:, -1, :]
```

### 3. Direction Finding
```python
# Use existing RepReaders on multimodal representations
if method == "pca":
    rep_reader = PCARepReader(n_components=1)
elif method == "cluster_mean":
    rep_reader = ClusterMeanRepReader()

# Extract emotion directions
directions = rep_reader.get_rep_directions(
    model, tokenizer, hidden_states, layers
)
```

### 4. Vector Application
```python
# Apply extracted vectors during text generation
hook.set_controller(
    direction=directions,
    intensity=1.5,
    position=-1  # Apply at emotion-relevant positions
)
```

## Advanced Features

### Layer Selection Strategy

For multimodal models, the system uses intelligent layer selection:

1. **Text/Language layers** (preferred): Where final text generation decisions are made
2. **Fusion layers** (backup): Where visual and textual information combine
3. **Cross-attention layers** (future): For cross-modal interaction analysis

### Quality Assessment

```python
# Validate extraction quality
extractor.validate_emotion_vectors(
    test_stimuli=validation_images,
    expected_emotions=ground_truth_labels
)

# Cross-emotion consistency check
consistency_score = extractor.check_consistency(
    emotion1_vectors=anger_vectors,
    emotion2_vectors=happiness_vectors
)
```

### Batch Optimization

```python
# Efficient batch processing for large image datasets
extractor.extract_from_dataset(
    image_dir="data/emotion_images/",
    batch_size=8,
    max_images_per_emotion=100
)
```

## Integration with Existing Systems

### Game Theory Experiments

```python
# In your existing experiment class
class EmotionGameExperiment:
    def setup_multimodal_emotion_control(self):
        # Load multimodal emotion vectors
        self.extractor = MultimodalEmotionExtractor(self.model_path)
        self.extractor.load_emotion_vectors(self.emotion_vector_path)
        
        # Setup vLLM hook with multimodal vectors
        self.emotion_hook = RepControlVLLMHook(...)
        
    def run_with_visual_emotion_priming(self, game_scenarios):
        # Apply visual-derived emotion vectors to text generation
        for scenario in game_scenarios:
            emotion = scenario.target_emotion
            self.emotion_hook.set_controller(
                direction=self.extractor.get_emotion_vector(emotion),
                intensity=scenario.emotion_intensity
            )
            response = self.model.generate(scenario.prompt, hooks=[self.emotion_hook])
```

### OpenAI Server Integration

The multimodal RepE system integrates with the existing OpenAI server:

```bash
# Start server with multimodal emotion control
python -m openai_server \
  --model /path/to/Qwen2.5-VL-7B-Instruct \
  --emotion anger \
  --emotion-vectors-path results/multimodal_emotions.pt
```

## Testing and Validation

### Unit Tests
```bash
python examples/test_multimodal_simple.py  # Basic functionality tests
python -m pytest neuro_manipulation/repe/tests/test_multimodal_rep_reading.py  # Full test suite
```

### Integration Tests
```bash
python examples/multimodal_emotion_extraction.py  # End-to-end examples
```

### Performance Validation
```python
# Test with actual Qwen-VL model
python -m neuro_manipulation.tests.test_multimodal_integration \
  --model-path /path/to/qwen-vl \
  --test-images data/validation_images/
```

## Best Practices

### 1. Image Quality
- Use high-resolution images (≥224x224) for better feature extraction
- Ensure clear emotional content in images
- Avoid ambiguous or mixed emotional expressions

### 2. Template Consistency
- Use consistent emotion inquiry templates across experiments
- Validate template effectiveness with human annotations
- Consider cultural variations in emotional expression

### 3. Vector Quality Control
- Cross-validate extracted vectors with held-out test sets  
- Check for consistency across different images of same emotion
- Monitor for bias in emotion representation

### 4. Computational Efficiency
- Batch process images when possible
- Use appropriate model precision (bfloat16) to save memory
- Cache extracted vectors to avoid recomputation

## Troubleshooting

### Common Issues

**Model Loading Errors**
```python
# Ensure multimodal model is available
if not os.path.exists(model_path):
    print(f"Model not found: {model_path}")
    # Download model or update path

# Check trust_remote_code setting
model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True  # Required for Qwen-VL
)
```

**Memory Issues**
```python
# Reduce batch size
pipeline_config['batch_size'] = 2

# Use smaller precision
model_config['torch_dtype'] = 'float16'

# Enable gradient checkpointing
model_config['gradient_checkpointing'] = True
```

**Poor Vector Quality**
```python
# Try different extraction layers
pipeline_config['hidden_layers'] = [-1, -2, -3, -4]

# Use cluster_mean instead of PCA
pipeline_config['direction_method'] = 'cluster_mean'

# Increase training data diversity
# Add more varied images per emotion
```

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Print model structure for layer debugging
from neuro_manipulation.model_layer_detector import ModelLayerDetector
ModelLayerDetector.print_model_structure(model, max_depth=3)

# Validate input processing
pipeline._debug_mode = True
result = pipeline.preprocess(multimodal_input)
```

## Future Extensions

### Planned Enhancements

1. **Cross-Modal Alignment**: Learn explicit mappings between visual and textual emotion representations
2. **Temporal Dynamics**: Support for video inputs and temporal emotion sequences  
3. **Fine-Grained Emotions**: Beyond basic 6-emotion model to more nuanced emotional states
4. **Cultural Adaptation**: Region-specific emotion templates and validation
5. **Active Learning**: Automatically identify optimal training stimuli

### Research Applications

- **Multimodal Bias Studies**: Analyze how visual priming affects linguistic decision-making
- **Cross-Cultural Psychology**: Study emotion expression differences across cultures  
- **Therapeutic Applications**: Emotion regulation through controlled visual-linguistic exposure
- **Creative AI**: Emotionally-aware image captioning and story generation

## API Reference

### MultimodalEmotionExtractor Class

```python
class MultimodalEmotionExtractor:
    def __init__(self, model_path: str, config_path: str = None)
    def setup_pipeline(self) -> bool
    def create_emotion_stimulus(self, image_path: str, emotion: str) -> dict
    def extract_emotion_vectors(self, stimuli: list, emotion_labels: list = None) -> dict
    def save_emotion_vectors(self, output_path: str) -> bool
    def load_emotion_vectors(self, input_path: str) -> bool
```

### Enhanced RepReadingPipeline Methods

```python
class RepReadingPipeline:
    def _is_multimodal_input(self, inputs: Any) -> bool
    def _prepare_multimodal_inputs(self, inputs: Union[Dict, List], **kwargs) -> Dict[str, Any]  
    def preprocess(self, inputs: Union[str, List, Dict], **kwargs) -> Dict[str, Any]
```

### ModelLayerDetector Extensions

```python 
class ModelLayerDetector:
    @staticmethod
    def is_multimodal_model(model) -> bool
    
    @staticmethod
    def get_multimodal_layer_info(model) -> dict
    
    @staticmethod  
    def get_model_layers(model) -> nn.ModuleList  # Enhanced for multimodal
```

For more examples and detailed usage, see `examples/multimodal_emotion_extraction.py` and the configuration file `config/multimodal_rep_reading_config.yaml`.
