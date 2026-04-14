# Multimodal VLM Pipeline - Adapter Pattern with Unified Interface Design

## Executive Summary

This document presents an **adapter-based design with a unified interface layer** that treats all VLMs through a common abstraction, using lightweight adapters to bridge model-specific differences. This approach emphasizes simplicity, type safety, and minimal runtime overhead while maintaining flexibility.

## 1. Core Philosophy

### Design Principles
- **Unified Interface First**: Single, consistent API for all models
- **Lightweight Adapters**: Minimal translation layers for model differences  
- **Compile-Time Type Safety**: Strong typing throughout the pipeline
- **Configuration as Code**: Model behaviors defined in code, not config
- **Composition over Inheritance**: Favor composition for flexibility

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│              Application Layer (Experiments)             │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│           Unified VLM Interface (Single API)             │
│         process() | extract() | configure()              │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│              Adapter Registry & Router                   │
│            (Static Registration, Fast Lookup)            │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────┬───────────┬───────────┐
        ▼           ▼           ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│  Qwen    │ │  LLaVA   │ │  BLIP    │ │  Gemma   │
│ Adapter  │ │ Adapter  │ │ Adapter  │ │ Adapter  │
└──────────┘ └──────────┘ └──────────┘ └──────────┘
        │           │           │           │
        ▼           ▼           ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│  Native  │ │  Native  │ │  Native  │ │  Native  │
│  Model   │ │  Model   │ │  Model   │ │  Model   │
└──────────┘ └──────────┘ └──────────┘ └──────────┘
```

## 3. Unified Interface Design

### 3.1 Core Types and Data Structures

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, TypeVar, Generic
from enum import Enum
import torch
from PIL import Image
import numpy as np

# Type definitions
T = TypeVar('T')
TensorType = Union[torch.Tensor, np.ndarray]

class ModalityType(Enum):
    """Supported modality types."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"

class ProcessingMode(Enum):
    """Processing modes for different use cases."""
    INFERENCE = "inference"
    FEATURE_EXTRACTION = "feature_extraction"
    EMOTION_EXTRACTION = "emotion_extraction"
    EMBEDDING = "embedding"

@dataclass
class MultimodalInput:
    """Unified input structure for all models."""
    text: Optional[str] = None
    images: Optional[List[Image.Image]] = None
    videos: Optional[List[Any]] = None
    audio: Optional[List[Any]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def modalities(self) -> List[ModalityType]:
        """Get list of present modalities."""
        present = []
        if self.text:
            present.append(ModalityType.TEXT)
        if self.images:
            present.append(ModalityType.IMAGE)
        if self.videos:
            present.append(ModalityType.VIDEO)
        if self.audio:
            present.append(ModalityType.AUDIO)
        return present

@dataclass
class ProcessingConfig:
    """Unified processing configuration."""
    mode: ProcessingMode = ProcessingMode.INFERENCE
    batch_size: int = 1
    max_length: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 0.95
    return_hidden_states: bool = False
    layers_to_extract: Optional[List[int]] = None
    device: str = "cuda"
    dtype: str = "float16"
    custom_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_params is None:
            self.custom_params = {}

@dataclass
class ProcessingOutput:
    """Unified output structure."""
    # Core outputs
    input_ids: Optional[TensorType] = None
    attention_mask: Optional[TensorType] = None
    pixel_values: Optional[TensorType] = None
    
    # Hidden states for emotion extraction
    hidden_states: Optional[List[TensorType]] = None
    
    # Generation outputs
    generated_text: Optional[str] = None
    generated_ids: Optional[TensorType] = None
    
    # Embeddings
    embeddings: Optional[TensorType] = None
    
    # Metadata
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
            
    def get_layer_hidden_states(self, layer: int) -> Optional[TensorType]:
        """Get hidden states for specific layer."""
        if self.hidden_states and abs(layer) <= len(self.hidden_states):
            return self.hidden_states[layer]
        return None
```

### 3.2 Unified VLM Interface

```python
from abc import ABC, abstractmethod

class UnifiedVLMInterface(ABC):
    """Unified interface that all VLM adapters must implement."""
    
    @abstractmethod
    def process(
        self,
        input_data: MultimodalInput,
        config: ProcessingConfig
    ) -> ProcessingOutput:
        """Process multimodal input with given configuration."""
        pass
    
    @abstractmethod
    def extract_features(
        self,
        input_data: MultimodalInput,
        layers: List[int]
    ) -> Dict[int, TensorType]:
        """Extract features from specified layers."""
        pass
    
    @abstractmethod
    def generate(
        self,
        input_data: MultimodalInput,
        max_length: int = 100,
        **kwargs
    ) -> str:
        """Generate text from multimodal input."""
        pass
    
    @abstractmethod
    def get_optimal_emotion_layers(self) -> List[int]:
        """Get optimal layers for emotion extraction."""
        pass
    
    @abstractmethod
    def format_emotion_prompt(
        self,
        emotion: str,
        template_style: str = "default"
    ) -> str:
        """Format emotion prompt for this model."""
        pass
    
    @abstractmethod
    def validate_input(self, input_data: MultimodalInput) -> bool:
        """Validate if input is compatible with model."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and capabilities."""
        pass
```

## 4. Adapter Implementation Pattern

### 4.1 Base Adapter Class

```python
class BaseVLMAdapter(UnifiedVLMInterface):
    """Base adapter with common functionality."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
        self._initialized = False
        
    def ensure_initialized(self):
        """Lazy initialization of model resources."""
        if not self._initialized:
            self._load_model()
            self._initialized = True
            
    @abstractmethod
    def _load_model(self):
        """Load model and processor - implemented by subclasses."""
        pass
    
    def validate_input(self, input_data: MultimodalInput) -> bool:
        """Common input validation."""
        # Check for required modalities
        required_modalities = self.get_required_modalities()
        present_modalities = input_data.modalities
        
        for required in required_modalities:
            if required not in present_modalities:
                return False
                
        # Check input sizes
        if input_data.images:
            max_size = self.get_max_image_size()
            for img in input_data.images:
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    return False
                    
        return True
    
    @abstractmethod
    def get_required_modalities(self) -> List[ModalityType]:
        """Get required modalities for this model."""
        pass
    
    @abstractmethod  
    def get_max_image_size(self) -> tuple:
        """Get maximum image size supported."""
        pass
```

### 4.2 Qwen Adapter Implementation

```python
class QwenVLAdapter(BaseVLMAdapter):
    """Adapter for Qwen Vision-Language models."""
    
    def _load_model(self):
        """Load Qwen model and processor."""
        from transformers import AutoModelForCausalLM, AutoProcessor
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            device_map=self.device
        )
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # Try to load Qwen utils
        try:
            from qwen_vl_utils import process_vision_info
            self.process_vision_info = process_vision_info
            self.has_qwen_utils = True
        except ImportError:
            self.has_qwen_utils = False
    
    def process(
        self,
        input_data: MultimodalInput,
        config: ProcessingConfig
    ) -> ProcessingOutput:
        """Process input through Qwen model."""
        self.ensure_initialized()
        
        # Convert to Qwen format
        messages = self._create_messages(input_data)
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process inputs
        if self.has_qwen_utils and input_data.images:
            image_inputs, video_inputs = self.process_vision_info(messages)
            model_inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt"
            )
        else:
            model_inputs = self.processor(
                text=[text],
                images=input_data.images,
                return_tensors="pt"
            )
        
        # Move to device
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
        
        # Process based on mode
        output = ProcessingOutput()
        
        if config.mode == ProcessingMode.FEATURE_EXTRACTION:
            with torch.no_grad():
                outputs = self.model(
                    **model_inputs,
                    output_hidden_states=True
                )
                
            output.hidden_states = outputs.hidden_states
            output.input_ids = model_inputs.get('input_ids')
            output.attention_mask = model_inputs.get('attention_mask')
            output.pixel_values = model_inputs.get('pixel_values')
            
        elif config.mode == ProcessingMode.INFERENCE:
            # Standard inference
            output.input_ids = model_inputs.get('input_ids')
            output.attention_mask = model_inputs.get('attention_mask')
            output.pixel_values = model_inputs.get('pixel_values')
            
        return output
    
    def _create_messages(self, input_data: MultimodalInput) -> List[Dict]:
        """Convert unified input to Qwen message format."""
        content = []
        
        if input_data.images:
            for image in input_data.images:
                content.append({"type": "image", "image": image})
                
        if input_data.text:
            content.append({"type": "text", "text": input_data.text})
            
        return [{"role": "user", "content": content}]
    
    def extract_features(
        self,
        input_data: MultimodalInput,
        layers: List[int]
    ) -> Dict[int, TensorType]:
        """Extract features from specified layers."""
        config = ProcessingConfig(
            mode=ProcessingMode.FEATURE_EXTRACTION,
            layers_to_extract=layers,
            return_hidden_states=True
        )
        
        output = self.process(input_data, config)
        
        features = {}
        for layer in layers:
            if output.hidden_states and abs(layer) <= len(output.hidden_states):
                features[layer] = output.hidden_states[layer]
                
        return features
    
    def generate(
        self,
        input_data: MultimodalInput,
        max_length: int = 100,
        **kwargs
    ) -> str:
        """Generate text from input."""
        self.ensure_initialized()
        
        config = ProcessingConfig(mode=ProcessingMode.INFERENCE)
        processed = self.process(input_data, config)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=processed.input_ids,
                attention_mask=processed.attention_mask,
                pixel_values=processed.pixel_values,
                max_length=max_length,
                **kwargs
            )
        
        # Decode
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        return generated_text
    
    def get_optimal_emotion_layers(self) -> List[int]:
        """Qwen optimal layers for emotion."""
        return [-5, -6, -7, -8, -9]
    
    def format_emotion_prompt(
        self,
        emotion: str,
        template_style: str = "default"
    ) -> str:
        """Format emotion prompt."""
        templates = {
            "default": "when you see this image, your emotion is {emotion}",
            "first_person": "Looking at this image, I feel {emotion}",
            "descriptive": "This image evokes a sense of {emotion}"
        }
        
        template = templates.get(template_style, templates["default"])
        return template.format(emotion=emotion)
    
    def get_required_modalities(self) -> List[ModalityType]:
        """Qwen requires text, optionally images."""
        return [ModalityType.TEXT]
    
    def get_max_image_size(self) -> tuple:
        """Qwen max image size."""
        return (1024, 1024)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Qwen model information."""
        return {
            "architecture": "Qwen-VL",
            "supports_batching": True,
            "max_context_length": 8192,
            "vision_backbone": "ViT",
            "optimal_dtype": "bfloat16"
        }
```

### 4.3 LLaVA Adapter Implementation

```python
class LLaVAAdapter(BaseVLMAdapter):
    """Adapter for LLaVA models."""
    
    def _load_model(self):
        """Load LLaVA model and processor."""
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path,
            device_map=self.device
        )
        
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        
        # LLaVA specific tokens
        self.image_token = "<image>"
        
    def process(
        self,
        input_data: MultimodalInput,
        config: ProcessingConfig
    ) -> ProcessingOutput:
        """Process through LLaVA model."""
        self.ensure_initialized()
        
        # Create LLaVA prompt
        prompt = self._create_prompt(input_data)
        
        # Process inputs
        if input_data.images:
            inputs = self.processor(
                text=prompt,
                images=input_data.images,
                return_tensors="pt"
            ).to(self.device)
        else:
            inputs = self.processor(
                text=prompt,
                return_tensors="pt"
            ).to(self.device)
        
        output = ProcessingOutput()
        
        if config.mode == ProcessingMode.FEATURE_EXTRACTION:
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True
                )
            
            output.hidden_states = outputs.hidden_states
            output.input_ids = inputs.get('input_ids')
            output.attention_mask = inputs.get('attention_mask')
            
        return output
    
    def _create_prompt(self, input_data: MultimodalInput) -> str:
        """Create LLaVA-style prompt."""
        num_images = len(input_data.images) if input_data.images else 0
        image_tokens = self.image_token * num_images
        
        prompt = f"USER: {image_tokens}"
        if input_data.text:
            prompt += f"\n{input_data.text}"
        prompt += "\nASSISTANT:"
        
        return prompt
    
    def get_optimal_emotion_layers(self) -> List[int]:
        """LLaVA optimal layers."""
        return [-1, -2, -3, -4]
    
    def format_emotion_prompt(
        self,
        emotion: str,
        template_style: str = "default"
    ) -> str:
        """LLaVA emotion template."""
        if template_style == "default":
            return f"Looking at this image, I feel {emotion}"
        else:
            return f"This image makes me feel {emotion}"
    
    def get_required_modalities(self) -> List[ModalityType]:
        """LLaVA requires text."""
        return [ModalityType.TEXT]
    
    def get_max_image_size(self) -> tuple:
        """LLaVA max image size."""
        return (336, 336)  # Default for LLaVA 1.5
    
    def get_model_info(self) -> Dict[str, Any]:
        """LLaVA model info."""
        return {
            "architecture": "LLaVA",
            "supports_batching": True,
            "max_context_length": 4096,
            "vision_backbone": "CLIP-ViT",
            "optimal_dtype": "float16"
        }
```

## 5. Adapter Registry and Router

### 5.1 Adapter Registry

```python
from typing import Type, Dict, Optional
import re

class AdapterRegistry:
    """Registry for VLM adapters with efficient routing."""
    
    def __init__(self):
        self._adapters: Dict[str, Type[BaseVLMAdapter]] = {}
        self._patterns: List[tuple] = []
        self._cache: Dict[str, BaseVLMAdapter] = {}
        
        # Register default adapters
        self._register_defaults()
        
    def _register_defaults(self):
        """Register default adapters."""
        self.register("qwen_vl", QwenVLAdapter, r"qwen.*vl")
        self.register("llava", LLaVAAdapter, r"llava")
        self.register("blip", BLIPAdapter, r"blip")
        self.register("gemma_vl", GemmaVLAdapter, r"gemma.*vl|gemma.*mm")
        
    def register(
        self,
        name: str,
        adapter_class: Type[BaseVLMAdapter],
        pattern: str = None
    ):
        """Register an adapter."""
        self._adapters[name] = adapter_class
        
        if pattern:
            self._patterns.append((re.compile(pattern, re.IGNORECASE), name))
            
    def get_adapter(
        self,
        model_path: str,
        use_cache: bool = True
    ) -> BaseVLMAdapter:
        """Get adapter for model path."""
        
        # Check cache
        if use_cache and model_path in self._cache:
            return self._cache[model_path]
        
        # Detect adapter type
        adapter_name = self._detect_adapter(model_path)
        
        if not adapter_name:
            raise ValueError(f"No adapter found for model: {model_path}")
        
        # Create adapter instance
        adapter_class = self._adapters[adapter_name]
        adapter = adapter_class(model_path)
        
        # Cache if requested
        if use_cache:
            self._cache[model_path] = adapter
            
        return adapter
    
    def _detect_adapter(self, model_path: str) -> Optional[str]:
        """Detect adapter type from model path."""
        
        # Try pattern matching
        for pattern, name in self._patterns:
            if pattern.search(model_path):
                return name
        
        # Try config file detection
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                
            arch = config.get("architectures", [""])[0].lower()
            
            for pattern, name in self._patterns:
                if pattern.search(arch):
                    return name
        
        return None
    
    def list_adapters(self) -> List[str]:
        """List registered adapters."""
        return list(self._adapters.keys())
    
    def clear_cache(self):
        """Clear adapter cache."""
        self._cache.clear()
```

### 5.2 Unified VLM Router

```python
class UnifiedVLMRouter:
    """High-level router for VLM operations."""
    
    def __init__(self):
        self.registry = AdapterRegistry()
        self.current_adapter: Optional[BaseVLMAdapter] = None
        self.current_model_path: Optional[str] = None
        
    def set_model(self, model_path: str):
        """Set current model."""
        if model_path != self.current_model_path:
            self.current_adapter = self.registry.get_adapter(model_path)
            self.current_model_path = model_path
            
    def process(
        self,
        input_data: Union[MultimodalInput, Dict, str],
        config: ProcessingConfig = None
    ) -> ProcessingOutput:
        """Process input with current model."""
        
        if not self.current_adapter:
            raise RuntimeError("No model set. Call set_model() first.")
        
        # Convert input to unified format
        if not isinstance(input_data, MultimodalInput):
            input_data = self._convert_input(input_data)
        
        # Use default config if not provided
        if config is None:
            config = ProcessingConfig()
        
        # Validate input
        if not self.current_adapter.validate_input(input_data):
            raise ValueError("Invalid input for current model")
        
        # Process
        return self.current_adapter.process(input_data, config)
    
    def _convert_input(
        self,
        input_data: Union[Dict, str]
    ) -> MultimodalInput:
        """Convert various input formats to unified format."""
        
        if isinstance(input_data, str):
            return MultimodalInput(text=input_data)
            
        if isinstance(input_data, dict):
            return MultimodalInput(
                text=input_data.get('text'),
                images=input_data.get('images'),
                videos=input_data.get('videos'),
                metadata=input_data.get('metadata', {})
            )
            
        raise ValueError(f"Unsupported input type: {type(input_data)}")
    
    def extract_emotion_vectors(
        self,
        stimuli: List[Dict[str, Any]],
        emotions: List[str]
    ) -> Dict[str, TensorType]:
        """Extract emotion vectors using current model."""
        
        if not self.current_adapter:
            raise RuntimeError("No model set")
        
        vectors = {}
        optimal_layers = self.current_adapter.get_optimal_emotion_layers()
        
        for emotion in emotions:
            # Format emotion prompt
            prompt = self.current_adapter.format_emotion_prompt(emotion)
            
            # Get stimuli for this emotion
            emotion_stimuli = [s for s in stimuli if s.get('emotion') == emotion]
            
            if not emotion_stimuli:
                continue
            
            # Process each stimulus
            hidden_states_list = []
            
            for stimulus in emotion_stimuli:
                input_data = MultimodalInput(
                    text=prompt,
                    images=stimulus.get('images')
                )
                
                # Extract features
                features = self.current_adapter.extract_features(
                    input_data,
                    optimal_layers
                )
                
                # Collect hidden states
                hidden_states_list.append(features)
            
            # Aggregate (simplified - would use PCA/clustering in practice)
            vectors[emotion] = hidden_states_list
        
        return vectors
```

## 6. Integration with Existing Pipeline

### 6.1 Modified RepReadingPipeline

```python
class RepReadingPipeline:
    """Pipeline using unified VLM interface."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.router = UnifiedVLMRouter()
        
        if model_path:
            self.router.set_model(model_path)
            
    def preprocess(
        self,
        inputs: Union[str, Dict, MultimodalInput],
        **kwargs
    ) -> ProcessingOutput:
        """Preprocess inputs using unified interface."""
        
        # Create processing config from kwargs
        config = ProcessingConfig(
            mode=ProcessingMode.FEATURE_EXTRACTION,
            return_hidden_states=True,
            **kwargs
        )
        
        # Process through router
        return self.router.process(inputs, config)
    
    def extract_emotion_vectors(
        self,
        stimuli: List[Dict],
        emotions: List[str] = None
    ) -> Dict[str, Any]:
        """Extract emotion vectors."""
        
        if emotions is None:
            emotions = ["anger", "happiness", "sadness", "fear", "disgust", "surprise"]
        
        return self.router.extract_emotion_vectors(stimuli, emotions)
    
    def forward(
        self,
        inputs: Union[str, Dict, MultimodalInput]
    ) -> ProcessingOutput:
        """Forward pass through model."""
        
        config = ProcessingConfig(mode=ProcessingMode.INFERENCE)
        return self.router.process(inputs, config)
```

### 6.2 Usage Examples

```python
# Example 1: Simple usage
pipeline = RepReadingPipeline("/models/Qwen2.5-VL-7B")

# Process text and image
input_data = MultimodalInput(
    text="What do you see in this image?",
    images=[Image.open("example.jpg")]
)

output = pipeline.forward(input_data)

# Example 2: Emotion extraction
emotion_stimuli = [
    {
        "emotion": "anger",
        "images": [Image.open("angry_face.jpg")]
    },
    {
        "emotion": "happiness", 
        "images": [Image.open("happy_scene.jpg")]
    }
]

emotion_vectors = pipeline.extract_emotion_vectors(emotion_stimuli)

# Example 3: Custom configuration
config = ProcessingConfig(
    mode=ProcessingMode.FEATURE_EXTRACTION,
    layers_to_extract=[-1, -2, -3],
    batch_size=8,
    device="cuda:1"
)

features = pipeline.preprocess(input_data, config=config)

# Example 4: Model switching
pipeline.router.set_model("/models/llava-1.5-7b")
llava_output = pipeline.forward(input_data)

pipeline.router.set_model("/models/blip2-opt-2.7b")
blip_output = pipeline.forward(input_data)
```

## 7. Type-Safe Configuration System

### 7.1 Model Configurations

```python
@dataclass
class QwenConfig:
    """Qwen-specific configuration."""
    use_qwen_utils: bool = True
    vision_feature_select: str = "mean"
    chat_template: bool = True
    optimal_layers: List[int] = None
    
    def __post_init__(self):
        if self.optimal_layers is None:
            self.optimal_layers = [-5, -6, -7, -8, -9]

@dataclass
class LLaVAConfig:
    """LLaVA-specific configuration."""
    use_image_token: bool = True
    vision_tower: str = "clip"
    merge_strategy: str = "spatial_unpad"
    optimal_layers: List[int] = None
    
    def __post_init__(self):
        if self.optimal_layers is None:
            self.optimal_layers = [-1, -2, -3, -4]

# Configuration factory
class ConfigFactory:
    """Factory for model-specific configurations."""
    
    @staticmethod
    def create_config(model_type: str) -> Any:
        """Create configuration for model type."""
        configs = {
            "qwen_vl": QwenConfig,
            "llava": LLaVAConfig,
            "blip": BLIPConfig,
            "gemma_vl": GemmaConfig
        }
        
        config_class = configs.get(model_type)
        if config_class:
            return config_class()
        
        return ProcessingConfig()  # Default config
```

### 7.2 Type-Safe Builder Pattern

```python
class VLMProcessingBuilder:
    """Builder for type-safe processing configuration."""
    
    def __init__(self):
        self._input = MultimodalInput()
        self._config = ProcessingConfig()
        
    def with_text(self, text: str) -> "VLMProcessingBuilder":
        """Add text input."""
        self._input.text = text
        return self
    
    def with_images(self, images: List[Image.Image]) -> "VLMProcessingBuilder":
        """Add image inputs."""
        self._input.images = images
        return self
    
    def with_mode(self, mode: ProcessingMode) -> "VLMProcessingBuilder":
        """Set processing mode."""
        self._config.mode = mode
        return self
    
    def with_device(self, device: str) -> "VLMProcessingBuilder":
        """Set device."""
        self._config.device = device
        return self
    
    def with_layers(self, layers: List[int]) -> "VLMProcessingBuilder":
        """Set layers to extract."""
        self._config.layers_to_extract = layers
        self._config.return_hidden_states = True
        return self
    
    def build(self) -> tuple[MultimodalInput, ProcessingConfig]:
        """Build input and config."""
        return self._input, self._config

# Usage
builder = VLMProcessingBuilder()
input_data, config = (builder
    .with_text("Describe this image")
    .with_images([image])
    .with_mode(ProcessingMode.FEATURE_EXTRACTION)
    .with_layers([-1, -2, -3])
    .build())

output = pipeline.router.process(input_data, config)
```

## 8. Performance Optimizations

### 8.1 Batch Processing Adapter

```python
class BatchProcessingMixin:
    """Mixin for efficient batch processing."""
    
    def process_batch(
        self,
        inputs: List[MultimodalInput],
        config: ProcessingConfig
    ) -> List[ProcessingOutput]:
        """Process batch of inputs efficiently."""
        
        # Group by modality pattern for optimal batching
        groups = self._group_by_modality(inputs)
        outputs = []
        
        for group in groups:
            # Process group together
            batch_output = self._process_group(group, config)
            outputs.extend(batch_output)
            
        return outputs
    
    def _group_by_modality(
        self,
        inputs: List[MultimodalInput]
    ) -> List[List[MultimodalInput]]:
        """Group inputs by modality pattern."""
        groups = {}
        
        for input_data in inputs:
            key = tuple(input_data.modalities)
            if key not in groups:
                groups[key] = []
            groups[key].append(input_data)
            
        return list(groups.values())
```

### 8.2 Caching Layer

```python
from functools import lru_cache
import hashlib

class CachingAdapter:
    """Adapter with caching capabilities."""
    
    def __init__(self, base_adapter: BaseVLMAdapter):
        self.adapter = base_adapter
        self._cache = {}
        
    @lru_cache(maxsize=128)
    def process_cached(
        self,
        input_hash: str,
        config_hash: str
    ) -> ProcessingOutput:
        """Process with caching."""
        # Actual processing done here
        # Input/config reconstructed from hash if needed
        pass
    
    def process(
        self,
        input_data: MultimodalInput,
        config: ProcessingConfig
    ) -> ProcessingOutput:
        """Process with automatic caching."""
        
        # Generate cache key
        input_hash = self._hash_input(input_data)
        config_hash = self._hash_config(config)
        cache_key = f"{input_hash}_{config_hash}"
        
        # Check cache
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Process
        output = self.adapter.process(input_data, config)
        
        # Cache result
        self._cache[cache_key] = output
        
        return output
    
    def _hash_input(self, input_data: MultimodalInput) -> str:
        """Generate hash for input."""
        hasher = hashlib.md5()
        
        if input_data.text:
            hasher.update(input_data.text.encode())
            
        if input_data.images:
            for img in input_data.images:
                hasher.update(str(img.size).encode())
                
        return hasher.hexdigest()
```

## 9. Testing Framework

### 9.1 Adapter Testing

```python
import pytest
from typing import Type

class AdapterTestSuite:
    """Test suite for VLM adapters."""
    
    @pytest.fixture
    def adapter_class(self) -> Type[BaseVLMAdapter]:
        """Override in subclasses."""
        raise NotImplementedError
    
    def test_interface_compliance(self, adapter_class):
        """Test adapter implements interface."""
        assert issubclass(adapter_class, UnifiedVLMInterface)
        
    def test_basic_processing(self, adapter_class):
        """Test basic processing works."""
        adapter = adapter_class("/fake/model/path")
        
        input_data = MultimodalInput(
            text="Test input",
            images=[Image.new('RGB', (224, 224))]
        )
        
        config = ProcessingConfig()
        
        # Should not raise
        adapter.validate_input(input_data)
        
    def test_feature_extraction(self, adapter_class):
        """Test feature extraction."""
        adapter = adapter_class("/fake/model/path")
        
        input_data = MultimodalInput(text="Test")
        layers = [-1, -2, -3]
        
        features = adapter.extract_features(input_data, layers)
        
        assert isinstance(features, dict)
        assert all(layer in features for layer in layers)
        
    def test_emotion_configuration(self, adapter_class):
        """Test emotion-specific configuration."""
        adapter = adapter_class("/fake/model/path")
        
        layers = adapter.get_optimal_emotion_layers()
        assert isinstance(layers, list)
        assert all(isinstance(l, int) for l in layers)
        
        prompt = adapter.format_emotion_prompt("anger")
        assert isinstance(prompt, str)
        assert "anger" in prompt.lower()

# Specific test class for Qwen
class TestQwenAdapter(AdapterTestSuite):
    @pytest.fixture
    def adapter_class(self):
        return QwenVLAdapter
```

### 9.2 Integration Testing

```python
def test_multi_model_consistency():
    """Test consistency across different models."""
    
    models = [
        "/models/qwen-vl",
        "/models/llava",
        "/models/blip"
    ]
    
    router = UnifiedVLMRouter()
    
    input_data = MultimodalInput(
        text="Test prompt",
        images=[Image.new('RGB', (224, 224), color='red')]
    )
    
    outputs = []
    
    for model_path in models:
        router.set_model(model_path)
        output = router.process(input_data)
        outputs.append(output)
    
    # Check all outputs have consistent structure
    assert all(isinstance(o, ProcessingOutput) for o in outputs)
    assert all(o.input_ids is not None for o in outputs)
```

## 10. Advantages and Trade-offs

### Advantages
1. **Type Safety**: Strong typing throughout the pipeline
2. **Simple Architecture**: Straightforward adapter pattern
3. **Performance**: Minimal runtime overhead
4. **Maintainability**: Clear separation of concerns
5. **Testability**: Easy to test each adapter independently
6. **Extensibility**: New models added by implementing adapter

### Trade-offs
1. **Less Dynamic**: Adapters must be known at compile time
2. **Code Duplication**: Some duplication across adapters
3. **Manual Registration**: Models must be explicitly registered
4. **Limited Runtime Flexibility**: Can't modify behavior dynamically
5. **Adapter Proliferation**: Need adapter for each model family

## 11. Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1)
- Implement unified interface and data structures
- Create base adapter class
- Build adapter registry and router

### Phase 2: Primary Adapters (Week 2)
- Implement Qwen adapter
- Implement LLaVA adapter
- Create testing framework

### Phase 3: Extended Support (Week 3)
- Add BLIP adapter
- Add Gemma adapter
- Implement caching and optimization

### Phase 4: Integration (Week 4)
- Update RepReadingPipeline
- Migrate existing code
- Performance testing

---

*Document Version: 1.0*  
*Architecture Type: Adapter Pattern with Unified Interface*  
*Last Updated: December 2024*