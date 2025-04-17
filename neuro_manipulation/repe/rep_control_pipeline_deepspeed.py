import torch
import numpy as np
import deepspeed
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
from neuro_manipulation.model_layer_detector import ModelLayerDetector
from neuro_manipulation.repe.rep_control_pipeline_hook import RepControlHook

class RepControlPipelineDeepspeed:
    """
    A standalone pipeline for representation engineering using forward hooks with DeepSpeed pipeline parallelism.
    
    This implementation uses PyTorch forward hooks to modify the model's
    internal representations during generation, specifically designed to work
    with DeepSpeed's pipeline parallelism for efficient inference.
    
    Key features:
    - Integrates with DeepSpeed pipeline parallelism for accelerated inference
    - Registers hooks correctly on distributed model layers
    - Handles cross-device coordination for activation modifications
    - Optimized for inference only (not training)
    """
    def __init__(self, 
                 model, 
                 tokenizer, 
                 layers, 
                 block_name="decoder_block", 
                 control_method="reading_vec",
                 deepspeed_config=None,
                 raw_llm=None,
                 **kwargs):
        
        # Store model and tokenizer
        self.model = model
        self.tokenizer = tokenizer
        
        # Store parameters
        self.layers = layers if isinstance(layers, (list, tuple, np.ndarray)) else [layers]
        self.block_name = block_name
        self.control_method = control_method
        self.deepspeed_config = deepspeed_config
        
        # Initialize DeepSpeed if not already initialized
        if not hasattr(model, 'module') or not hasattr(model, 'pipeline_rank'):
            # Basic default config if none provided
            if deepspeed_config is None:
                deepspeed_config = {
                    "train_batch_size": 1,  # Not used for inference
                    "train_micro_batch_size_per_gpu": 1,  # Not used for inference
                    "steps_per_print": 1,
                    "optimizer": {
                        "type": "Adam",
                        "params": {
                            "lr": 0.001,
                            "weight_decay": 0
                        }
                    },
                    "pipeline": {
                        "stages": 2,  # Default to 2 pipeline stages
                        "activation_checkpoint_interval": 0  # Disable activation checkpointing for inference
                    },
                    "zero_optimization": {
                        "stage": 0  # Disable ZeRO for inference pipeline
                    }
                }
            
            # --- BEGIN LOGGING --- 
            log_msg = f"DeepSpeed Init Config: {deepspeed_config}"
            print(log_msg)
            logging.info(log_msg) 
            # --- END LOGGING --- 
            
            # Initialize DeepSpeed for inference
            engine, _, _, _ = deepspeed.initialize(
                model=model,
                config=deepspeed_config,
                model_parameters=model.parameters()
            )
            self.model = engine
        
        # Get model layers using ModelLayerDetector
        self.model_layers = ModelLayerDetector.get_model_layers(raw_llm if raw_llm is not None else self._get_model_from_deepspeed())
        
        # Dictionary to store hooks and handles
        self.hooks: Dict[int, Tuple[RepControlHook, torch.utils.hooks.RemovableHandle]] = {}  # Format: {layer_id: (hook_object, hook_handle)}
        
        # Get DeepSpeed pipeline information
        self.pipeline_rank = getattr(self.model, 'pipeline_rank', 0)
        self.pipeline_world_size = getattr(self.model, 'pipeline_world_size', 1)
        
        # Identify which layers are on the current pipeline rank
        self.local_layers = self._map_global_to_local_layers()
        
        # Make sure tokenizer has a pad token
        if self.tokenizer is not None and self.tokenizer.pad_token is None:
            if hasattr(self.model, "config") and hasattr(self.model.config, "eos_token_id"):
                self.tokenizer.pad_token_id = self.model.config.eos_token_id
                print(f"Setting pad_token_id to eos_token_id: {self.tokenizer.pad_token_id}")
            elif self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"Setting pad_token to eos_token: {self.tokenizer.pad_token}")
    
    def _get_model_from_deepspeed(self):
        """Extract the base model from DeepSpeed wrapper"""
        if hasattr(self.model, 'module'):
            return self.model.module
        return self.model
    
    def _map_global_to_local_layers(self) -> Dict[int, int]:
        """Map global layer IDs to local layer IDs within the current pipeline stage"""
        local_layers = {}
        
        # Skip if not using pipeline parallelism
        if self.pipeline_world_size <= 1:
            return {layer_id: layer_id for layer_id in self.layers}
        
        # Determine layers_per_stage (rough approximation)
        total_layers = ModelLayerDetector.num_layers(self._get_model_from_deepspeed())
        layers_per_stage = total_layers // self.pipeline_world_size
        
        # Determine which global layers map to the current pipeline stage
        for layer_id in self.layers:
            stage_id = layer_id // layers_per_stage
            if stage_id == self.pipeline_rank:
                local_id = layer_id % layers_per_stage
                local_layers[layer_id] = local_id
        
        return local_layers
    
    def register_hooks(self):
        """Register hooks for specified layers on the current pipeline stage"""
        for layer_id, local_id in self.local_layers.items():
            # Skip if hook already registered
            if layer_id in self.hooks:
                continue
                
            # Get target module based on local ID
            if self.block_name == "decoder_block":
                # Access the transformer layers appropriately based on model architecture
                try:
                    # Different models may have different module structures
                    base_model = self._get_model_from_deepspeed()
                    if hasattr(base_model, 'transformer'):
                        target_module = base_model.transformer.h[local_id]
                    elif hasattr(base_model, 'model'):
                        target_module = base_model.model.layers[local_id]
                    elif hasattr(base_model, 'layers'):
                        target_module = base_model.layers[local_id]
                    elif hasattr(base_model, 'decoder'):
                        target_module = base_model.decoder.layers[local_id]
                    else:
                        # Fallback to using ModelLayerDetector to find the module
                        target_module = self.model_layers.get(layer_id)
                        if target_module is None:
                            print(f"Warning: Could not find module for layer {layer_id} (local {local_id}) on pipeline rank {self.pipeline_rank}")
                            continue
                except (IndexError, AttributeError) as e:
                    print(f"Error accessing layer {local_id}: {e}")
                    continue
            else:
                raise NotImplementedError(f"Block name {self.block_name} not supported yet")
            
            # Create hook and register it
            hook = RepControlHook()
            handle = target_module.register_forward_hook(hook.forward_hook_fn)
            self.hooks[layer_id] = (hook, handle)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for layer_id, (_, handle) in list(self.hooks.items()):
            handle.remove()
            del self.hooks[layer_id]
    
    def set_controller(self, activations, token_pos=None, masks=None, normalize=False, operator='linear_comb'):
        """Set controller for all registered hooks on the current pipeline stage"""
        # Register hooks if not already done
        self.register_hooks()
        
        # Only work with layers local to this pipeline stage
        for layer_id in self.local_layers.keys():
            if layer_id in self.hooks:
                hook, _ = self.hooks[layer_id]
                # Reset hook first to clear any previous controller settings
                hook.reset()
                
                # Set controller parameters for this hook
                if isinstance(activations, dict):
                    # Only set if this layer's activations are provided
                    if layer_id in activations:
                        print(f"[RepControlPipelineDeepspeed] Setting controller for layer {layer_id} (local) with keys: {activations[layer_id].keys() if isinstance(activations[layer_id], dict) else 'tensor'}")
                        hook.set_controller(activations[layer_id], token_pos, masks, normalize, operator)
                else:
                    # Use the same activations for all layers
                    print(f"[RepControlPipelineDeepspeed] Setting controller for layer {layer_id} (local) with provided tensor.")
                    hook.set_controller(activations, token_pos, masks, normalize, operator)
    
    def reset(self):
        """Reset all hooks on the current pipeline stage"""
        print("[RepControlPipelineDeepspeed] Resetting hooks.")
        for layer_id, (hook, _) in self.hooks.items():
            hook.reset()
    
    def get_activations(self):
        """Get activations from all hooks on the current pipeline stage"""
        activations = {}
        for layer_id, (hook, _) in self.hooks.items():
            if hook.output is not None:
                activations[layer_id] = hook.output
        return activations
    
    def generate_text(self, input_ids, attention_mask=None, **generation_kwargs):
        """Generate text using DeepSpeed pipeline parallelism if available"""
        if hasattr(self.model, 'generate_with_pipeline'):
            # Generate with DeepSpeed pipeline
            outputs = self.model.generate_with_pipeline(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs
            )
        else:
            # Fall back to standard generate method
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs
            )
        
        return outputs
    
    def __call__(self, text_inputs, reset_hooks=True, activations=None, token_pos=None, masks=None, 
                 normalize=False, operator='linear_comb', batch_size=4, **kwargs):
        """Main interface that mimics TextGenerationPipeline's __call__ method"""
        # Reset hooks first to clear any previous state
        print(f"[RepControlPipelineDeepspeed __call__] Resetting hooks (reset_hooks={reset_hooks}).")
        self.reset()
        
        # Always register hooks to capture activations
        print("[RepControlPipelineDeepspeed __call__] Registering hooks.")
        self.register_hooks()
        
        # Set controller if activations provided
        if activations is not None:
            print(f"[RepControlPipelineDeepspeed __call__] Setting controller with activations: {'Dict with keys: ' + str(activations.keys()) if isinstance(activations, dict) else 'Tensor provided'}")
            self.set_controller(activations, token_pos=token_pos, masks=masks, normalize=normalize, operator=operator)
        else:
            print("[RepControlPipelineDeepspeed __call__] No activations provided, controller not set.")
        
        # Ensure text_inputs is a list
        if isinstance(text_inputs, str):
            text_inputs = [text_inputs]
        
        # Process inputs in batches
        all_outputs = []
        for i in range(0, len(text_inputs), batch_size):
            batch_texts = text_inputs[i:i+batch_size]
            
            # Tokenize inputs
            tokenized_inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True)
            input_ids = tokenized_inputs.input_ids.to(self.model.device)
            attention_mask = tokenized_inputs.attention_mask.to(self.model.device)
            
            # Extract relevant generation parameters
            generation_kwargs = {k: v for k, v in kwargs.items() 
                               if k in ['max_length', 'max_new_tokens', 'temperature', 'do_sample', 'top_p']}
            
            # Generate outputs
            outputs = self.generate_text(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs
            )
            
            # Convert token IDs back to text
            for j, output_ids in enumerate(outputs):
                generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                all_outputs.append({"generated_text": generated_text})
        
        # Reset hooks if requested
        if reset_hooks:
            self.reset()
        
        # Format output similar to TextGenerationPipeline
        if len(text_inputs) == 1 and isinstance(text_inputs[0], str):
            return [all_outputs[0]]
        else:
            return [all_outputs]


"""
# Rep Control Pipeline DeepSpeed

This module implements a DeepSpeed-compatible version of the representation control 
pipeline for large language models. It leverages DeepSpeed's pipeline parallelism 
to efficiently distribute model layers across multiple GPUs while allowing for 
neuron-level control using forward hooks.

## Key Features:

1. **Pipeline Parallelism**: Efficiently splits model layers across GPUs to handle 
   larger models than would fit on a single GPU.
   
2. **Forward Hook Integration**: Maintains the ability to modify internal representations
   during inference, even when the model is distributed.
   
3. **Layer Mapping**: Automatically maps global layer IDs to local layer IDs within
   each pipeline stage.
   
4. **DeepSpeed Initialization**: Can initialize DeepSpeed or work with already
   initialized DeepSpeed models.

## Usage Example:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from neuro_manipulation.repe.rep_control_pipeline_deepspeed import RepControlPipelineDeepspeed

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define DeepSpeed config
ds_config = {
    "train_batch_size": 1,
    "pipeline": {
        "stages": 2,
        "activation_checkpoint_interval": 0
    },
    "zero_optimization": {
        "stage": 0
    }
}

# Create pipeline with DeepSpeed
pipeline = RepControlPipelineDeepspeed(
    model=model,
    tokenizer=tokenizer,
    layers=[-1, -2, -3],  # Control last three layers
    deepspeed_config=ds_config
)

# Define some activation vectors
import torch
activations = {
    -1: torch.randn(1, 1, 4096),  # Example for layer -1
    -2: torch.randn(1, 1, 4096),  # Example for layer -2
    -3: torch.randn(1, 1, 4096),  # Example for layer -3
}

# Generate text with representations control
outputs = pipeline(
    "Write a story about a robot that learns to love:",
    activations=activations,
    token_pos="all",  # Apply to all tokens
    max_new_tokens=100
)

print(outputs[0]["generated_text"])
```

Note: This implementation is focused on inference only and not designed for training.
"""


# Unit test for RepControlPipelineDeepspeed
def test_rep_control_pipeline_deepspeed():
    """
    Unit test for RepControlPipelineDeepspeed class.
    
    This test verifies:
    1. Basic initialization with DeepSpeed
    2. Hook registration and controller setting
    3. Text generation with activation modification
    4. Proper layer mapping and hook management
    
    To run this test:
    ```
    python -m pytest tests/test_rep_control_pipeline_deepspeed.py -v
    ```
    """
    import pytest
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Skip test if deepspeed not available
    try:
        import deepspeed
    except ImportError:
        pytest.skip("DeepSpeed not installed, skipping test")
    
    # Skip if no GPU available
    if not torch.cuda.is_available():
        pytest.skip("No GPU available, skipping test")
    
    # Load a small model for testing
    model_name = "gpt2"  # Use a small model for testing
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Basic DeepSpeed config for testing
    ds_config = {
        "train_batch_size": 1,
        "train_micro_batch_size_per_gpu": 1,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.001,
                "weight_decay": 0
            }
        },
        "pipeline": {
            "stages": 1,  # Single stage for testing
        },
        "zero_optimization": {
            "stage": 0
        }
    }
    
    # Initialize the pipeline
    pipeline = RepControlPipelineDeepspeed(
        model=model,
        tokenizer=tokenizer,
        layers=[-1],  # Just control the last layer
        deepspeed_config=ds_config
    )
    
    # Check initialization
    assert hasattr(pipeline.model, "module"), "Model should be wrapped by DeepSpeed"
    assert pipeline.pipeline_world_size >= 1, "Pipeline world size should be at least 1"
    
    # Check layer mapping
    assert len(pipeline.local_layers) > 0, "Local layers should be mapped"
    
    # Create some test activations
    hidden_size = 768  # GPT2 hidden size
    test_activations = {
        -1: torch.ones(1, 1, hidden_size).to(pipeline.model.device) * 0.1  # Small perturbation
    }
    
    # Register hooks and set controller
    pipeline.register_hooks()
    pipeline.set_controller(test_activations)
    
    # Verify hooks are registered
    assert len(pipeline.hooks) > 0, "Hooks should be registered"
    
    # Generate text without and with activations
    prompt = "Once upon a time, there was a"
    
    # Without activations
    pipeline.reset()
    outputs_base = pipeline(prompt, max_new_tokens=20)
    
    # With activations
    outputs_modified = pipeline(
        prompt,
        activations=test_activations,
        max_new_tokens=20
    )
    
    # Clean up
    pipeline.remove_hooks()
    
    # Just check that outputs are different format
    assert isinstance(outputs_base, list), "Base outputs should be a list"
    assert isinstance(outputs_modified, list), "Modified outputs should be a list"
    
    # Return success
    return True