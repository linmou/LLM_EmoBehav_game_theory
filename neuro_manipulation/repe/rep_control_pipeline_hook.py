from transformers.pipelines import TextGenerationPipeline
from neuro_manipulation.model_layer_detector import ModelLayerDetector
import torch
import numpy as np
import logging

class RepControlHook:
    """Hook class to store controller settings and modify outputs"""
    def __init__(self):
        self.output = None
        self.controller = None
        self.mask = None
        self.token_pos = None
        self.normalize = False
        self.operator = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
    
    def forward_hook_fn(self, module, inputs, output):
        """Forward hook function to modify layer outputs"""
        self.logger.info(f"[Hook {id(self)}] Forward hook called for module {module.__class__.__name__}")
        
        # Store original output for later retrieval
        if isinstance(output, tuple):
            # Store a deep copy to avoid modifications affecting the stored output
            self.output = output[0].detach().clone()
            modified = output[0].clone()  # Clone to avoid modifying original tensor
        else:
            self.output = output.detach().clone()
            modified = output.clone()
        
        # Only modify if controller is set
        if self.controller is not None:
            self.logger.info(f"[Hook {id(self)}] Controller is set. Applying modification.")
            
            # Get norm before modification (for normalization option)
            norm_pre = torch.norm(modified, dim=-1, keepdim=True)
            self.logger.debug(f"[Hook {id(self)}] Norm before modification: {norm_pre.mean().item()}")
            
            # Handle masking
            if self.mask is not None:
                mask = self.mask
            # Handle position_ids-based masking (for padding tokens)
            elif len(inputs) > 0 and isinstance(inputs[-1], dict) and "position_ids" in inputs[-1]:
                pos = inputs[-1]["position_ids"]
                zero_indices = (pos == 0).cumsum(1).argmax(1, keepdim=True)
                col_indices = torch.arange(pos.size(1), device=pos.device).unsqueeze(0)
                target_shape = modified.shape
                mask = (col_indices >= zero_indices).float().reshape(target_shape[0], target_shape[1], 1)
                mask = mask.to(modified.dtype)
            else:
                mask = 1.0
            
            # Reshape controller to match modified tensor shape
            if len(self.controller.shape) == 1:
                # Expand from vector to match batch x seq_len x hidden_dim
                controller = self.controller.reshape(1, 1, -1)
                controller = controller.expand(modified.shape[0], modified.shape[1], -1)
            elif len(self.controller.shape) == 2:
                # Expand from [batch, hidden_dim] to [batch, seq_len, hidden_dim]
                controller = self.controller.unsqueeze(1)
                controller = controller.expand(modified.shape[0], modified.shape[1], -1)
            elif len(self.controller.shape) == 3:
                # Already in [batch, seq_len, hidden_dim] format, but may need to expand
                if self.controller.shape[1] == 1 and modified.shape[1] > 1:
                    controller = self.controller.expand(modified.shape[0], modified.shape[1], -1)
                else:
                    controller = self.controller
            else:
                raise ValueError(f"Unsupported controller shape {self.controller.shape}")
            
            # Ensure controller is on same device
            controller = controller.to(modified.device)
            if isinstance(mask, torch.Tensor):
                mask = mask.to(modified.device)
            
            # Apply controller based on token position
            if self.token_pos == -1:
                # Special case for last token: apply to the last token of each sequence
                self.logger.info(f"[Hook {id(self)}] Applying controller to last token only.")
                modified_val_before = modified[0, -1].mean().item()
                for i in range(modified.shape[0]):  # Loop over batch dimension
                    modified[i, -1] = self.operator(modified[i, -1], controller[i, 0])
                modified_val_after = modified[0, -1].mean().item()
                self.logger.info(f"[Hook {id(self)}] Last token mean before: {modified_val_before:.4f}, after: {modified_val_after:.4f}")
            elif isinstance(self.token_pos, int):
                self.logger.info(f"[Hook {id(self)}] Applying controller to token pos {self.token_pos}.")
                modified[:, self.token_pos] = self.operator(modified[:, self.token_pos], controller[:, 0] * mask)
            elif isinstance(self.token_pos, (list, tuple, np.ndarray)):
                self.logger.info(f"[Hook {id(self)}] Applying controller to token pos list: {self.token_pos}.")
                for pos in self.token_pos:
                    modified[:, pos] = self.operator(modified[:, pos], controller[:, 0] * mask)
            elif isinstance(self.token_pos, str):
                if self.token_pos == "end":
                    self.logger.info(f"[Hook {id(self)}] Applying controller to token pos 'end'.")
                    modified[:, -1] = self.operator(modified[:, -1], controller[:, 0] * mask)
                elif self.token_pos == "start":
                    self.logger.info(f"[Hook {id(self)}] Applying controller to token pos 'start'.")
                    modified[:, 0] = self.operator(modified[:, 0], controller[:, 0] * mask)
                else:
                    assert False, f"Unknown token position {self.token_pos}."
            else:
                # Apply to all tokens
                self.logger.info(f"[Hook {id(self)}] Applying controller to all tokens.")
                modified_val_before = modified.mean().item()
                modified = self.operator(modified, controller * mask)
                modified_val_after = modified.mean().item()
                self.logger.info(f"[Hook {id(self)}] All tokens mean before: {modified_val_before:.4f}, after: {modified_val_after:.4f}")
            
            # Normalize if requested
            if self.normalize:
                self.logger.info(f"[Hook {id(self)}] Normalizing output.")
                norm_post = torch.norm(modified, dim=-1, keepdim=True)
                modified = modified / norm_post * norm_pre
        else:
            self.logger.info(f"[Hook {id(self)}] Controller is None. No modification applied.")
        
        # Return modified output in same format as original
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        else:
            return modified
    
    def set_controller(self, activations, token_pos=None, masks=None, normalize=False, operator='linear_comb'):
        """Set controller parameters"""
        self.logger.info(f"[Hook {id(self)}] Setting controller: token_pos={token_pos}, normalize={normalize}, operator={operator}")
        # Convert activations to tensor if it's not already
        if not isinstance(activations, torch.Tensor):
            activations = torch.tensor(activations)
        
        self.controller = activations
        self.token_pos = token_pos
        self.mask = masks
        self.normalize = normalize
        
        # Set the operator function
        if operator == 'linear_comb':
            def op(current, controller):
                return current + controller
        elif operator == 'piecewise_linear':
            def op(current, controller):
                sign = torch.sign((current * controller).sum(-1, keepdim=True))
                return current + controller * sign
        elif operator == 'projection':
            def op(current, controller):
                raise NotImplementedError
        else:
            raise NotImplementedError(f"Operator {operator} not implemented.")
        self.operator = op
    
    def reset(self):
        """Reset all controller parameters"""
        self.logger.info(f"[Hook {id(self)}] Resetting hook state.")
        self.output = None
        self.controller = None
        self.mask = None
        self.token_pos = None
        self.operator = None

class RepControlPipelineHook(TextGenerationPipeline):
    """
    A pipeline for representation engineering using forward hooks.
    
    This implementation uses PyTorch forward hooks to modify the model's
    internal representations during generation. The implementation is designed
    to be functionally equivalent to RepControlPipelineWrappedBlock.
    
    Key improvements:
    - Better handling of different shapes for controller vs. layer outputs
    - Careful handling of token positions, especially for -1 indices
    - Clear separation of custom parameters from those passed to generate()
    - Consistent application of activations across both implementations
    """
    def __init__(self, 
                 model, 
                 tokenizer, 
                 layers, 
                 block_name="decoder_block", 
                 control_method="reading_vec",
                 raw_llm=None,
                 **kwargs):
        
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
        
        # Store parameters
        self.layers = layers if isinstance(layers, (list, tuple, np.ndarray)) else [layers]
        self.block_name = block_name
        self.control_method = control_method
        
        # Get model layers using ModelLayerDetector
        self.model_layers = ModelLayerDetector.get_model_layers(raw_llm if raw_llm is not None else model)
        
        # Dictionary to store hooks and handles
        self.hooks: dict[int, tuple[RepControlHook, torch.utils.hooks.RemovableHandle]] = {}  # Format: {layer_id: (hook_object, hook_handle)}
        
        # Make sure tokenizer has a pad token
        if self.tokenizer is not None and self.tokenizer.pad_token is None:
            if hasattr(self.model, "config") and hasattr(self.model.config, "eos_token_id"):
                self.tokenizer.pad_token_id = self.model.config.eos_token_id
                print(f"Setting pad_token_id to eos_token_id: {self.tokenizer.pad_token_id}")
            elif self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"Setting pad_token to eos_token: {self.tokenizer.pad_token}")
    
    def register_hooks(self):
        """Register hooks for specified layers"""
        for layer_id in self.layers:
            # Skip if hook already registered
            if layer_id in self.hooks:
                continue
                
            # Get target module
            if self.block_name == "decoder_block":
                target_module = self.model_layers[layer_id]
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
        """Set controller for all registered hooks"""
        # Register hooks if not already done
        self.register_hooks()
        
        for layer_id in self.layers:
            if layer_id in self.hooks:
                hook, _ = self.hooks[layer_id]
                # Reset hook first to clear any previous controller settings
                hook.reset()
                # Set controller parameters for this hook
                if isinstance(activations, dict):
                    hook.set_controller(activations[layer_id], token_pos, masks, normalize, operator)
                else:
                    hook.set_controller(activations, token_pos, masks, normalize, operator)
    
    def reset(self):
        """Reset all hooks"""
        # Reset the RepControlHook objects
        for layer_id, (hook, _) in self.hooks.items():
            hook.reset()
    
    def get_activations(self):
        """Get activations from all hooks"""
        activations = {}
        for layer_id, (hook, _) in self.hooks.items():
            if hook.output is not None:
                activations[layer_id] = hook.output
        return activations
    
    def __call__(self, text_inputs, reset_hooks=True, activations=None, token_pos=None, masks=None, normalize=False, operator='linear_comb', **kwargs):
        """Modified __call__ to handle activations"""
        # Reset hooks first to clear any previous state
        self.reset()
        
        # Always register hooks to capture activations
        self.register_hooks()
        
        # Set controller if activations provided
        if activations is not None:
            self.set_controller(activations, token_pos=token_pos, masks=masks, normalize=normalize, operator=operator)
        
        # Call parent class method without our custom parameters
        outputs = super().__call__(text_inputs, **kwargs)
        
        # Reset hooks if requested
        if reset_hooks:
            self.reset()
        
        return outputs