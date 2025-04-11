import torch
import torch.nn as nn
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelLayerDetector:
    """
    Utility class for automatically detecting transformer layers in any model architecture
    without hardcoding model-specific paths.
    """
    
    @staticmethod
    def get_model_layers(model):
        '''
        Find model layers using breadth-first search tree traversal
        without hardcoding architecture patterns
        
        Args:
            model: A PyTorch model (nn.Module)
            
        Returns:
            nn.ModuleList: The detected transformer layers
            
        Raises:
            ValueError: If no transformer layers could be detected
        '''
        # Characteristics that likely indicate transformer layers
        def is_transformer_layer(module):
            # Check if module has attention components
            has_attention = any(
                attr in dir(module) for attr in 
                ['attention', 'self_attn', 'self_attention', 'attn']
            )
            
            # Check if module has common transformer components
            has_transformer_components = any(
                attr in dir(module) for attr in 
                ['mlp', 'ffn', 'feed_forward', 'layernorm', 'layer_norm', 'norm']
            )
            
            # Return True if module has attention OR other transformer components
            # The OR condition helps with models that don't use explicit attention
            return has_attention # or has_transformer_components
        
        # Helper to check if a ModuleList is likely transformer layers
        def is_transformer_layers(module_list):
            if not isinstance(module_list, nn.ModuleList) or len(module_list) == 0:
                return False
            
            # Check first few layers to confirm they're transformer layers
            sample_size = min(3, len(module_list))
            return sum(is_transformer_layer(module_list[i]) for i in range(sample_size)) >= sample_size/2
        
        # BFS traversal to find transformer layers
        queue = deque([('', model)])
        transformer_layers_candidates = []
        
        # Track visited modules to avoid circular references
        visited = set()
        
        while queue:
            path, module = queue.popleft()
            
            # Skip if we've seen this module before (avoid cycles)
            module_id = id(module)
            if module_id in visited:
                continue
            visited.add(module_id)
            
            # If the module has a 'layers' attribute that's a ModuleList, check it
            if hasattr(module, 'layers') and isinstance(module.layers, nn.ModuleList) and len(module.layers) > 0:
                if is_transformer_layers(module.layers):
                    transformer_layers_candidates.append((f"{path}.layers", module.layers))
            
            # Queue named children for BFS traversal
            for name, child in module.named_children():
                child_path = f"{path}.{name}" if path else name
                queue.append((child_path, child))
            
            # Check if this module itself is a ModuleList that could be transformer layers
            if isinstance(module, nn.ModuleList) and len(module) > 0:
                if is_transformer_layers(module):
                    transformer_layers_candidates.append((path, module))
        
        # Process candidates, preferring ones named 'layers'
        # Sort by priority: 1) has 'layers' in name 2) path length (shorter is better)
        transformer_layers_candidates.sort(
            key=lambda x: (0 if 'layers' in x[0] else 1, len(x[0].split('.')))
        )
        
        if transformer_layers_candidates:
            logger.info(f"Found transformer layers at: {transformer_layers_candidates[0][0]}")
            return transformer_layers_candidates[0][1]
        
        # Last resort: find any ModuleList that has many similar modules
        for name, module in model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 0:
                # Check if modules have similar structure (same class)
                first_module_type = type(module[0])
                if all(isinstance(layer, first_module_type) for layer in module):
                    logger.info(f"Found possible layers at: {name}")
                    return module
        
        raise ValueError(f"Could not find transformer layers in model of type {type(model)}")

    @staticmethod
    def print_model_structure(model, max_depth=3):
        """
        Print the structure of a PyTorch model to help with debugging
        
        Args:
            model: A PyTorch model
            max_depth: Maximum depth to print
        """
        def _print_structure(module, prefix='', depth=0):
            if depth > max_depth:
                return
                
            for name, child in module.named_children():
                child_type = type(child).__name__
                
                # Print the current module
                print(f"{prefix}├── {name} ({child_type})")
                
                # For ModuleLists, print the first item type and count
                if isinstance(child, nn.ModuleList):
                    if len(child) > 0:
                        first_module_type = type(child[0]).__name__
                        print(f"{prefix}│   └── {len(child)} x {first_module_type}")
                    else:
                        print(f"{prefix}│   └── (empty)")
                else:
                    # Recursively print children
                    _print_structure(child, prefix + "│   ", depth + 1)
        
        model_type = type(model).__name__
        print(f"Model: {model_type}")
        _print_structure(model)
        
    @staticmethod
    def num_layers(model):
        return len(ModelLayerDetector.get_model_layers(model))