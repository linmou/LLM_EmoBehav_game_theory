import torch
import unittest
from collections import deque
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelLayerDetector:
    @staticmethod
    def get_model_layers(model):
        '''
        Find model layers using breadth-first search tree traversal
        without hardcoding architecture patterns
        '''
        import torch.nn as nn
        
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
            
            # Return True if module has attention OR other transformer components (for RWKV)
            return has_attention or has_transformer_components
        
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
        
        while queue:
            path, module = queue.popleft()
            
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

class TestModelLayerDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set environment variables for models that require them
        os.environ["HF_ENDPOINT"] = "https://huggingface.co"
        
        # Skip tests if CUDA is not available
        cls.skip_gpu_tests = not torch.cuda.is_available()
        if cls.skip_gpu_tests:
            logger.warning("CUDA not available, skipping GPU tests")
    
    def test_layer_detection(self):
        """Test automatic layer detection with various models"""
        # List of models to test
        models_to_test = [
            # Small models for faster testing
            "gpt2",
            "facebook/opt-125m",
            "EleutherAI/pythia-70m",
        ]
        
        for model_name in models_to_test:
            with self.subTest(model=model_name):
                try:
                    logger.info(f"Testing layer detection on {model_name}")
                    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
                    
                    # Test layer detection
                    layers = ModelLayerDetector.get_model_layers(model)
                    
                    # Verify the result is a ModuleList
                    self.assertIsInstance(layers, torch.nn.ModuleList)
                    
                    # Verify it has more than one layer
                    self.assertGreater(len(layers), 0)
                    
                    logger.info(f"✓ Successfully detected layers for {model_name}")
                except Exception as e:
                    logger.error(f"Error with model {model_name}: {str(e)}")
                    raise
    
    @unittest.skipIf(True, "Skipping large model tests by default")
    def test_chatglm_model(self):
        """Test layer detection specifically with ChatGLM models"""
        if self.skip_gpu_tests:
            self.skipTest("CUDA not available, skipping ChatGLM test")
            
        try:
            logger.info("Testing layer detection on ChatGLM model")
            from transformers import AutoModel
            
            model = AutoModel.from_pretrained(
                "THUDM/glm-4-9b", 
                torch_dtype=torch.float16, 
                device_map="auto", 
                trust_remote_code=True
            )
            
            # Test layer detection
            layers = ModelLayerDetector.get_model_layers(model)
            
            # Verify the result is a ModuleList
            self.assertIsInstance(layers, torch.nn.ModuleList)
            
            # Verify it has more than one layer
            self.assertGreater(len(layers), 0)
            
            logger.info(f"✓ Successfully detected layers for ChatGLM model")
        except Exception as e:
            logger.error(f"Error with ChatGLM model: {str(e)}")
            raise
    
    @unittest.skipIf(True, "Skipping RWKV model tests by default")
    def test_rwkv_model(self):
        """Test layer detection specifically with RWKV models"""
        if self.skip_gpu_tests:
            self.skipTest("CUDA not available, skipping RWKV test")
            
        try:
            logger.info("Testing layer detection on RWKV model")
            
            # Loading RWKV requires special handling
            from transformers import AutoModelForCausalLM
            
            model = AutoModelForCausalLM.from_pretrained(
                "BlinkDL/rwkv-4-raven-1b5", 
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            # Test layer detection
            layers = ModelLayerDetector.get_model_layers(model)
            
            # Verify the result is a ModuleList
            self.assertIsInstance(layers, torch.nn.ModuleList)
            
            # Verify it has more than one layer
            self.assertGreater(len(layers), 0)
            
            logger.info(f"✓ Successfully detected layers for RWKV model")
        except Exception as e:
            logger.error(f"Error with RWKV model: {str(e)}")
            raise

if __name__ == "__main__":
    unittest.main() 