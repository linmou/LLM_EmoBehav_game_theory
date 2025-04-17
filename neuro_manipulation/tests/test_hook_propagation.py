import torch
import torch.nn as nn
import unittest

class SimpleNN(nn.Module):
    """A simple 3-layer neural network to demonstrate hook propagation"""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 4)
        self.layer2 = nn.Linear(4, 4)
        self.layer3 = nn.Linear(4, 1)
        self.activation = nn.ReLU()
        
        # Store intermediate values for testing
        self.layer1_output = None
        self.layer2_output = None
        self.layer3_output = None
    
    def forward(self, x):
        x = self.layer1(x)
        self.layer1_output = x.clone()  # Store for testing
        x = self.activation(x)
        
        x = self.layer2(x)
        self.layer2_output = x.clone()  # Store for testing
        x = self.activation(x)
        
        x = self.layer3(x)
        self.layer3_output = x.clone()  # Store for testing
        return x

class TestHookPropagation(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Create model
        self.model = SimpleNN()
        
        # Create test input
        self.test_input = torch.tensor([[1.0, 2.0]], requires_grad=True)
        
        # Store original outputs without hooks
        self.original_output = self.model(self.test_input)
        self.original_layer1_output = self.model.layer1_output.clone()
        self.original_layer2_output = self.model.layer2_output.clone()
        self.original_layer3_output = self.model.layer3_output.clone()
    
    def test_hook_propagation(self):
        """Test that hook modifications affect subsequent layer computations"""
        
        # Define a hook that doubles the output of layer1
        def double_output_hook(module, input, output):
            return output * 2
        
        # Register the hook on layer1
        hook_handle = self.model.layer1.register_forward_hook(double_output_hook)
        
        # Run forward pass with hook
        modified_output = self.model(self.test_input)
        
        # Store modified layer outputs
        modified_layer1_output = self.model.layer1_output
        modified_layer2_output = self.model.layer2_output
        modified_layer3_output = self.model.layer3_output
        
        # Remove hook
        hook_handle.remove()
        
        # Test 1: Layer 1 output should be doubled
        self.assertTrue(torch.allclose(modified_layer1_output, 
                                     self.original_layer1_output * 2, 
                                     rtol=1e-4))
        
        # Test 2: Layer 2 output should be different due to modified input
        self.assertFalse(torch.allclose(modified_layer2_output, 
                                       self.original_layer2_output, 
                                       rtol=1e-4))
        
        # Test 3: Layer 3 output should be different due to propagated changes
        self.assertFalse(torch.allclose(modified_layer3_output, 
                                       self.original_layer3_output, 
                                       rtol=1e-4))
        
        # Print values for inspection
        print("\nOriginal outputs:")
        print(f"Layer 1: {self.original_layer1_output}")
        print(f"Layer 2: {self.original_layer2_output}")
        print(f"Layer 3: {self.original_layer3_output}")
        
        print("\nModified outputs:")
        print(f"Layer 1: {modified_layer1_output}")
        print(f"Layer 2: {modified_layer2_output}")
        print(f"Layer 3: {modified_layer3_output}")
    
    def test_multiple_hooks(self):
        """Test interaction between multiple hooks on different layers"""
        
        # Define hooks that modify outputs in different ways
        def double_output_hook(module, input, output):
            return output * 2
        
        def add_constant_hook(module, input, output):
            return output + 1.0
        
        # Register hooks on different layers
        hook1_handle = self.model.layer1.register_forward_hook(double_output_hook)
        hook2_handle = self.model.layer2.register_forward_hook(add_constant_hook)
        
        # Run forward pass with both hooks
        modified_output = self.model(self.test_input)
        
        # Store modified layer outputs
        modified_layer1_output = self.model.layer1_output
        modified_layer2_output = self.model.layer2_output
        modified_layer3_output = self.model.layer3_output
        
        # Remove hooks
        hook1_handle.remove()
        hook2_handle.remove()
        
        # Test 1: Layer 1 output should be doubled
        self.assertTrue(torch.allclose(modified_layer1_output, 
                                     self.original_layer1_output * 2, 
                                     rtol=1e-4))
        
        # Test 2: Layer 2 output should reflect both modifications
        # First doubled by layer1's hook, then constant added
        self.assertFalse(torch.allclose(modified_layer2_output, 
                                       self.original_layer2_output, 
                                       rtol=1e-4))
        
        # Print values for inspection
        print("\nOriginal outputs (multiple hooks):")
        print(f"Layer 1: {self.original_layer1_output}")
        print(f"Layer 2: {self.original_layer2_output}")
        print(f"Layer 3: {self.original_layer3_output}")
        
        print("\nModified outputs (multiple hooks):")
        print(f"Layer 1: {modified_layer1_output}")
        print(f"Layer 2: {modified_layer2_output}")
        print(f"Layer 3: {modified_layer3_output}")

if __name__ == '__main__':
    unittest.main() 