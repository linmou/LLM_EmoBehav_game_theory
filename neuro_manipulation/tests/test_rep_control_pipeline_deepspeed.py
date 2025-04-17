import pytest
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuro_manipulation.repe.rep_control_pipeline_deepspeed import RepControlPipelineDeepspeed
import os 

# Skip entire module if deepspeed not available
deepspeed_available = True
try:
    import deepspeed
except ImportError:
    deepspeed_available = False
    
pytestmark = pytest.mark.skipif(not deepspeed_available, reason="DeepSpeed not installed")

# Skip if no GPU available
if not torch.cuda.is_available():
    gpu_mark = pytest.mark.skipif(True, reason="No GPU available")
    pytestmark = [pytestmark, gpu_mark]


class TestRepControlPipelineDeepspeed:
    """Test suite for RepControlPipelineDeepspeed class"""
    
    @pytest.fixture(scope="class")
    def model_and_tokenizer(self):
        """Fixture to load model and tokenizer once for all tests"""
        model_name = "gpt2"  # Small model for testing
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    
    @pytest.fixture(scope="class")
    def ds_config(self):
        """Fixture for DeepSpeed config"""
        return {
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
            },
            "master_addr": "localhost",
            "master_port": "12355"
        }
    
    @pytest.fixture
    def pipeline(self, model_and_tokenizer, ds_config):
        """Fixture for pipeline instance with hooks registered"""
        model, tokenizer = model_and_tokenizer

        # Define desired distributed settings
        master_addr = ds_config.get("master_addr", "localhost")
        master_port = ds_config.get("master_port", "12355") # Use port from ds_config
        rank = "0"
        world_size = "1" # Assuming single process for this test

        # Store original env variables and set new ones
        original_env = {}
        vars_to_set = {
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": master_port,
            "RANK": rank,
            "WORLD_SIZE": world_size,
            "LOCAL_RANK": rank # Often needed as well
        }
        
        for var, value in vars_to_set.items():
            original_env[var] = os.environ.get(var) # Store original value (or None)
            os.environ[var] = value
            print(f"[Fixture] Set {var}={value}") # Add print statement for verification

        pipeline = None # Initialize pipeline to None for cleanup
        try:
            # Initialize pipeline - DeepSpeed should now use the env vars we set
            pipeline = RepControlPipelineDeepspeed(
                model=model,
                tokenizer=tokenizer,
                layers=[-1],  # Just control the last layer
                deepspeed_config=ds_config # Pass the config dict anyway
            )
            
            # Register hooks
            pipeline.register_hooks()
            
            yield pipeline

        finally:
            # Cleanup hooks if pipeline was initialized
            if pipeline is not None:
                pipeline.remove_hooks()
                
            # Restore original environment variables
            for var, original_value in original_env.items():
                if original_value is None:
                    # If the variable didn't exist originally, remove it
                    if var in os.environ:
                         print(f"[Fixture] Unsetting {var}")
                         del os.environ[var]
                else:
                    # Otherwise, restore its original value
                    print(f"[Fixture] Restoring {var}={original_value}")
                    os.environ[var] = original_value
    
    @pytest.mark.basic
    def test_initialization(self, pipeline):
        """Test pipeline initialization and DeepSpeed configuration"""
        # Check DeepSpeed wrapper
        assert hasattr(pipeline.model, "module"), "Model should be wrapped by DeepSpeed"
        
        # Check pipeline configuration
        assert pipeline.pipeline_world_size >= 1, "Pipeline world size should be at least 1"
        assert pipeline.pipeline_rank >= 0, "Pipeline rank should be valid"
        
        # Check layer mapping
        assert len(pipeline.local_layers) > 0, "Local layers should be mapped"
    
    @pytest.mark.basic
    def test_hook_registration(self, pipeline):
        """Test hook registration and controller setting"""
        # Create test activations
        hidden_size = 768  # GPT2 hidden size
        test_activations = {
            -1: torch.ones(1, 1, hidden_size).to(pipeline.model.device) * 0.1
        }
        
        # Set controller
        pipeline.set_controller(test_activations)
        
        # Verify hooks are registered
        assert len(pipeline.hooks) > 0, "Hooks should be registered"
        
        # Verify controller is set
        for layer_id, (hook, _) in pipeline.hooks.items():
            assert hook.controller is not None, f"Controller should be set for layer {layer_id}"
    
    @pytest.mark.basic
    def test_text_generation(self, pipeline, model_and_tokenizer):
        """Test basic text generation without activation modifications"""
        prompt = "Once upon a time, there was a"
        
        # Generate text
        outputs = pipeline(prompt, max_new_tokens=20)
        
        # Check output format
        assert isinstance(outputs, list), "Output should be a list"
        assert "generated_text" in outputs[0], "Output should contain generated_text"
        assert len(outputs[0]["generated_text"]) > len(prompt), "Output should be longer than prompt"
    
    @pytest.mark.basic
    def test_activation_modification(self, pipeline):
        """Test activation modification during generation"""
        # Create test activations - using a large value to ensure visible effect
        hidden_size = 768  # GPT2 hidden size
        test_activations = {
            -1: torch.ones(1, 1, hidden_size).to(pipeline.model.device) * 5.0
        }
        
        prompt = "Once upon a time, there was a"
        
        # Generate text without activations
        pipeline.reset()
        outputs_base = pipeline(prompt, max_new_tokens=20)
        
        # Generate text with activations
        outputs_modified = pipeline(
            prompt,
            activations=test_activations,
            max_new_tokens=20
        )
        
        # The outputs should be different with such a large activation modifier
        assert outputs_base[0]["generated_text"] != outputs_modified[0]["generated_text"], \
            f"Activation modification generated text\n{outputs_modified[0]['generated_text']}\nshould be different from base text\n{outputs_base[0]['generated_text']}"
    
    @pytest.mark.basic
    def test_batch_processing(self, pipeline):
        """Test batch processing capability"""
        prompts = [
            "Once upon a time,",
            "In a galaxy far away,",
            "It was a dark and stormy night,"
        ]
        
        # Generate with batch_size=2 (so we need multiple batches)
        outputs = pipeline(prompts, batch_size=2, max_new_tokens=10)
        
        # Check outputs
        assert len(outputs) == 1, "Should return a list with one element"
        assert len(outputs[0]) == len(prompts), "Should generate output for each prompt"
        
        # Check content
        for i, out in enumerate(outputs[0]):
            assert isinstance(out, dict), "Each output should be a dictionary"
            assert "generated_text" in out, "Each output should have generated_text"
            assert prompts[i] in out["generated_text"], "Output should contain the prompt"
    
    @pytest.mark.basic
    def test_reset(self, pipeline):
        """Test hook reset functionality"""
        # Create test activations
        hidden_size = 768  # GPT2 hidden size
        test_activations = {
            -1: torch.ones(1, 1, hidden_size).to(pipeline.model.device) * 0.1
        }
        
        # Set controller
        pipeline.set_controller(test_activations)
        
        # Verify controller is set
        for layer_id, (hook, _) in pipeline.hooks.items():
            assert hook.controller is not None, f"Controller should be set for layer {layer_id}"
        
        # Reset hooks
        pipeline.reset()
        
        # Verify controller is reset
        for layer_id, (hook, _) in pipeline.hooks.items():
            assert hook.controller is None, f"Controller should be reset for layer {layer_id}"
    
    @pytest.mark.basic
    def test_remove_hooks(self, model_and_tokenizer, ds_config):
        """Test hook removal"""
        model, tokenizer = model_and_tokenizer
        
        # Create a new pipeline instance
        pipeline = RepControlPipelineDeepspeed(
            model=model,
            tokenizer=tokenizer,
            layers=[-1],
            deepspeed_config=ds_config
        )
        
        # Register hooks
        pipeline.register_hooks()
        
        # Store initial hook count
        initial_hook_count = len(pipeline.hooks)
        assert initial_hook_count > 0, "Hooks should be registered"
        
        # Remove hooks
        pipeline.remove_hooks()
        
        # Verify hooks are removed
        assert len(pipeline.hooks) == 0, "All hooks should be removed"
    
    @pytest.mark.advanced
    def test_token_position_specific(self, pipeline):
        """Test hook with specific token position control"""
        hidden_size = 768  # GPT2 hidden size
        prompt = "Once upon a time, there was a"
        
        # Create test activations for controlling only specific tokens
        test_activations = {
            -1: torch.ones(1, 1, hidden_size).to(pipeline.model.device) * 10.0  # Large value for clear effect
        }
        
        # Generate text without modifications
        pipeline.reset()
        outputs_base = pipeline(prompt, max_new_tokens=20)
        
        # Test with first token position
        outputs_first = pipeline(
            prompt,
            activations=test_activations,
            token_pos=0,  # Only modify first token
            max_new_tokens=20
        )
        
        # Test with last token position
        outputs_last = pipeline(
            prompt,
            activations=test_activations,
            token_pos=-1,  # Only modify last token
            max_new_tokens=20
        )
        
        # Test with multiple specific positions
        outputs_multi = pipeline(
            prompt,
            activations=test_activations,
            token_pos=[0, 2, 4],  # Modify tokens at positions 0, 2, and 4
            max_new_tokens=20
        )
        
        # Verify all outputs are different due to different token positions being affected
        assert outputs_base[0]["generated_text"] != outputs_first[0]["generated_text"], \
            "Modifying first token should produce different output"
        
        assert outputs_first[0]["generated_text"] != outputs_last[0]["generated_text"], \
            "Modifying first vs. last token should produce different outputs"
        
        assert outputs_multi[0]["generated_text"] != outputs_first[0]["generated_text"], \
            "Modifying multiple tokens should produce different output from first token only"
    
    @pytest.mark.advanced
    def test_hook_operators(self, pipeline):
        """Test different hook operators"""
        hidden_size = 768  # GPT2 hidden size
        prompt = "Once upon a time, there was a"
        
        # Create test activations 
        test_activations = {
            -1: torch.ones(1, 1, hidden_size).to(pipeline.model.device) * 5.0
        }
        
        # Generate with linear_comb operator (default)
        outputs_linear = pipeline(
            prompt,
            activations=test_activations,
            operator='linear_comb',
            max_new_tokens=20
        )
        
        # Generate with piecewise_linear operator
        outputs_piecewise = pipeline(
            prompt,
            activations=test_activations,
            operator='piecewise_linear',
            max_new_tokens=20
        )
        
        # Different operators should produce different results
        assert outputs_linear[0]["generated_text"] != outputs_piecewise[0]["generated_text"], \
            "Different operators should produce different outputs"
    
    @pytest.mark.advanced
    def test_hook_normalization(self, pipeline):
        """Test normalization in hooks"""
        hidden_size = 768  # GPT2 hidden size
        prompt = "Once upon a time, there was a"
        
        # Create test activations
        test_activations = {
            -1: torch.ones(1, 1, hidden_size).to(pipeline.model.device) * 10.0
        }
        
        # Generate without normalization
        outputs_no_norm = pipeline(
            prompt,
            activations=test_activations,
            normalize=False,
            max_new_tokens=20
        )
        
        # Generate with normalization
        outputs_norm = pipeline(
            prompt,
            activations=test_activations,
            normalize=True,
            max_new_tokens=20
        )
        
        # Normalization should affect the output
        assert outputs_no_norm[0]["generated_text"] != outputs_norm[0]["generated_text"], \
            "Normalization should affect the generated text"
    
    @pytest.mark.advanced
    def test_hook_with_masks(self, pipeline):
        """Test hooks with masks"""
        hidden_size = 768  # GPT2 hidden size
        prompt = "Once upon a time, there was a"
        
        # Create test activations
        test_activations = {
            -1: torch.ones(1, 1, hidden_size).to(pipeline.model.device) * 5.0
        }
        
        # Create a binary mask (1s for even indices, 0s for odd indices)
        token_count = len(pipeline.tokenizer.encode(prompt))
        checkerboard_mask = torch.zeros(1, token_count, 1, device=pipeline.model.device)
        checkerboard_mask[0, ::2, 0] = 1.0  # Set even positions to 1
        
        # Generate without mask
        outputs_no_mask = pipeline(
            prompt,
            activations=test_activations,
            max_new_tokens=20
        )
        
        # Generate with mask
        outputs_with_mask = pipeline(
            prompt,
            activations=test_activations,
            masks=checkerboard_mask,
            max_new_tokens=20
        )
        
        # Mask should affect the output
        assert outputs_no_mask[0]["generated_text"] != outputs_with_mask[0]["generated_text"], \
            "Using a mask should affect the generated text"
    
    @pytest.mark.advanced
    def test_different_activation_patterns(self, pipeline):
        """Test hooks with different activation patterns"""
        hidden_size = 768  # GPT2 hidden size
        prompt = "Once upon a time, there was a"
        
        # Generate reference output with no modifications
        pipeline.reset()
        reference_output = pipeline(prompt, max_new_tokens=20)
        
        # Test with uniform activations
        uniform_activations = {
            -1: torch.ones(1, 1, hidden_size).to(pipeline.model.device) * 3.0
        }
        
        # Test with random activations
        random_activations = {
            -1: torch.randn(1, 1, hidden_size).to(pipeline.model.device) * 3.0
        }
        
        # Test with structured activations (alternating positive/negative)
        structured_activations = {
            -1: torch.ones(1, 1, hidden_size).to(pipeline.model.device)
        }
        # Make every other dimension negative
        structured_activations[-1][:, :, ::2] *= -1
        structured_activations[-1] *= 3.0
        
        # Generate with each activation pattern
        outputs_uniform = pipeline(
            prompt,
            activations=uniform_activations,
            max_new_tokens=20
        )
        
        outputs_random = pipeline(
            prompt,
            activations=random_activations,
            max_new_tokens=20
        )
        
        outputs_structured = pipeline(
            prompt,
            activations=structured_activations,
            max_new_tokens=20
        )
        
        # All patterns should affect the output differently
        assert reference_output[0]["generated_text"] != outputs_uniform[0]["generated_text"], \
            "Uniform activations should affect output"
        
        assert outputs_uniform[0]["generated_text"] != outputs_random[0]["generated_text"], \
            "Random activations should differ from uniform activations"
        
        assert outputs_random[0]["generated_text"] != outputs_structured[0]["generated_text"], \
            "Structured activations should differ from random activations"
    
    @pytest.mark.advanced
    def test_multiple_layer_hook_interaction(self, model_and_tokenizer, ds_config):
        """Test interaction between hooks on multiple layers"""
        model, tokenizer = model_and_tokenizer
        
        # Create a new pipeline with multiple layer hooks
        pipeline = RepControlPipelineDeepspeed(
            model=model,
            tokenizer=tokenizer,
            layers=[-1, -2, -3],  # Control multiple layers
            deepspeed_config=ds_config
        )
        
        # Register hooks
        pipeline.register_hooks()
        
        try:
            hidden_size = 768  # GPT2 hidden size
            prompt = "Once upon a time, there was a"
            
            # Test activations for each layer individually
            layer1_activations = {
                -1: torch.ones(1, 1, hidden_size).to(pipeline.model.device) * 3.0
            }
            
            layer2_activations = {
                -2: torch.ones(1, 1, hidden_size).to(pipeline.model.device) * 3.0
            }
            
            layer3_activations = {
                -3: torch.ones(1, 1, hidden_size).to(pipeline.model.device) * 3.0
            }
            
            # Combined activations
            combined_activations = {
                -1: torch.ones(1, 1, hidden_size).to(pipeline.model.device) * 3.0,
                -2: torch.ones(1, 1, hidden_size).to(pipeline.model.device) * 3.0,
                -3: torch.ones(1, 1, hidden_size).to(pipeline.model.device) * 3.0
            }
            
            # Generate with each layer individually
            outputs_layer1 = pipeline(
                prompt,
                activations=layer1_activations,
                max_new_tokens=20
            )
            
            outputs_layer2 = pipeline(
                prompt,
                activations=layer2_activations,
                max_new_tokens=20
            )
            
            outputs_layer3 = pipeline(
                prompt,
                activations=layer3_activations,
                max_new_tokens=20
            )
            
            # Generate with all layers combined
            outputs_combined = pipeline(
                prompt,
                activations=combined_activations,
                max_new_tokens=20
            )
            
            # Each layer individually should produce different results
            assert outputs_layer1[0]["generated_text"] != outputs_layer2[0]["generated_text"], \
                "Different layers should produce different outputs"
            
            assert outputs_layer2[0]["generated_text"] != outputs_layer3[0]["generated_text"], \
                "Different layers should produce different outputs"
            
            # Combined effect should be different from individual layers
            assert outputs_layer1[0]["generated_text"] != outputs_combined[0]["generated_text"], \
                "Combined layers should differ from layer 1 alone"
            
            assert outputs_layer2[0]["generated_text"] != outputs_combined[0]["generated_text"], \
                "Combined layers should differ from layer 2 alone"
            
            assert outputs_layer3[0]["generated_text"] != outputs_combined[0]["generated_text"], \
                "Combined layers should differ from layer 3 alone"
            
        finally:
            # Clean up
            pipeline.remove_hooks()
    
    @pytest.mark.advanced
    def test_get_activations(self, pipeline):
        """Test capturing activations via hooks"""
        prompt = "Once upon a time, there was a"
        
        # Generate text without modifications to capture activations
        pipeline.reset()
        _ = pipeline(prompt, max_new_tokens=5, reset_hooks=False)  # Don't reset hooks to keep activations
        
        # Get captured activations
        activations = pipeline.get_activations()
        
        # Verify activations were captured
        assert len(activations) > 0, "Should have captured activations"
        
        # Check structure of captured activations
        for layer_id, activation in activations.items():
            assert isinstance(activation, torch.Tensor), f"Activation for layer {layer_id} should be a tensor"
            assert activation.dim() == 3, f"Activation shape should be 3D (batch, seq, hidden)"
            
            # The sequence length should be at least the number of tokens in the prompt
            min_seq_len = len(pipeline.tokenizer.encode(prompt))
            assert activation.shape[1] >= min_seq_len, f"Sequence dimension should be at least {min_seq_len}"
    
    @pytest.mark.advanced
    def test_consistency_with_repeated_runs(self, pipeline):
        """Test consistency of hook effects with identical inputs"""
        hidden_size = 768  # GPT2 hidden size
        prompt = "Once upon a time, there was a"
        
        # Create test activations
        test_activations = {
            -1: torch.ones(1, 1, hidden_size).to(pipeline.model.device) * 3.0
        }
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        # Generate first output
        pipeline.reset()
        outputs1 = pipeline(
            prompt,
            activations=test_activations,
            max_new_tokens=20,
            do_sample=False  # Deterministic generation
        )
        
        # Generate second output with identical settings
        pipeline.reset()
        outputs2 = pipeline(
            prompt,
            activations=test_activations,
            max_new_tokens=20,
            do_sample=False  # Deterministic generation
        )
        
        # Outputs should be identical with deterministic generation and same activations
        assert outputs1[0]["generated_text"] == outputs2[0]["generated_text"], \
            "Identical inputs and settings should produce identical outputs"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 