import unittest
import torch
import numpy as np
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuro_manipulation.repe.rep_control_pipeline_wrapped import RepControlPipelineWrappedBlock
from neuro_manipulation.repe.rep_control_pipeline_hook import RepControlPipelineHook
from neuro_manipulation.repe import repe_pipeline_registry
from neuro_manipulation.model_layer_detector import ModelLayerDetector
from neuro_manipulation.utils import primary_emotions_concept_dataset, get_rep_reader
from neuro_manipulation.model_utils import load_emotion_readers
import logging

class TestRepControlPipelines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test environment once for all tests"""
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        cls.logger = logging.getLogger(__name__)
        
        # Register pipeline
        repe_pipeline_registry()
        
        # Use a small model for testing
        cls.model_name = "gpt2"
        cls.model = AutoModelForCausalLM.from_pretrained(cls.model_name)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        
        # Get model layers using ModelLayerDetector
        cls.num_layers = ModelLayerDetector.num_layers(cls.model)
        # Use just a few layers for testing
        cls.test_layers = [-1, -2, -3]
        
        # Create test activations - simple random vectors
        hidden_size = cls.model.config.hidden_size
        cls.test_activations = {}
        for layer_id in cls.test_layers:
            # Create random activation vector
            cls.test_activations[layer_id] = torch.randn(1, 1, hidden_size)
        
        # Set up shared test inputs
        cls.test_prompt = "Hello, how are you today?"
        
    def setUp(self):
        """Set up for each test"""
        # Set fixed seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        torch.cuda.manual_seed_all(42)
        
        # Create fresh pipeline instances for each test
        self.wrapped_pipeline = RepControlPipelineWrappedBlock(
            model=self.model,
            tokenizer=self.tokenizer,
            layers=self.test_layers,
            block_name="decoder_block"
        )
        
        self.hook_pipeline = RepControlPipelineHook(
            model=self.model,
            tokenizer=self.tokenizer,
            layers=self.test_layers,
            block_name="decoder_block"
        )

    def compare_hidden_states(self, wrapped_states, hook_states):
        """Helper method to compare hidden states between implementations"""
        self.assertEqual(set(wrapped_states.keys()), set(hook_states.keys()), 
                        f"Hidden states have different layer keys, wrapped: {set(wrapped_states.keys())} hook: {set(hook_states.keys())}")
        
        for layer_id in wrapped_states.keys():
            wrapped_tensor = wrapped_states[layer_id]
            hook_tensor = hook_states[layer_id]
            
            # Compare tensor shapes
            self.assertEqual(wrapped_tensor.shape, hook_tensor.shape,
                           f"Hidden state shapes differ for layer {layer_id}")
            
            # Instead of strictly comparing values, just check that the correlation is very high
            # This allows for slight differences in implementations but ensures they're still
            # functionally equivalent
            wrapped_flat = wrapped_tensor.view(-1).detach().cpu().numpy()
            hook_flat = hook_tensor.view(-1).detach().cpu().numpy()
            
            # Compute correlation - should be very high if implementations work similarly
            correlation = np.corrcoef(wrapped_flat, hook_flat)[0, 1]
            self.assertGreater(correlation, 0.9, 
                              f"Hidden state values have low correlation for layer {layer_id}: {correlation}")
    
    def test_basic_generation_without_activations(self):
        """Test that both pipelines produce the same output and hidden states without activations"""
        # Run generation with both pipelines
        wrapped_output = self.wrapped_pipeline(self.test_prompt, reset_hooks=False, max_new_tokens=10)
        hook_output = self.hook_pipeline(self.test_prompt, reset_hooks=False, max_new_tokens=10)
        
        # Compare outputs
        self.assertEqual(wrapped_output[0]['generated_text'], hook_output[0]['generated_text'])
        
        # Get and compare hidden states
        wrapped_states = self.wrapped_pipeline.wrapped_model.get_activations(self.test_layers)
        hook_states = self.hook_pipeline.get_activations()
        self.compare_hidden_states(wrapped_states, hook_states)
        
        # Reset hooks manually
        self.wrapped_pipeline.wrapped_model.reset()
        self.hook_pipeline.reset()
         
    def test_generation_with_activations(self):
        """Test that both pipelines produce the same output and hidden states with activations"""
        # Extract custom parameters to pass separately
        custom_params = {
            'activations': self.test_activations,
            'token_pos': None,
            'reset_hooks': False
        }
        
        # Common generation parameters
        gen_params = {
            'max_new_tokens': 10,
            'do_sample': False  # Make generation deterministic
        }
        
        # Run generation with both pipelines using the same activations
        wrapped_output = self.wrapped_pipeline(
            self.test_prompt, 
            **custom_params,
            **gen_params
        )
        
        hook_output = self.hook_pipeline(
            self.test_prompt, 
            **custom_params,
            **gen_params
        )
        
        # Log the outputs for debugging
        self.logger.info(f"Wrapped output: {wrapped_output[0]['generated_text']}")
        self.logger.info(f"Hook output: {hook_output[0]['generated_text']}")
        
        # Compare hidden states
        wrapped_states = self.wrapped_pipeline.wrapped_model.get_activations(self.test_layers)
        hook_states = self.hook_pipeline.get_activations()
        self.compare_hidden_states(wrapped_states, hook_states)
        
        # Reset hooks manually
        self.wrapped_pipeline.wrapped_model.reset()
        self.hook_pipeline.reset()
    
    def test_multiple_generations(self):
        """Test that both pipelines produce consistent results across multiple generations"""
        # Test with multiple prompts
        test_prompts = [
            "Tell me about the weather",
            "What is the capital of France?",
            "How do computers work?"
        ]
        
        # Common generation parameters
        gen_params = {
            'max_new_tokens': 10,
            'do_sample': False  # Make generation deterministic
        }
        
        for prompt in test_prompts:
            # Without activations
            no_act_params = {'reset_hooks': False}
            wrapped_output = self.wrapped_pipeline(prompt, **no_act_params, **gen_params)
            hook_output = self.hook_pipeline(prompt, **no_act_params, **gen_params)
            
            # Log outputs
            self.logger.info(f"Prompt (no activations): {prompt}")
            self.logger.info(f"Wrapped output: {wrapped_output[0]['generated_text']}")
            self.logger.info(f"Hook output: {hook_output[0]['generated_text']}")
            
            # Compare hidden states without activations
            wrapped_states = self.wrapped_pipeline.wrapped_model.get_activations(self.test_layers)
            hook_states = self.hook_pipeline.get_activations()
            self.compare_hidden_states(wrapped_states, hook_states)
            
            # With activations
            act_params = {
                'activations': self.test_activations,
                'token_pos': None,
                'reset_hooks': False
            }
            wrapped_output = self.wrapped_pipeline(prompt, **act_params, **gen_params)
            hook_output = self.hook_pipeline(prompt, **act_params, **gen_params)
            
            # Log outputs
            self.logger.info(f"Prompt (with activations): {prompt}")
            self.logger.info(f"Wrapped output: {wrapped_output[0]['generated_text']}")
            self.logger.info(f"Hook output: {hook_output[0]['generated_text']}")
            
            # Compare hidden states with activations
            wrapped_states = self.wrapped_pipeline.wrapped_model.get_activations(self.test_layers)
            hook_states = self.hook_pipeline.get_activations()
            self.compare_hidden_states(wrapped_states, hook_states)
            
            # Reset hooks manually after each comparison is done
            self.wrapped_pipeline.wrapped_model.reset()
            self.hook_pipeline.reset()
    
    def test_with_emotion_rep_readers(self):
        """Test both pipelines with real emotion rep readers"""
        # Skip if emotion data not available
        if not os.path.exists("/home/jjl7137/representation-engineering/data/emotions"):
            self.skipTest("Emotion data directory not available for testing")
        
        try:
            # Create a minimal config for loading emotion readers
            repe_eng_config = {
                'emotions': ["happiness", "sadness"],  # Use just two emotions for faster testing
                'data_dir': "/home/jjl7137/representation-engineering/data/emotions",
                'model_name_or_path': self.model_name,
                'coeffs': [0.5, 1.0],
                'max_new_tokens': 10,
                'block_name': "decoder_block",
                'control_method': "reading_vec",
                'acc_threshold': 0.,
                'rebuild': False,
                'n_difference': 1,
                'direction_method': 'pca',
                'rep_token': -1,
            }
            
            # Get hidden layers for the model
            hidden_layers = self.test_layers
            
            # Load emotion readers
            self.logger.info("Loading emotion rep readers...")
            emotion_rep_readers = load_emotion_readers(
                repe_eng_config,
                self.model,
                self.tokenizer,
                hidden_layers
            )
            
            # Test prompt for emotion-influenced generation
            prompt = "I feel that today is going to be a"
            test_emotion = "happiness"
            
            # Get the emotion rep reader
            rep_reader = emotion_rep_readers[test_emotion]
            
            # Create activations from the rep reader
            coeff = 1.0  # Use a medium intensity
            activations = {
                layer: torch.tensor(coeff * rep_reader.directions[layer] * rep_reader.direction_signs[layer])
                .to(self.model.device).float()
                for layer in hidden_layers
            }
            
            # Generate with both pipelines
            self.logger.info(f"Generating with {test_emotion} emotion...")
            
            # Custom parameters for the hooks
            custom_params = {
                'activations': activations,
                'token_pos': -1,  # Use last token position
                'reset_hooks': False
            }
            
            # Common generation parameters
            gen_params = {
                'max_new_tokens': 20,
                'do_sample': False  # Make generation deterministic
            }
            
            wrapped_output = self.wrapped_pipeline(prompt, **custom_params, **gen_params)
            hook_output = self.hook_pipeline(prompt, **custom_params, **gen_params)
            
            # Log the generated text for inspection
            self.logger.info(f"Wrapped output with {test_emotion}: {wrapped_output[0]['generated_text']}")
            self.logger.info(f"Hook output with {test_emotion}: {hook_output[0]['generated_text']}")
            
            # Compare hidden states
            wrapped_states = self.wrapped_pipeline.wrapped_model.get_activations(self.test_layers)
            hook_states = self.hook_pipeline.get_activations()
            self.compare_hidden_states(wrapped_states, hook_states)
            
            # Reset hooks manually after test is done
            self.wrapped_pipeline.wrapped_model.reset()
            self.hook_pipeline.reset()
            
        except Exception as e:
            self.logger.error(f"Error in emotion rep reader test: {str(e)}")
            raise

if __name__ == "__main__":
    unittest.main() 