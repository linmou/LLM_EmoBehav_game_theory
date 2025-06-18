#!/usr/bin/env python3
"""
Unit tests for the anger activation demo.

Tests the AngerActivationDemo class functionality without requiring
full model loading for faster testing.
"""

import unittest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

class TestAngerActivationDemo(unittest.TestCase):
    """Test cases for AngerActivationDemo class."""

    def setUp(self):
        """Set up test fixtures."""
        self.model_path = "/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct"
        
    @patch('neuro_manipulation.repe.sequence_prob_demo.setup_model_and_tokenizer')
    @patch('neuro_manipulation.repe.sequence_prob_demo.load_emotion_readers')
    @patch('neuro_manipulation.repe.sequence_prob_demo.get_pipeline')
    @patch('neuro_manipulation.repe.sequence_prob_demo.ModelLayerDetector')
    def test_demo_initialization(self, mock_detector, mock_pipeline, mock_readers, mock_setup):
        """Test AngerActivationDemo initialization."""
        # Mock dependencies
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [26921]  # Mock token for "angry"
        mock_prompt_format = "test_format"
        
        mock_setup.return_value = (mock_model, mock_tokenizer, mock_prompt_format)
        mock_detector.num_layers.return_value = 24
        mock_readers.return_value = {"anger": Mock()}
        mock_pipeline.return_value = Mock()
        
        # Import and initialize demo
        from neuro_manipulation.repe.sequence_prob_demo import AngerActivationDemo
        demo = AngerActivationDemo(self.model_path)
        
        # Verify initialization
        self.assertEqual(demo.model_path, self.model_path)
        self.assertEqual(demo.target_word, "angry")
        self.assertEqual(demo.target_tokens, [26921])
        self.assertIsNotNone(demo.hidden_layers)
        
        # Verify mocked calls
        self.assertEqual(mock_setup.call_count, 2)  # Called twice: HF then vLLM
        mock_readers.assert_called_once()
        mock_pipeline.assert_called_once()

    def test_create_test_prompts(self):
        """Test test prompt creation."""
        # Mock the demo class without full initialization
        with patch('neuro_manipulation.repe.sequence_prob_demo.AngerActivationDemo.__init__', return_value=None):
            from neuro_manipulation.repe.sequence_prob_demo import AngerActivationDemo
            demo = AngerActivationDemo.__new__(AngerActivationDemo)
            
            prompts = demo.create_test_prompts()
            
            # Verify prompt properties
            self.assertIsInstance(prompts, list)
            self.assertGreater(len(prompts), 5)
            self.assertTrue(all(isinstance(p, str) for p in prompts))
            
            # Check some expected prompts
            expected_fragments = ["felt very", "reaction was", "became increasingly"]
            self.assertTrue(any(frag in prompt for prompt in prompts for frag in expected_fragments))

    def test_get_anger_activations(self):
        """Test anger activation generation."""
        # Create mock emotion reader
        mock_reader = Mock()
        mock_reader.directions = {-1: np.array([0.1, 0.2, 0.3]), -2: np.array([0.4, 0.5, 0.6])}
        mock_reader.direction_signs = {-1: 1.0, -2: -1.0}
        
        # Mock the demo class
        with patch('neuro_manipulation.repe.sequence_prob_demo.AngerActivationDemo.__init__', return_value=None):
            from neuro_manipulation.repe.sequence_prob_demo import AngerActivationDemo
            demo = AngerActivationDemo.__new__(AngerActivationDemo)
            demo.emotion_rep_readers = {"anger": mock_reader}
            demo.hidden_layers = [-1, -2]
            demo.device = torch.device("cpu")
            
            # Test activation generation
            intensity = 1.5
            activations = demo.get_anger_activations(intensity)
            
            # Verify activation structure
            self.assertIsInstance(activations, dict)
            self.assertEqual(set(activations.keys()), {-1, -2})
            
            # Verify activation values
            self.assertIsInstance(activations[-1], torch.Tensor)
            self.assertIsInstance(activations[-2], torch.Tensor)
            
            # Check that activations are properly scaled
            expected_1 = torch.tensor(intensity * mock_reader.directions[-1] * mock_reader.direction_signs[-1])
            expected_2 = torch.tensor(intensity * mock_reader.directions[-2] * mock_reader.direction_signs[-2])
            
            torch.testing.assert_close(activations[-1].float(), expected_1.float())
            torch.testing.assert_close(activations[-2].float(), expected_2.float())

    @patch('neuro_manipulation.repe.sequence_prob_demo.CombinedVLLMHook')
    def test_measure_word_probability(self, mock_hook_class):
        """Test word probability measurement."""
        # Setup mocks
        mock_hook = Mock()
        mock_hook.get_log_prob.return_value = [
            {'sequence': 'angry', 'prob': 0.1},
            {'sequence': 'angry', 'prob': 0.2}
        ]
        mock_hook_class.return_value = mock_hook
        
        mock_pipeline = Mock()
        mock_pipeline.return_value = [Mock(), Mock()]
        
        # Mock the demo class
        with patch('neuro_manipulation.repe.sequence_prob_demo.AngerActivationDemo.__init__', return_value=None):
            from neuro_manipulation.repe.sequence_prob_demo import AngerActivationDemo
            demo = AngerActivationDemo.__new__(AngerActivationDemo)
            demo.model = Mock()
            demo.target_word = "angry"
            demo.tokenizer = Mock()
            demo.rep_control_pipeline = mock_pipeline
            
            prompts = ["Test prompt 1", "Test prompt 2"]
            activations = {-1: torch.tensor([0.1, 0.2])}
            
            # Test probability measurement
            probabilities = demo.measure_word_probability(prompts, activations)
            
            # Verify results
            self.assertEqual(len(probabilities), 2)
            self.assertEqual(probabilities, [0.1, 0.2])
            
            # Verify hook usage
            mock_hook.get_log_prob.assert_called_once()

    @patch('neuro_manipulation.repe.sequence_prob_demo.AngerActivationDemo.measure_word_probability')
    @patch('neuro_manipulation.repe.sequence_prob_demo.AngerActivationDemo.get_anger_activations')
    @patch('neuro_manipulation.repe.sequence_prob_demo.AngerActivationDemo.create_test_prompts')
    def test_run_probability_analysis(self, mock_prompts, mock_activations, mock_measure):
        """Test complete probability analysis."""
        # Setup mocks
        mock_prompts.return_value = ["prompt1", "prompt2"]
        mock_activations.return_value = {-1: torch.tensor([0.1])}
        mock_measure.side_effect = [
            [0.01, 0.02],  # baseline
            [0.02, 0.03],  # intensity 0.5
            [0.03, 0.04],  # intensity 1.0
            [0.04, 0.05],  # intensity 1.5
            [0.05, 0.06],  # intensity 2.0
        ]
        
        # Mock the demo class
        with patch('neuro_manipulation.repe.sequence_prob_demo.AngerActivationDemo.__init__', return_value=None):
            from neuro_manipulation.repe.sequence_prob_demo import AngerActivationDemo
            demo = AngerActivationDemo.__new__(AngerActivationDemo)
            demo.target_word = "angry"
            
            # Test analysis
            results = demo.run_probability_analysis()
            
            # Verify results structure
            self.assertIn('prompts', results)
            self.assertIn('intensities', results)
            self.assertIn('probabilities', results)
            
            # Verify intensities
            expected_intensities = [0.0, 0.5, 1.0, 1.5, 2.0]
            self.assertEqual(results['intensities'], expected_intensities)
            
            # Verify probability conditions
            expected_conditions = ['baseline', 'anger_0.5', 'anger_1.0', 'anger_1.5', 'anger_2.0']
            self.assertEqual(set(results['probabilities'].keys()), set(expected_conditions))
            
            # Verify call counts
            self.assertEqual(mock_measure.call_count, 5)
            self.assertEqual(mock_activations.call_count, 4)  # Not called for baseline

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.figure')
    def test_create_visualization(self, mock_figure, mock_savefig):
        """Test visualization creation."""
        # Setup test data
        results = {
            'intensities': [0.0, 1.0, 2.0],
            'prompts': ['prompt1', 'prompt2'],
            'probabilities': {
                'baseline': [0.01, 0.02],
                'anger_1.0': [0.03, 0.04],
                'anger_2.0': [0.05, 0.06]
            }
        }
        
        # Mock the demo class
        with patch('neuro_manipulation.repe.sequence_prob_demo.AngerActivationDemo.__init__', return_value=None):
            from neuro_manipulation.repe.sequence_prob_demo import AngerActivationDemo
            demo = AngerActivationDemo.__new__(AngerActivationDemo)
            demo.target_word = "angry"
            
            # Test visualization
            with patch('neuro_manipulation.repe.sequence_prob_demo.Path') as mock_path:
                mock_path.return_value.mkdir.return_value = None
                plot_path = demo.create_visualization(results)
                
                # Verify plot creation
                mock_figure.assert_called_once()
                mock_savefig.assert_called_once()
                self.assertIsNotNone(plot_path)

    def test_emotion_intensity_scaling(self):
        """Test that emotion intensities are properly scaled."""
        # Test different intensity values
        intensities = [0.0, 0.5, 1.0, 1.5, 2.0]
        base_direction = np.array([1.0, 2.0, 3.0])
        direction_sign = 1.0
        
        for intensity in intensities:
            expected = intensity * base_direction * direction_sign
            
            # Mock reader
            mock_reader = Mock()
            mock_reader.directions = {-1: base_direction}
            mock_reader.direction_signs = {-1: direction_sign}
            
            # Mock demo
            with patch('neuro_manipulation.repe.sequence_prob_demo.AngerActivationDemo.__init__', return_value=None):
                from neuro_manipulation.repe.sequence_prob_demo import AngerActivationDemo
                demo = AngerActivationDemo.__new__(AngerActivationDemo)
                demo.emotion_rep_readers = {"anger": mock_reader}
                demo.hidden_layers = [-1]
                demo.device = torch.device("cpu")
                
                activations = demo.get_anger_activations(intensity)
                actual = activations[-1].numpy()
                
                np.testing.assert_array_almost_equal(actual, expected, decimal=6)

    def test_target_word_tokenization(self):
        """Test that target word is properly tokenized."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [12345, 67890]
        
        with patch('neuro_manipulation.repe.sequence_prob_demo.AngerActivationDemo.__init__', return_value=None):
            from neuro_manipulation.repe.sequence_prob_demo import AngerActivationDemo
            demo = AngerActivationDemo.__new__(AngerActivationDemo)
            demo.tokenizer = mock_tokenizer
            demo.target_word = "angry"
            demo.target_tokens = demo.tokenizer.encode(demo.target_word, add_special_tokens=False)
            
            self.assertEqual(demo.target_tokens, [12345, 67890])
            mock_tokenizer.encode.assert_called_with("angry", add_special_tokens=False)


class TestDemoIntegration(unittest.TestCase):
    """Integration tests for demo functionality."""
    
    def test_probability_trend_validation(self):
        """Test that probabilities show expected trend with increasing intensity."""
        # Simulate realistic probability data
        baseline_probs = [0.001, 0.002, 0.001, 0.003]
        low_intensity_probs = [0.003, 0.004, 0.002, 0.005]
        high_intensity_probs = [0.008, 0.009, 0.007, 0.012]
        
        # Calculate averages
        baseline_avg = np.mean(baseline_probs)
        low_avg = np.mean(low_intensity_probs)
        high_avg = np.mean(high_intensity_probs)
        
        # Validate expected trend
        self.assertLess(baseline_avg, low_avg, "Low intensity should be higher than baseline")
        self.assertLess(low_avg, high_avg, "High intensity should be higher than low intensity")
        
        # Calculate improvement ratios
        low_ratio = low_avg / baseline_avg
        high_ratio = high_avg / baseline_avg
        
        self.assertGreater(low_ratio, 1.0, "Low intensity should show improvement")
        self.assertGreater(high_ratio, low_ratio, "High intensity should show more improvement")

    def test_error_handling(self):
        """Test error handling in demo methods."""
        with patch('neuro_manipulation.repe.sequence_prob_demo.AngerActivationDemo.__init__', return_value=None):
            from neuro_manipulation.repe.sequence_prob_demo import AngerActivationDemo
            demo = AngerActivationDemo.__new__(AngerActivationDemo)
            
            # Test with invalid emotion reader
            demo.emotion_rep_readers = {}
            demo.hidden_layers = [-1]
            demo.device = torch.device("cpu")
            
            with self.assertRaises(KeyError):
                demo.get_anger_activations(1.0)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2) 