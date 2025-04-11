import unittest
import torch
import yaml
from pathlib import Path
from neuro_manipulation.experiments.emotion_game_experiment import EmotionGameExperiment
from neuro_manipulation.game_theory_exp_0205 import repe_pipeline_registry
from neuro_manipulation.configs.experiment_config import get_exp_config, get_repe_eng_config, get_game_config
from constants import GameNames
import logging


class TestEmotionGameExperiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup test environment once for all tests"""
        # Register pipeline
        repe_pipeline_registry()
        
        # Load test configuration
        cls.exp_config = get_exp_config('config/test_emotion_game_config.yaml')
        cls.game_name = GameNames.from_string(cls.exp_config['experiment']['game']['name'])
        cls.model_name = cls.exp_config['experiment']['llm']['model_name']
        cls.repe_eng_config = get_repe_eng_config(cls.model_name)
        cls.game_config = get_game_config(cls.game_name)
        
        if cls.game_name.is_sequential():
            cls.game_config['previous_actions_length'] = cls.exp_config['experiment']['game']['previous_actions_length']
    
    def setUp(self):
        """Setup for each test"""
        self.experiment = EmotionGameExperiment(
            self.repe_eng_config,
            self.exp_config,
            self.game_config,
            batch_size=2,
            repeat=self.exp_config['experiment']['repeat']
        )
        
    def test_post_process_batch_alignment(self):
        """Test if the batch processing maintains proper alignment of inputs and outputs"""
        # Create a mock batch with known values
        batch = {
            'prompt': ['prompt1', 'prompt2'],
            'scenario': ['scenario1', 'scenario2'],
            'description': ['desc1', 'desc2'],
            'options': [('Cooperate1', 'Defect1'), ('Cooperate2', 'Defect2')]  # Using tuples for options
        }
        
        # Create mock control outputs that include the prompts
        control_outputs = [
            [{'generated_text': 'prompt1 {"decision": "Cooperate", "rationale": "why1", "option_id": 1}'}],
            [{'generated_text': 'prompt2 {"decision": "Defect", "rationale": "why2", "option_id": 2}'}]
        ]
        
        self.experiment.cur_emotion = 'happy'
        self.experiment.cur_coeff = 0.5
        
        # Process batch
        results = self.experiment._post_process_batch(batch, control_outputs)
        
        # Verify alignments
        self.assertEqual(len(results), 2, "Should have same number of results as inputs")
        
        # Check first result
        self.assertEqual(results[0]['scenario'], 'scenario1')
        self.assertEqual(results[0]['description'], 'desc1')
        self.assertEqual(results[0]['input'], 'prompt1')
        self.assertEqual(results[0]['decision'], 'Cooperate')
        self.assertEqual(results[0]['rationale'], 'why1')
        self.assertEqual(results[0]['category'], 1)
        
        # Check second result
        self.assertEqual(results[1]['scenario'], 'scenario2')
        self.assertEqual(results[1]['description'], 'desc2')
        self.assertEqual(results[1]['input'], 'prompt2')
        self.assertEqual(results[1]['decision'], 'Defect')
        self.assertEqual(results[1]['rationale'], 'why2')
        self.assertEqual(results[1]['category'], 2)
        
    def test_post_process_batch_with_repeats(self):
        """Test if the batch processing handles repeats correctly"""
        # Create a mock batch with known values
        batch = {
            'prompt': ['prompt1'],
            'scenario': ['scenario1'],
            'description': ['desc1'],
            'options': [['Cooperate', 'Defect']]  # Using actual game options
        }
        
        # Create mock control outputs with repeated data
        control_outputs = [
            [{'generated_text': 'prompt1 {"decision": "Cooperate", "rationale": "why1", "option_id": 1}'}],
            [{'generated_text': 'prompt1 {"decision": "Defect", "rationale": "why1_repeat", "option_id": 2}'}]
        ]
        
        self.experiment.cur_emotion = 'happy'
        self.experiment.cur_coeff = 0.5
        
        # Process batch
        results = self.experiment._post_process_batch(batch, control_outputs)
        
        # Verify repeats
        self.assertEqual(len(results), 2, "Should have number of results equal to repeats")
        
        # Check first result
        self.assertEqual(results[0]['scenario'], 'scenario1')
        self.assertEqual(results[0]['repeat_num'], 0)
        self.assertEqual(results[0]['decision'], 'Cooperate')
        
        # Check second result (repeated)
        self.assertEqual(results[1]['scenario'], 'scenario1')
        self.assertEqual(results[1]['repeat_num'], 1)
        self.assertEqual(results[1]['decision'], 'Defect')

if __name__ == '__main__':
    unittest.main() 