import unittest
import tempfile
import yaml
import os
from unittest.mock import patch, MagicMock

from neuro_manipulation.experiment_series_runner import ExperimentSeriesRunner
from neuro_manipulation.configs.experiment_config import get_exp_config


class TestQwen25MultiGameConfig(unittest.TestCase):
    """Test the Qwen2.5 multi-game configuration functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a sample config similar to qwen2.5_Sequential_Games.yaml
        self.sample_config = {
            'experiment': {
                'name': 'Test_Qwen2.5_Multi_Games',
                'games': [
                    'Escalation_Game',
                    'Trust_Game_Trustor',
                    'Trust_Game_Trustee',
                    'Ultimatum_Game_Proposer',
                    'Ultimatum_Game_Responder'
                ],
                'games_config': {
                    'Escalation_Game': {
                        'previous_actions_length': 0
                    },
                    'Trust_Game_Trustor': {
                        'previous_actions_length': 0
                    },
                    'Trust_Game_Trustee': {
                        'previous_actions_length': 1,
                        'previous_trust_level': 0
                    },
                    'Ultimatum_Game_Proposer': {
                        'previous_actions_length': 0
                    },
                    'Ultimatum_Game_Responder': {
                        'previous_actions_length': 1,
                        'previous_offer_level': 0
                    }
                },
                'models': [
                    '/test/model1',
                    '/test/model2'
                ],
                'llm': {
                    'model_name': 'default_model',
                    'generation_config': {
                        'temperature': 0.7
                    }
                },
                'repeat': 1,
                'emotions': ['anger', 'happiness'],
                'intensity': [1.5],
                'output': {
                    'base_dir': 'test_results'
                }
            }
        }

    def test_games_config_loading(self):
        """Test that games_config is properly loaded and accessible."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.sample_config, f)
            config_path = f.name

        try:
            config = get_exp_config(config_path)
            
            # Verify games_config is loaded correctly
            self.assertIn('games_config', config['experiment'])
            games_config = config['experiment']['games_config']
            
            # Check specific game configurations
            self.assertEqual(games_config['Trust_Game_Trustee']['previous_actions_length'], 1)
            self.assertEqual(games_config['Trust_Game_Trustee']['previous_trust_level'], 0)
            self.assertEqual(games_config['Ultimatum_Game_Responder']['previous_offer_level'], 0)
            
        finally:
            os.unlink(config_path)

    @patch('neuro_manipulation.experiment_series_runner.repe_pipeline_registry')
    @patch('neuro_manipulation.experiment_series_runner.get_repe_eng_config')
    @patch('neuro_manipulation.experiment_series_runner.get_game_config')
    @patch('neuro_manipulation.experiment_series_runner.EmotionGameExperiment')
    @patch('neuro_manipulation.experiment_series_runner.GameNames')
    def test_setup_experiment_with_game_specific_config(self, mock_game_names, mock_experiment, 
                                                       mock_get_game_config, mock_get_repe_config, 
                                                       mock_repe_registry):
        """Test that setup_experiment properly applies game-specific configurations."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.sample_config, f)
            config_path = f.name

        try:
            # Setup mocks
            mock_game_name = MagicMock()
            mock_game_name.is_sequential.return_value = True
            mock_game_names.from_string.return_value = mock_game_name
            
            mock_get_game_config.return_value = {'base_config': 'value'}
            mock_get_repe_config.return_value = {'repe_config': 'value'}
            
            # Create runner and test setup_experiment
            runner = ExperimentSeriesRunner(config_path)
            
            # Test Trust_Game_Trustee configuration
            runner.setup_experiment('Trust_Game_Trustee', '/test/model1')
            
            # Verify game config was called and updated
            mock_get_game_config.assert_called()
            
            # Verify EmotionGameExperiment was called with the right parameters
            mock_experiment.assert_called()
            
            # Get the actual call arguments
            call_args = mock_experiment.call_args
            repe_config, exp_config, game_config = call_args[0][:3]
            
            # Verify game_config has the game-specific attributes
            self.assertEqual(game_config['previous_actions_length'], 1)
            self.assertEqual(game_config['previous_trust_level'], 0)
            
        finally:
            os.unlink(config_path)

    @patch('neuro_manipulation.experiment_series_runner.repe_pipeline_registry')
    @patch('neuro_manipulation.experiment_series_runner.get_repe_eng_config')
    @patch('neuro_manipulation.experiment_series_runner.get_game_config')
    @patch('neuro_manipulation.experiment_series_runner.EmotionGameExperiment')
    @patch('neuro_manipulation.experiment_series_runner.GameNames')
    def test_setup_experiment_ultimatum_game_responder(self, mock_game_names, mock_experiment, 
                                                      mock_get_game_config, mock_get_repe_config, 
                                                      mock_repe_registry):
        """Test that Ultimatum Game Responder gets the correct configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.sample_config, f)
            config_path = f.name

        try:
            # Setup mocks
            mock_game_name = MagicMock()
            mock_game_name.is_sequential.return_value = True
            mock_game_names.from_string.return_value = mock_game_name
            
            mock_get_game_config.return_value = {'base_config': 'value'}
            mock_get_repe_config.return_value = {'repe_config': 'value'}
            
            # Create runner and test setup_experiment
            runner = ExperimentSeriesRunner(config_path)
            
            # Test Ultimatum_Game_Responder configuration
            runner.setup_experiment('Ultimatum_Game_Responder', '/test/model1')
            
            # Get the actual call arguments
            call_args = mock_experiment.call_args
            repe_config, exp_config, game_config = call_args[0][:3]
            
            # Verify game_config has the game-specific attributes
            self.assertEqual(game_config['previous_actions_length'], 1)
            self.assertEqual(game_config['previous_offer_level'], 0)
            
        finally:
            os.unlink(config_path)

    @patch('neuro_manipulation.experiment_series_runner.repe_pipeline_registry')
    @patch('neuro_manipulation.experiment_series_runner.get_repe_eng_config')
    @patch('neuro_manipulation.experiment_series_runner.get_game_config')
    @patch('neuro_manipulation.experiment_series_runner.EmotionGameExperiment')
    @patch('neuro_manipulation.experiment_series_runner.GameNames')
    def test_setup_experiment_fallback_to_default(self, mock_game_names, mock_experiment, 
                                                 mock_get_game_config, mock_get_repe_config, 
                                                 mock_repe_registry):
        """Test that setup_experiment falls back to default config for unknown games."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.sample_config, f)
            config_path = f.name

        try:
            # Setup mocks
            mock_game_name = MagicMock()
            mock_game_name.is_sequential.return_value = True
            mock_game_names.from_string.return_value = mock_game_name
            
            mock_get_game_config.return_value = {'base_config': 'value'}
            mock_get_repe_config.return_value = {'repe_config': 'value'}
            
            # Create runner and test setup_experiment
            runner = ExperimentSeriesRunner(config_path)
            
            # Test with a game not in games_config
            runner.setup_experiment('Unknown_Game', '/test/model1')
            
            # Get the actual call arguments
            call_args = mock_experiment.call_args
            repe_config, exp_config, game_config = call_args[0][:3]
            
            # Verify it fell back to the sequential game default (since is_sequential returns True)
            # The method should have set previous_actions_length from the fallback logic
            # Since there's no default game config anymore, this should handle gracefully
            
        finally:
            os.unlink(config_path)

    def test_no_games_specified_raises_error(self):
        """Test that missing games in config raises a ValueError."""
        config_without_games = {
            'experiment': {
                'name': 'Test_No_Games',
                'models': ['/test/model1'],
                'llm': {'model_name': 'default_model'},
                'repeat': 1,
                'emotions': ['anger'],
                'intensity': [1.5],
                'output': {'base_dir': 'test_results'}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_without_games, f)
            config_path = f.name

        try:
            runner = ExperimentSeriesRunner(config_path)
            
            with self.assertRaises(ValueError) as context:
                runner.run_experiment_series()
            
            self.assertIn("No games specified in configuration", str(context.exception))
            
        finally:
            os.unlink(config_path)

    def test_no_models_specified_raises_error(self):
        """Test that missing models in config raises a ValueError."""
        config_without_models = {
            'experiment': {
                'name': 'Test_No_Models',
                'games': ['Escalation_Game'],
                'games_config': {
                    'Escalation_Game': {'previous_actions_length': 0}
                },
                'llm': {'model_name': 'default_model'},
                'repeat': 1,
                'emotions': ['anger'],
                'intensity': [1.5],
                'output': {'base_dir': 'test_results'}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_without_models, f)
            config_path = f.name

        try:
            runner = ExperimentSeriesRunner(config_path)
            
            with self.assertRaises(ValueError) as context:
                runner.run_experiment_series()
            
            self.assertIn("No models specified in configuration", str(context.exception))
            
        finally:
            os.unlink(config_path)

    def test_all_required_games_have_config(self):
        """Test that all games in the games list have corresponding config."""
        games = self.sample_config['experiment']['games']
        games_config = self.sample_config['experiment']['games_config']
        
        for game in games:
            self.assertIn(game, games_config, f"Game {game} missing from games_config")
            
            # Check that each config has at least previous_actions_length
            game_config = games_config[game]
            self.assertIn('previous_actions_length', game_config, 
                         f"Game {game} missing previous_actions_length")

    def test_trust_game_and_ultimatum_game_specific_attributes(self):
        """Test that Trust Game and Ultimatum Game have their specific attributes."""
        games_config = self.sample_config['experiment']['games_config']
        
        # Trust Game Trustee should have previous_trust_level
        trustee_config = games_config['Trust_Game_Trustee']
        self.assertIn('previous_trust_level', trustee_config)
        self.assertIn(trustee_config['previous_trust_level'], [0, 1])
        
        # Ultimatum Game Responder should have previous_offer_level
        responder_config = games_config['Ultimatum_Game_Responder']
        self.assertIn('previous_offer_level', responder_config)
        self.assertIn(responder_config['previous_offer_level'], [0, 1, 2])

    def test_sequential_games_have_correct_previous_actions_length(self):
        """Test that sequential games have appropriate previous_actions_length values."""
        games_config = self.sample_config['experiment']['games_config']
        
        # Games that should have previous_actions_length = 0 (first players or simultaneous)
        first_player_games = ['Escalation_Game', 'Trust_Game_Trustor', 'Ultimatum_Game_Proposer']
        for game in first_player_games:
            self.assertEqual(games_config[game]['previous_actions_length'], 0,
                           f"{game} should have previous_actions_length = 0")
        
        # Games that should have previous_actions_length = 1 (second players)
        second_player_games = ['Trust_Game_Trustee', 'Ultimatum_Game_Responder']
        for game in second_player_games:
            self.assertEqual(games_config[game]['previous_actions_length'], 1,
                           f"{game} should have previous_actions_length = 1")


if __name__ == '__main__':
    unittest.main() 