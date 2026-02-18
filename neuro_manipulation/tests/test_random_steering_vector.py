"""
Tests for neuro_manipulation/utils.py and configs/experiment_config.py

Verifies that direction_finder_kwargs flows from YAML repe_eng_config
through get_repe_eng_config -> load_emotion_readers -> get_rep_reader
-> rep_reading_pipeline.get_directions, enabling RandomRepReader with
needs_hiddens=False to skip the expensive HF forward pass entirely.
"""

import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np


class TestDirectionFinderKwargsFlow(unittest.TestCase):
    """Test that direction_finder_kwargs propagates from YAML config to get_directions."""

    def test_get_repe_eng_config_includes_direction_finder_kwargs(self):
        """get_repe_eng_config must preserve direction_finder_kwargs from yaml_config."""
        from neuro_manipulation.configs.experiment_config import get_repe_eng_config

        yaml_config = {
            "direction_method": "random",
            "direction_finder_kwargs": {"needs_hiddens": False},
        }
        cfg = get_repe_eng_config("some/model", yaml_config=yaml_config)

        self.assertEqual(cfg["direction_method"], "random")
        self.assertEqual(cfg["direction_finder_kwargs"], {"needs_hiddens": False})

    def test_get_repe_eng_config_default_has_no_direction_finder_kwargs(self):
        """By default, direction_finder_kwargs should not be in base config (not needed for pca)."""
        from neuro_manipulation.configs.experiment_config import get_repe_eng_config

        cfg = get_repe_eng_config("some/model")
        # pca is default and needs no extra kwargs — key should be absent or empty
        self.assertNotIn("direction_finder_kwargs", cfg)

    def test_get_rep_reader_passes_direction_finder_kwargs(self):
        """get_rep_reader must forward direction_finder_kwargs to pipeline.get_directions."""
        from neuro_manipulation.utils import get_rep_reader

        mock_pipeline = MagicMock()
        mock_reader = MagicMock()
        mock_reader.direction_signs = {-1: 1}
        mock_reader.directions = {-1: np.zeros((1, 4))}
        mock_pipeline.get_directions.return_value = mock_reader

        # Stub test_direction to avoid needing real hidden states
        with patch("neuro_manipulation.utils.test_direction", return_value=({-1: 1.0}, {})):
            get_rep_reader(
                rep_reading_pipeline=mock_pipeline,
                train_data={"data": ["a", "b"], "labels": [[1, 0], [0, 1]]},
                test_data={"data": ["c"], "labels": [[1, 0]]},
                hidden_layers=[-1],
                rep_token=-1,
                n_difference=1,
                direction_method="random",
                direction_finder_kwargs={"needs_hiddens": False},
            )

        mock_pipeline.get_directions.assert_called_once()
        _, kwargs = mock_pipeline.get_directions.call_args
        self.assertEqual(kwargs.get("direction_finder_kwargs"), {"needs_hiddens": False})

    def test_random_rep_reader_with_needs_hiddens_false_skips_forward_pass(self):
        """RandomRepReader(needs_hiddens=False) must not trigger hidden state extraction."""
        from neuro_manipulation.repe.rep_readers import RandomRepReader, DIRECTION_FINDERS

        # Confirm "random" is registered
        self.assertIn("random", DIRECTION_FINDERS)

        reader = RandomRepReader(needs_hiddens=False)
        self.assertFalse(reader.needs_hiddens)

        # get_rep_directions should work with None hidden_states (no forward pass needed)
        mock_model = MagicMock()
        mock_model.config.hidden_size = 16
        directions = reader.get_rep_directions(
            model=mock_model,
            tokenizer=None,
            hidden_states=None,  # No hidden states needed
            hidden_layers=[-1, -2],
        )

        self.assertIn(-1, directions)
        self.assertIn(-2, directions)
        # Each direction should be shape (1, hidden_size)
        self.assertEqual(directions[-1].shape, (1, 16))
        self.assertEqual(directions[-2].shape, (1, 16))


class TestRandomRepReaderNormality(unittest.TestCase):
    """Verify the generated random vector has correct statistical properties."""

    def test_random_vector_is_standard_normal(self):
        """np.random.randn produces zero-mean, unit-variance vectors in expectation."""
        from neuro_manipulation.repe.rep_readers import RandomRepReader

        np.random.seed(42)
        reader = RandomRepReader(needs_hiddens=False)
        mock_model = MagicMock()
        mock_model.config.hidden_size = 4096

        directions = reader.get_rep_directions(
            model=mock_model, tokenizer=None, hidden_states=None, hidden_layers=[-1]
        )
        vec = directions[-1][0]  # shape (4096,)

        # For d=4096, mean should be near 0, std near 1
        self.assertAlmostEqual(float(np.mean(vec)), 0.0, delta=0.1)
        self.assertAlmostEqual(float(np.std(vec)), 1.0, delta=0.1)

    def test_different_layers_get_independent_vectors(self):
        """Each layer must get its own independently sampled direction."""
        from neuro_manipulation.repe.rep_readers import RandomRepReader

        reader = RandomRepReader(needs_hiddens=False)
        mock_model = MagicMock()
        mock_model.config.hidden_size = 64

        directions = reader.get_rep_directions(
            model=mock_model, tokenizer=None, hidden_states=None, hidden_layers=[-1, -2, -3]
        )
        # Vectors for different layers must not be identical
        self.assertFalse(np.allclose(directions[-1], directions[-2]))
        self.assertFalse(np.allclose(directions[-2], directions[-3]))


if __name__ == "__main__":
    unittest.main()
