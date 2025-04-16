import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_creation.scenario_creation_graph import (
    ScenarioCreationState,
    build_scenario_creation_graph,
    create_scenario,
    finalize_scenario,
    propose_scenario,
    should_continue,
    verify_scenario,
)


class TestScenarioCreationGraph(unittest.TestCase):
    """Test the scenario creation graph components."""

    def setUp(self):
        """Set up test fixtures."""
        self.initial_state: ScenarioCreationState = {
            "game_name": "Prisoners_Dilemma",
            "participants": ["Alice", "Bob"],
            "participant_jobs": ["Software Developer", "Project Manager"],
            "scenario_draft": None,
            "feedback": [],
            "iteration_count": 0,
            "final_scenario": None,
            "converged": False,
        }

        # Example scenario draft for testing
        self.scenario_draft = {
            "scenario": "Software Project Deadline",
            "description": "Two developers are working on a critical project with a tight deadline. Each needs to decide whether to work overtime or maintain regular hours.",
            "participants": [
                {"name": "Alice", "profile": "Software Developer"},
                {"name": "Bob", "profile": "Project Manager"},
            ],
            "behavior_choices": {
                "cooperate": "Maintain Regular Hours",
                "defect": "Work Overtime",
            },
        }

    @patch("scenario_creation_graph.get_game_config")
    @patch("scenario_creation_graph.Game")
    @patch("scenario_creation_graph.get_llm")
    def test_propose_scenario_first_iteration(
        self, mock_get_llm, mock_Game, mock_get_game_config
    ):
        """Test the propose_scenario function for the first iteration."""
        # Setup mocks
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            self.scenario_draft
        )  # Now directly return the dict, not JSON string
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        mock_game = MagicMock()
        mock_game.example_scenario = {"example": "scenario"}
        mock_Game.return_value = mock_game

        mock_get_game_config.return_value = {
            "scenario_class": MagicMock(),
            "decision_class": MagicMock(),
            "payoff_matrix": {},
        }

        # Call the function
        result = propose_scenario(self.initial_state)

        # Check the result
        self.assertEqual(result["iteration_count"], 1)
        self.assertEqual(result["scenario_draft"], self.scenario_draft)
        self.assertEqual(result["feedback"], [])
        mock_llm.invoke.assert_called_once()
        # Check for proper temperature and json_mode
        mock_get_llm.assert_called_once_with(temperature=0.7, json_mode=True)

    @patch("scenario_creation_graph.get_game_config")
    @patch("scenario_creation_graph.Game")
    @patch("scenario_creation_graph.get_llm")
    def test_verify_scenario(self, mock_get_llm, mock_Game, mock_get_game_config):
        """Test the verify_scenario function."""
        # Setup state with draft
        state = {**self.initial_state, "scenario_draft": self.scenario_draft}

        # Setup mocks
        mock_llm = MagicMock()
        mock_response = MagicMock()
        verification_result = {
            "feedback": [
                "Make the scenario more realistic",
                "Add more details about the project",
            ],
            "converged": False,
        }
        mock_response.content = (
            verification_result  # Now directly return the dict, not JSON string
        )
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        mock_game = MagicMock()
        mock_game.example_scenario = {"example": "scenario"}
        mock_Game.return_value = mock_game

        mock_get_game_config.return_value = {
            "scenario_class": MagicMock(),
            "decision_class": MagicMock(),
            "payoff_matrix": {},
        }

        # Call the function
        result = verify_scenario(state)

        # Check the result
        self.assertEqual(result["feedback"], verification_result["feedback"])
        self.assertEqual(result["converged"], verification_result["converged"])
        mock_llm.invoke.assert_called_once()
        # Check for proper temperature and json_mode
        mock_get_llm.assert_called_once_with(temperature=0.3, json_mode=True)

    def test_should_continue_not_converged(self):
        """Test the should_continue function when not converged and under max iterations."""
        state = {**self.initial_state, "converged": False, "iteration_count": 2}
        result = should_continue(state)
        self.assertEqual(result, "refine")

    def test_should_continue_converged(self):
        """Test the should_continue function when converged."""
        state = {**self.initial_state, "converged": True, "iteration_count": 2}
        result = should_continue(state)
        self.assertEqual(result, "finalize")

    def test_should_continue_max_iterations(self):
        """Test the should_continue function when max iterations reached."""
        state = {**self.initial_state, "converged": False, "iteration_count": 5}
        result = should_continue(state)
        self.assertEqual(result, "finalize")

    @patch("scenario_creation_graph.Path")
    @patch("builtins.open", new_callable=unittest.mock.mock_open())
    @patch("json.dump")
    @patch("builtins.__import__")
    def test_finalize_scenario(self, mock_import, mock_json_dump, mock_open, mock_Path):
        """Test the finalize_scenario function."""
        # Setup
        state = {**self.initial_state, "scenario_draft": self.scenario_draft}

        # Mock datetime
        mock_datetime = MagicMock()
        mock_datetime.datetime.now.return_value.strftime.return_value = (
            "20230101_120000"
        )
        mock_import.return_value = mock_datetime

        # Mock Path
        mock_path_instance = MagicMock()
        mock_Path.return_value = mock_path_instance
        mock_path_instance.__truediv__.return_value = (
            "fake/path/Prisoners_Dilemma_20230101_120000.json"
        )

        # Call the function
        result = finalize_scenario(state)

        # Check the result
        self.assertEqual(result["final_scenario"], self.scenario_draft)

        # Check that json.dump was called with the scenario_draft and any file-like object
        mock_json_dump.assert_called_once()
        args, kwargs = mock_json_dump.call_args
        self.assertEqual(
            args[0], self.scenario_draft
        )  # First arg should be scenario_draft
        self.assertEqual(kwargs.get("indent"), 2)  # Check indent was set to 2


if __name__ == "__main__":
    unittest.main()
