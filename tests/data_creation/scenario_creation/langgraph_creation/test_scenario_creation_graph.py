import json
import unittest

# Assume ScenarioCreationState is importable or define a mock version
# from data_creation.scenario_creation.langgraph_creation.scenario_creation_graph import (
#     ScenarioCreationState, should_continue, verify_narrative, verify_preference_order, verify_pay_off, aggregate_verification
# )
# Mocking the state type hint for standalone testing
from typing import Any, Dict, List, Optional, TypedDict
from unittest.mock import MagicMock, patch

# Import the constant at the top level
from data_creation.scenario_creation.langgraph_creation.scenario_creation_graph import (
    PAYOFF_VALIDATION_QUESTION_FORMAT,
)
from games.game import Game
from games.game_configs import get_game_config


class ScenarioCreationState(TypedDict):
    game_name: str
    participants: List[str]
    participant_jobs: Optional[List[str]]
    scenario_draft: Optional[Dict[str, Any]]
    narrative_feedback: Optional[List[str]]
    preference_feedback: Optional[List[str]]  # Renamed from mechanics
    payoff_feedback: Optional[List[str]]
    iteration_count: int
    final_scenario: Optional[Dict[str, Any]]
    narrative_converged: bool
    preference_converged: bool  # Renamed from mechanics
    payoff_converged: bool
    auto_save_path: Optional[str]


# Mock functions need to be defined here if not importing the real ones
def should_continue(state: ScenarioCreationState) -> str:
    narrative_converged = state.get("narrative_converged", False)
    preference_converged = state.get(
        "preference_converged", False
    )  # Renamed from mechanics
    payoff_converged = state.get("payoff_converged", False)
    iteration_count = state["iteration_count"]
    all_converged = narrative_converged and preference_converged and payoff_converged
    if all_converged or iteration_count >= 5:
        return "finalize"
    else:
        return "refine"


def aggregate_verification(state: ScenarioCreationState) -> ScenarioCreationState:
    """Mock implementation for testing."""
    return state


# --- Test Class --- #


class TestScenarioCreationGraphNodes(unittest.TestCase):

    def test_should_continue_logic(self):
        """Tests the logic of the should_continue conditional edge."""

        # Case 1: All converged
        state_all_converged = ScenarioCreationState(
            iteration_count=1,
            narrative_converged=True,
            preference_converged=True,  # Renamed
            payoff_converged=True,
        )
        self.assertEqual(should_continue(state_all_converged), "finalize")

        # Case 2: Payoff not converged
        state_payoff_fail = ScenarioCreationState(
            iteration_count=1,
            narrative_converged=True,
            preference_converged=True,  # Renamed
            payoff_converged=False,
        )
        self.assertEqual(should_continue(state_payoff_fail), "refine")

        # Case 3: Preference not converged
        state_preference_fail = ScenarioCreationState(
            iteration_count=1,
            narrative_converged=True,
            preference_converged=False,  # Renamed
            payoff_converged=True,
        )
        self.assertEqual(should_continue(state_preference_fail), "refine")

        # Case 4: Narrative not converged
        state_narrative_fail = ScenarioCreationState(
            iteration_count=1,
            narrative_converged=False,
            preference_converged=True,  # Renamed
            payoff_converged=True,
        )
        self.assertEqual(should_continue(state_narrative_fail), "refine")

        # Case 5: None converged, low iteration count
        state_none_converged = ScenarioCreationState(
            iteration_count=1,
            narrative_converged=False,
            preference_converged=False,  # Renamed
            payoff_converged=False,
        )
        self.assertEqual(should_continue(state_none_converged), "refine")

        # Case 6: None converged, max iterations reached
        state_max_iterations = ScenarioCreationState(
            iteration_count=5,
            narrative_converged=False,
            preference_converged=False,  # Renamed
            payoff_converged=False,
        )
        self.assertEqual(should_continue(state_max_iterations), "finalize")

        # Case 7: All converged, max iterations (should still finalize)
        state_all_converged_max_iter = ScenarioCreationState(
            iteration_count=5,
            narrative_converged=True,
            preference_converged=True,  # Renamed
            payoff_converged=True,
        )
        self.assertEqual(should_continue(state_all_converged_max_iter), "finalize")

    # --- Mocked Tests for Verification Nodes --- #

    # Helper to create a mock game config
    def _create_mock_game_config(self):
        mock_game_config_instance = MagicMock()
        mock_payoff_matrix = MagicMock()
        mock_leaf_1 = MagicMock()
        mock_leaf_1.actions = ["ActionA", "ActionB"]
        mock_leaf_2 = MagicMock()
        mock_leaf_2.actions = ["ActionC", "ActionD"]
        mock_payoff_matrix.payoff_leaves = [mock_leaf_1, mock_leaf_2]
        mock_payoff_matrix.ordered_payoff_leaves = [
            ["A", "B"],
            ["C", "D"],
        ]  # Example structure
        mock_game_config_instance.payoff_matrix = mock_payoff_matrix
        return {
            "scenario_class": MagicMock(),
            "decision_class": MagicMock(),
            "payoff_matrix": mock_payoff_matrix,
        }

    # Helper to create a base initial state
    def _get_base_initial_state(self):
        return ScenarioCreationState(
            game_name="TestGame",
            participants=["P1", "P2"],
            participant_jobs=None,
            scenario_draft={
                "scenario": "Test Scenario",
                "description": "Test scenario description.",
                "behavior_choices": {
                    "ActionA": "Do X",
                    "ActionB": "Do Y",
                    "ActionC": "Do Z",
                    "ActionD": "Do W",
                },
                "payoff_matrix": {
                    "P1: ActionA , P2: ActionB": [
                        "Outcome 1 happens",
                        "Outcome 1 happens",
                    ],
                    "P1: ActionC , P2: ActionD": [
                        "Outcome 2 happens",
                        "Outcome 2 happens",
                    ],
                },
            },
            narrative_feedback=[],
            preference_feedback=[],  # Renamed
            payoff_feedback=[],
            iteration_count=1,
            final_scenario=None,
            narrative_converged=False,
            preference_converged=False,  # Renamed
            payoff_converged=False,
            auto_save_path=None,
        )

    @patch(
        "data_creation.scenario_creation.langgraph_creation.scenario_creation_graph.get_llm"
    )
    def test_verify_narrative_updates_state_success(self, mock_get_llm):
        mock_llm_instance = MagicMock()
        mock_response_content = json.dumps(
            {"feedback": ["Narrative looks good"], "converged": True}
        )
        mock_llm_instance.invoke.return_value = MagicMock(content=mock_response_content)
        mock_get_llm.return_value = mock_llm_instance
        initial_state = self._get_base_initial_state()
        # Use the actual function for testing
        from data_creation.scenario_creation.langgraph_creation.scenario_creation_graph import (
            verify_narrative,
        )

        updated_state = verify_narrative(initial_state)
        mock_get_llm.assert_called_once_with(temperature=0.3, json_mode=True)
        self.assertEqual(updated_state["narrative_feedback"], ["Narrative looks good"])
        self.assertTrue(updated_state["narrative_converged"])

    @patch(
        "data_creation.scenario_creation.langgraph_creation.scenario_creation_graph.get_llm"
    )
    def test_verify_narrative_handles_bad_json(self, mock_get_llm):
        mock_llm_instance = MagicMock()
        mock_response_content = "This is not JSON"
        mock_llm_instance.invoke.return_value = MagicMock(content=mock_response_content)
        mock_get_llm.return_value = mock_llm_instance
        initial_state = self._get_base_initial_state()
        from data_creation.scenario_creation.langgraph_creation.scenario_creation_graph import (
            verify_narrative,
        )

        updated_state = verify_narrative(initial_state)
        self.assertIn(
            "Error parsing narrative verification result",
            updated_state["narrative_feedback"],
        )
        self.assertFalse(updated_state["narrative_converged"])

    @unittest.skip("Skipping preference order verification test")
    @patch(
        "data_creation.scenario_creation.langgraph_creation.scenario_creation_graph.get_llm"
    )
    @patch(
        "data_creation.scenario_creation.langgraph_creation.scenario_creation_graph.get_game_config"
    )
    def test_verify_preference_order_updates_state_success(
        self, mock_get_game_config, mock_get_llm
    ):
        """Tests if verify_preference_order updates state correctly on mocked success."""
        # Arrange
        mock_llm_instance = MagicMock()
        mock_response_content = json.dumps(
            {"feedback": ["Preference OK"], "converged": True}
        )
        mock_llm_instance.invoke.return_value = MagicMock(content=mock_response_content)
        mock_get_llm.return_value = mock_llm_instance
        mock_get_game_config.return_value = self._create_mock_game_config()
        initial_state = self._get_base_initial_state()
        # Use the actual function for testing
        from data_creation.scenario_creation.langgraph_creation.scenario_creation_graph import (
            verify_preference_order,
        )

        # Act
        updated_state = verify_preference_order(initial_state)

        # Assert
        mock_get_llm.assert_called_once_with(temperature=0.2, json_mode=True)
        mock_llm_instance.invoke.assert_called_once()
        self.assertEqual(updated_state["preference_feedback"], ["Preference OK"])
        self.assertTrue(updated_state["preference_converged"])
        # Remove assertions for keys not returned by verify_preference_order
        # self.assertFalse(
        #     updated_state["narrative_converged"]
        # )  # Narrative remains false
        # self.assertFalse(updated_state["payoff_converged"])  # Payoff still false

    @patch(
        "data_creation.scenario_creation.langgraph_creation.scenario_creation_graph.get_llm"
    )
    @patch(
        "data_creation.scenario_creation.langgraph_creation.scenario_creation_graph.get_game_config"
    )
    def test_verify_pay_off_updates_state_success(
        self, mock_get_game_config, mock_get_llm
    ):
        """Tests if verify_pay_off updates state correctly on mocked success."""
        # Arrange
        mock_llm_instance = MagicMock()
        mock_response_content = json.dumps(
            {"feedback": ["YES, plausible.", "YES, plausible."], "converged": True}
        )
        mock_llm_instance.invoke.return_value = MagicMock(content=mock_response_content)
        mock_get_llm.return_value = mock_llm_instance
        mock_get_game_config.return_value = self._create_mock_game_config()
        initial_state = self._get_base_initial_state()
        # Use the actual function for testing
        from data_creation.scenario_creation.langgraph_creation.scenario_creation_graph import (
            verify_pay_off,
        )

        # Act
        updated_state = verify_pay_off(initial_state)

        # Assert
        mock_get_game_config.assert_called_once_with("TestGame")
        mock_get_llm.assert_called_once_with(temperature=0.1, json_mode=True)
        mock_llm_instance.invoke.assert_called_once()

        # --- Add detailed prompt verification --- #
        # Capture the arguments passed to invoke
        invoke_args, _ = mock_llm_instance.invoke.call_args
        # The first argument is the list of messages
        messages = invoke_args[0]
        # Find the HumanMessage
        human_message = next((msg for msg in messages if msg.type == "human"), None)
        self.assertIsNotNone(human_message, "HumanMessage not found in LLM call")
        human_message_content = human_message.content

        # Verify the generated questions within the prompt
        # Define the expected values for the format placeholders
        expected_behavior_1 = (
            "Participant 1 (P1) chooses 'Do X' and Participant 2 (P2) chooses 'Do Y'"
        )
        expected_p1_outcome_1 = "Outcome 1 happens"
        expected_p2_outcome_1 = "Outcome 1 happens"

        expected_behavior_2 = (
            "Participant 1 (P1) chooses 'Do Z' and Participant 2 (P2) chooses 'Do W'"
        )
        expected_p1_outcome_2 = "Outcome 2 happens"
        expected_p2_outcome_2 = "Outcome 2 happens"

        # Use the imported constant to format the expected questions
        expected_question_1 = PAYOFF_VALIDATION_QUESTION_FORMAT.format(
            behavior_description=expected_behavior_1,
            p1_outcome=expected_p1_outcome_1,
            p2_outcome=expected_p2_outcome_1,
        )
        expected_question_2 = PAYOFF_VALIDATION_QUESTION_FORMAT.format(
            behavior_description=expected_behavior_2,
            p1_outcome=expected_p1_outcome_2,
            p2_outcome=expected_p2_outcome_2,
        )

        # Check that the prompt contains the correctly formatted questions
        self.assertIn(expected_question_1, human_message_content)
        self.assertIn(expected_question_2, human_message_content)
        # --- End of detailed prompt verification --- #

        # Check state update
        self.assertEqual(
            updated_state["payoff_feedback"], ["YES, plausible.", "YES, plausible."]
        )
        self.assertTrue(updated_state["payoff_converged"])
        # Remove assertions for keys not returned by verify_pay_off
        # self.assertFalse(updated_state["narrative_converged"])
        # self.assertFalse(updated_state["preference_converged"])

    # REMOVED: test_pay_off_validation_skips_if_mechanics_failed (no longer skips)

    @patch(
        "data_creation.scenario_creation.langgraph_creation.scenario_creation_graph.get_llm"
    )
    @patch(
        "data_creation.scenario_creation.langgraph_creation.scenario_creation_graph.get_game_config"
    )
    def test_verify_pay_off_handles_missing_keys_in_draft(
        self, mock_get_game_config, mock_get_llm
    ):
        """Tests verify_pay_off handling of incomplete scenario_draft."""
        # Arrange
        initial_state = self._get_base_initial_state()
        del initial_state["scenario_draft"]["behavior_choices"]
        mock_get_game_config.return_value = self._create_mock_game_config()
        from data_creation.scenario_creation.langgraph_creation.scenario_creation_graph import (
            verify_pay_off,
        )

        # Act
        updated_state = verify_pay_off(initial_state)

        # Assert
        mock_get_llm.assert_not_called()  # LLM should not be called if keys missing
        self.assertIn("missing required keys", updated_state["payoff_feedback"][0])
        self.assertFalse(updated_state["payoff_converged"])

    def test_aggregate_verification_node(self):
        """Tests the aggregation node simply passes state through."""
        initial_state = self._get_base_initial_state()
        # Simulate state after parallel execution
        initial_state["narrative_converged"] = True
        initial_state["preference_converged"] = False
        initial_state["payoff_converged"] = True
        initial_state["narrative_feedback"] = ["Narrative feedback"]
        initial_state["preference_feedback"] = ["Preference feedback"]
        initial_state["payoff_feedback"] = ["Payoff feedback"]

        from data_creation.scenario_creation.langgraph_creation.scenario_creation_graph import (
            aggregate_verification,
        )

        updated_state = aggregate_verification(initial_state)

        # Check that the state contains the 'all_converged' key and its value is correct
        self.assertIn("all_converged", updated_state)
        # Based on initial_state: narrative=True, preference=False, payoff=True => all_converged=False
        self.assertFalse(updated_state["all_converged"])

        # Test case where all should converge
        initial_state["preference_converged"] = True  # Make all true
        updated_state_all_true = aggregate_verification(initial_state)
        self.assertTrue(updated_state_all_true["all_converged"])

        # Check that the returned dict ONLY contains 'all_converged'
        self.assertEqual(len(updated_state), 1)
        self.assertEqual(len(updated_state_all_true), 1)


if __name__ == "__main__":
    # Adjust imports for direct execution
    import os
    import sys

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../../..")
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # from data_creation.scenario_creation.langgraph_creation.scenario_creation_graph import (
    #     verify_preference_order,  # Renamed
    # )
    from data_creation.scenario_creation.langgraph_creation.scenario_creation_graph import (
        aggregate_verification,  # Added
    )
    from data_creation.scenario_creation.langgraph_creation.scenario_creation_graph import (
        verify_pay_off,  # Renamed
    )
    from data_creation.scenario_creation.langgraph_creation.scenario_creation_graph import (
        ScenarioCreationState,
        should_continue,
        verify_narrative,
    )

    unittest.main()
