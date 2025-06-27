import asyncio
import json
import queue
import sys
import threading
import time
import unittest

# Assume ScenarioCreationState is importable or define a mock version
# from data_creation.scenario_creation.langgraph_creation.scenario_creation_graph import (
#     ScenarioCreationState, should_continue, verify_narrative, verify_preference_order, verify_pay_off, aggregate_verification
# )
# Mocking the state type hint for standalone testing
from typing import Any, Dict, List, Optional, TypedDict
from unittest.mock import AsyncMock, MagicMock, Mock, patch

# Import the constant at the top level
from data_creation.scenario_creation.langgraph_creation.scenario_creation_graph import (
    PAYOFF_VALIDATION_QUESTION_FORMAT,
    build_scenario_creation_graph,
    create_scenario,
    finalize_scenario,
    get_llm,
    propose_scenario,
    verify_pay_off,
    verify_preference_order,
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


# --- Comprehensive Bug Detection Tests --- #


class TestBugDetectionAndRobustness(unittest.TestCase):
    """Comprehensive tests to catch bugs that could cause timeouts, infinite loops, and other issues."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_state = {
            "game_name": "Escalation_Game",
            "participants": ["You", "Bob"],
            "participant_jobs": ["lawyer", "lawyer"],
            "scenario_draft": {
                "scenario": "Legal Dispute",
                "description": "A legal dispute scenario.",
                "behavior_choices": {
                    "withdraw": "Withdraw the case",
                    "escalate": "File additional motions",
                },
                "payoff_matrix_description": {
                    "You: withdraw , Bob: withdraw": ["Low cost", "Low cost"],
                    "You: escalate , Bob: withdraw": ["Win case", "Lose case"],
                    "You: escalate , Bob: escalate": ["High cost", "High cost"],
                },
            },
            "narrative_feedback": [],
            "preference_feedback": [],
            "payoff_feedback": [],
            "iteration_count": 0,
            "final_scenario": None,
            "narrative_converged": False,
            "preference_converged": False,
            "payoff_converged": False,
            "all_converged": None,
            "auto_save_path": None,
        }

    def test_bug_1_aggregate_verification_missing_scenario_key(self):
        """Test Bug 1: KeyError when scenario_draft missing 'scenario' key."""
        from data_creation.scenario_creation.langgraph_creation.scenario_creation_graph import (
            aggregate_verification,
        )

        state = self.valid_state.copy()
        # Remove 'scenario' key to trigger the bug
        state["scenario_draft"] = {"description": "Test", "behavior_choices": {}}

        # This should not crash
        result = aggregate_verification(state)
        self.assertIn("all_converged", result)
        self.assertIsInstance(result["all_converged"], bool)

    def test_bug_1_aggregate_verification_none_scenario_draft(self):
        """Test Bug 1: KeyError when scenario_draft is None."""
        from data_creation.scenario_creation.langgraph_creation.scenario_creation_graph import (
            aggregate_verification,
        )

        state = self.valid_state.copy()
        state["scenario_draft"] = None

        # This should not crash
        result = aggregate_verification(state)
        self.assertIn("all_converged", result)
        self.assertIsInstance(result["all_converged"], bool)

    def test_bug_3_convergence_logic_robustness(self):
        """Test Bug 3: Incorrect convergence logic with None and missing values."""
        from data_creation.scenario_creation.langgraph_creation.scenario_creation_graph import (
            aggregate_verification,
        )

        test_cases = [
            # Normal case - all converged
            {
                "narrative_converged": True,
                "preference_converged": True,
                "payoff_converged": True,
                "expected": True,
            },
            # Normal case - none converged
            {
                "narrative_converged": False,
                "preference_converged": False,
                "payoff_converged": False,
                "expected": False,
            },
            # Edge case - missing flags
            {
                "narrative_converged": True,
                "preference_converged": True,
                "expected": False,
            },
            # Edge case - None values
            {
                "narrative_converged": None,
                "preference_converged": True,
                "payoff_converged": True,
                "expected": False,
            },
        ]

        for i, test_case in enumerate(test_cases):
            with self.subTest(case=i):
                state = self.valid_state.copy()
                # Remove all convergence flags first
                for key in [
                    "narrative_converged",
                    "preference_converged",
                    "payoff_converged",
                ]:
                    if key in state:
                        del state[key]

                # Add back only the specified flags
                for key, value in test_case.items():
                    if key != "expected":
                        state[key] = value

                result = aggregate_verification(state)
                self.assertEqual(result["all_converged"], test_case["expected"])

    def test_bug_4_infinite_loop_protection_max_iterations(self):
        """Test Bug 4: Infinite loops prevented by max iteration limit."""
        from data_creation.scenario_creation.langgraph_creation.scenario_creation_graph import (
            should_continue,
        )

        state = self.valid_state.copy()
        state["iteration_count"] = 8  # At the max limit
        state["all_converged"] = False  # Never converges

        result = should_continue(state)
        self.assertEqual(result, "finalize", "Should finalize at max iterations")

    @patch(
        "data_creation.scenario_creation.langgraph_creation.scenario_creation_graph.get_llm"
    )
    def test_propose_scenario_json_parsing_error(self, mock_get_llm):
        """Test that propose_scenario handles invalid JSON from LLM."""
        # Mock LLM to return invalid JSON
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "This is not valid JSON {incomplete"
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        state = self.valid_state.copy()

        result = propose_scenario(state)

        # Should handle JSON parsing error gracefully
        self.assertIn("error", result["scenario_draft"])
        self.assertEqual(
            result["scenario_draft"]["error"], "Failed to parse JSON from response"
        )

    @patch(
        "data_creation.scenario_creation.langgraph_creation.scenario_creation_graph.get_llm"
    )
    def test_propose_scenario_llm_timeout_handling(self, mock_get_llm):
        """Test that propose_scenario handles LLM timeouts gracefully."""
        # Mock LLM to simulate timeout
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = asyncio.TimeoutError("LLM timeout")
        mock_get_llm.return_value = mock_llm

        state = self.valid_state.copy()

        # This should raise an exception but not hang
        with self.assertRaises(asyncio.TimeoutError):
            propose_scenario(state)

    @patch(
        "data_creation.scenario_creation.langgraph_creation.scenario_creation_graph.get_llm"
    )
    def test_verify_narrative_error_in_scenario_draft(self, mock_get_llm):
        """Test verify_narrative handles error in scenario draft."""
        from data_creation.scenario_creation.langgraph_creation.scenario_creation_graph import (
            verify_narrative,
        )

        state = self.valid_state.copy()
        state["scenario_draft"] = {
            "error": "Previous step failed",
            "raw_content": "Some error content",
        }

        result = verify_narrative(state)

        self.assertIn(
            "Cannot verify narrative due to error", result["narrative_feedback"][0]
        )
        self.assertFalse(result["narrative_converged"])

    @patch(
        "data_creation.scenario_creation.langgraph_creation.scenario_creation_graph.get_game_config"
    )
    def test_verify_pay_off_missing_payoff_leaves(self, mock_get_game_config):
        """Test verify_pay_off handles missing payoff_leaves gracefully."""
        # Mock game config with missing payoff_leaves
        mock_game = MagicMock()
        mock_payoff_matrix = MagicMock()
        mock_payoff_matrix.payoff_leaves = None  # Missing payoff_leaves
        mock_game.payoff_matrix = mock_payoff_matrix

        mock_game_config = {
            "scenario_class": MagicMock(),
            "decision_class": MagicMock(),
            "payoff_matrix": mock_payoff_matrix,
        }
        mock_get_game_config.return_value = mock_game_config

        state = self.valid_state.copy()

        result = verify_pay_off(state)

        self.assertIn("payoff_leaves", result["payoff_feedback"][0])
        self.assertFalse(result["payoff_converged"])

    @patch(
        "data_creation.scenario_creation.langgraph_creation.scenario_creation_graph.get_game_config"
    )
    def test_verify_pay_off_game_config_loading_error(self, mock_get_game_config):
        """Test verify_pay_off handles game config loading errors."""
        # Mock game config to raise exception
        mock_get_game_config.side_effect = Exception("Failed to load game config")

        state = self.valid_state.copy()

        result = verify_pay_off(state)

        self.assertIn("Error loading game config", result["payoff_feedback"][0])
        self.assertFalse(result["payoff_converged"])

    def test_verify_pay_off_missing_behavior_choices(self):
        """Test verify_pay_off handles missing behavior_choices in scenario draft."""
        state = self.valid_state.copy()
        # Remove behavior_choices to simulate missing key
        del state["scenario_draft"]["behavior_choices"]

        result = verify_pay_off(state)

        self.assertIn("missing required keys", result["payoff_feedback"][0])
        self.assertFalse(result["payoff_converged"])

    def test_finalize_scenario_with_convergence(self):
        """Test finalize_scenario when all converged."""
        state = self.valid_state.copy()
        state["all_converged"] = True

        result = finalize_scenario(state)

        self.assertIsNotNone(result["final_scenario"])
        self.assertEqual(result["final_scenario"], state["scenario_draft"])

    def test_finalize_scenario_without_convergence(self):
        """Test finalize_scenario when not converged."""
        state = self.valid_state.copy()
        state["all_converged"] = False

        result = finalize_scenario(state)

        self.assertIsNone(result["final_scenario"])

    def test_performance_large_state(self):
        """Test performance with large state objects."""
        from data_creation.scenario_creation.langgraph_creation.scenario_creation_graph import (
            aggregate_verification,
        )

        large_state = self.valid_state.copy()
        large_state["narrative_feedback"] = [f"feedback_{i}" for i in range(1000)]
        large_state["preference_feedback"] = [f"feedback_{i}" for i in range(1000)]
        large_state["payoff_feedback"] = [f"feedback_{i}" for i in range(1000)]

        start_time = time.time()
        result = aggregate_verification(large_state)
        end_time = time.time()

        processing_time = end_time - start_time
        self.assertLess(
            processing_time,
            1.0,
            f"Large state processing took too long: {processing_time} seconds",
        )
        self.assertIn("all_converged", result)

    def test_timeout_simulation(self):
        """Test that operations complete within reasonable time."""
        from data_creation.scenario_creation.langgraph_creation.scenario_creation_graph import (
            aggregate_verification,
            should_continue,
        )

        state = self.valid_state.copy()

        # Test basic operations
        start_time = time.time()

        # Test should_continue
        should_continue(state)

        # Test aggregate_verification
        aggregate_verification(state)

        # Test finalize_scenario
        finalize_scenario(state)

        end_time = time.time()
        total_time = end_time - start_time

        self.assertLess(
            total_time, 5.0, f"Basic operations took too long: {total_time} seconds"
        )

    def test_memory_usage_with_large_feedback(self):
        """Test memory usage doesn't grow excessively with large feedback arrays."""
        from data_creation.scenario_creation.langgraph_creation.scenario_creation_graph import (
            aggregate_verification,
        )

        state = self.valid_state.copy()

        # Create large feedback arrays
        large_feedback = [f"feedback_{i}" * 100 for i in range(100)]  # Large strings
        state["narrative_feedback"] = large_feedback
        state["preference_feedback"] = large_feedback
        state["payoff_feedback"] = large_feedback

        # Measure memory before
        initial_size = sys.getsizeof(state)

        # Run aggregation
        result = aggregate_verification(state)

        # Result should be small
        result_size = sys.getsizeof(result)
        self.assertLess(
            result_size, initial_size * 0.5, "Result should be much smaller than input"
        )

    def test_thread_safety_simulation(self):
        """Simulate potential thread safety issues."""
        from data_creation.scenario_creation.langgraph_creation.scenario_creation_graph import (
            aggregate_verification,
        )

        results = queue.Queue()

        def worker():
            state = self.valid_state.copy()
            state["narrative_converged"] = True
            state["preference_converged"] = True
            state["payoff_converged"] = True
            result = aggregate_verification(state)
            results.put(result)

        # Run multiple workers concurrently
        threads = []
        for i in range(10):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        # Wait for all to complete
        for t in threads:
            t.join()

        # All results should be consistent
        all_results = []
        while not results.empty():
            all_results.append(results.get())

        self.assertEqual(len(all_results), 10)
        for result in all_results:
            self.assertTrue(result["all_converged"])

    def test_edge_case_none_values_in_state(self):
        """Test handling of None values in critical state fields."""
        from data_creation.scenario_creation.langgraph_creation.scenario_creation_graph import (
            should_continue,
        )

        state = {
            "game_name": "Test_Game",
            "participants": ["Alice", "Bob"],
            "participant_jobs": None,  # This should be handled gracefully
            "scenario_draft": None,
            "narrative_feedback": None,  # This could cause issues
            "preference_feedback": None,
            "payoff_feedback": None,
            "iteration_count": 1,
            "final_scenario": None,
            "narrative_converged": None,  # This could cause issues in should_continue
            "preference_converged": None,
            "payoff_converged": None,
            "all_converged": None,
            "auto_save_path": None,
        }

        # Test should_continue with None values
        result = should_continue(state)
        self.assertIn(result, ["finalize", "refine"])  # Should not crash

    def test_iteration_count_increments_correctly(self):
        """Test that iteration count increments properly to prevent infinite loops."""
        with patch(
            "data_creation.scenario_creation.langgraph_creation.scenario_creation_graph.get_llm"
        ) as mock_get_llm:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = json.dumps(
                {
                    "scenario": "Test",
                    "description": "Test",
                    "behavior_choices": {},
                    "payoff_matrix_description": {},
                }
            )
            mock_llm.invoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            state = self.valid_state.copy()
            state["iteration_count"] = 5

            result = propose_scenario(state)

            # Should increment iteration count
            self.assertEqual(result["iteration_count"], 6)

    def test_llm_response_parsing_performance(self):
        """Test that LLM response parsing is fast enough."""
        # Create a large mock response
        large_response = {
            "feedback": [f"feedback_{i}" for i in range(1000)],
            "converged": False,
        }
        response_json = json.dumps(large_response)

        start_time = time.time()
        # Simulate parsing like the actual functions do
        parsed_response = json.loads(response_json)
        processing_result = len(parsed_response["feedback"])
        end_time = time.time()

        parsing_time = end_time - start_time
        self.assertLess(
            parsing_time, 0.1, f"Response parsing took too long: {parsing_time} seconds"
        )
        self.assertEqual(processing_result, 1000)

    def test_debug_mode_functionality(self):
        """Test that debug_mode correctly handles format errors."""
        from data_creation.scenario_creation.langgraph_creation.scenario_creation_graph import (
            verify_pay_off,
        )

        # Create a test state that will trigger a format error (missing payoff keys)
        test_state = {
            "game_name": "Escalation_Game",
            "scenario_draft": {
                "behavior_choices": {
                    "escalate": "escalate behavior",
                    "withdraw": "withdraw behavior",
                },
                "payoff_matrix_description": {
                    "player 1: escalate, player 2: withdraw": ["outcome1", "outcome2"]
                    # Note: deliberately missing other required keys to trigger format error
                },
                "description": "Test scenario description",
            },
            "participants": ["Player1", "Player2"],
        }

        # Test with debug_mode=False (should return error in feedback, not raise exception)
        result_no_debug = verify_pay_off(test_state, debug_mode=False)
        self.assertIn("payoff_feedback", result_no_debug)
        self.assertFalse(result_no_debug["payoff_converged"])
        self.assertTrue(
            any(
                "Format error" in str(feedback)
                for feedback in result_no_debug["payoff_feedback"]
            )
        )

        # Test with debug_mode=True (should raise exception)
        with self.assertRaises((ValueError, KeyError)) as context:
            verify_pay_off(test_state, debug_mode=True)
        self.assertTrue(str(context.exception))  # Should have an error message

    def test_debug_mode_missing_behavior_choices(self):
        """Test debug_mode handling when behavior_choices are missing required actions."""
        from data_creation.scenario_creation.langgraph_creation.scenario_creation_graph import (
            verify_pay_off,
        )

        # Create a test state with incomplete behavior_choices
        test_state = {
            "game_name": "Escalation_Game",
            "scenario_draft": {
                "behavior_choices": {
                    "escalate": "escalate behavior"
                },  # Missing 'withdraw'
                "payoff_matrix_description": {},
                "description": "Test scenario description",
            },
            "participants": ["Player1", "Player2"],
        }

        # Test with debug_mode=False
        result_no_debug = verify_pay_off(test_state, debug_mode=False)
        self.assertIn("payoff_feedback", result_no_debug)
        self.assertFalse(result_no_debug["payoff_converged"])
        self.assertTrue(
            any(
                "Format error" in str(feedback)
                for feedback in result_no_debug["payoff_feedback"]
            )
        )

        # Test with debug_mode=True
        with self.assertRaises(KeyError):
            verify_pay_off(test_state, debug_mode=True)

    def test_debug_mode_missing_description(self):
        """Test debug_mode handling when description field is missing."""
        from data_creation.scenario_creation.langgraph_creation.scenario_creation_graph import (
            verify_pay_off,
        )

        # Create a test state missing the description field but with valid payoff structure
        # to ensure we reach the description check
        test_state = {
            "game_name": "Escalation_Game",
            "scenario_draft": {
                "behavior_choices": {
                    "withdraw": "withdraw behavior"
                },  # Only one action to avoid payoff errors
                "payoff_matrix_description": {
                    "player 1: withdraw": [
                        "outcome1",
                        "outcome2",
                    ]  # Valid payoff structure
                },
                # Missing 'description' field - this should be the error we hit
            },
            "participants": ["Player1", "Player2"],
        }

        # Test with debug_mode=False
        result_no_debug = verify_pay_off(test_state, debug_mode=False)
        self.assertIn("payoff_feedback", result_no_debug)
        self.assertFalse(result_no_debug["payoff_converged"])
        self.assertTrue(
            any(
                "Format error" in str(feedback)
                for feedback in result_no_debug["payoff_feedback"]
            )
        )

        # Test with debug_mode=True
        with self.assertRaises(KeyError):
            verify_pay_off(test_state, debug_mode=True)

    def test_recursion_limit_handling(self):
        """Test that recursion limits are properly handled."""
        # Create a graph with very low recursion limit
        graph = build_scenario_creation_graph(debug_mode=False)

        initial_state = {
            "game_name": "Test_Game",
            "participants": ["Alice", "Bob"],
            "participant_jobs": ["teacher", "doctor"],
            "scenario_draft": None,
            "narrative_feedback": [],
            "preference_feedback": [],
            "payoff_feedback": [],
            "iteration_count": 0,
            "final_scenario": None,
            "narrative_converged": False,
            "preference_converged": False,
            "payoff_converged": False,
            "all_converged": None,
            "auto_save_path": None,
        }

        config = {
            "configurable": {
                "thread_id": "test_recursion",
                "recursion_limit": 2,  # Very low limit
            }
        }

        # This should fail gracefully, not hang
        with self.assertRaises(Exception):
            graph.invoke(initial_state, config)


# --- Original Test Class --- #


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
                "payoff_matrix_description": {
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

    @patch(
        "data_creation.scenario_creation.langgraph_creation.scenario_creation_graph.get_llm"
    )
    @patch(
        "data_creation.scenario_creation.langgraph_creation.scenario_creation_graph.get_game_config"
    )
    def test_verify_pay_off_sequential_game(self, mock_get_game_config, mock_get_llm):
        """Tests if verify_pay_off handles sequential games correctly."""
        # Arrange
        mock_llm_instance = MagicMock()
        mock_response_content = json.dumps(
            {
                "feedback": [
                    "YES, plausible.",
                    "YES, plausible.",
                    "YES, plausible.",
                    "YES, plausible.",
                ],
                "converged": True,
            }
        )
        mock_llm_instance.invoke.return_value = MagicMock(content=mock_response_content)
        mock_get_llm.return_value = mock_llm_instance

        # Create mock for sequential game (Escalation Game structure)
        mock_game_config_instance = MagicMock()
        mock_payoff_matrix = MagicMock()

        # Mock the payoff leaves for Escalation Game structure
        mock_leaf_1 = MagicMock()
        mock_leaf_1.actions = ("withdraw",)
        mock_leaf_2 = MagicMock()
        mock_leaf_2.actions = ("escalate", "withdraw")
        mock_leaf_3 = MagicMock()
        mock_leaf_3.actions = ("escalate", "escalate", "withdraw")
        mock_leaf_4 = MagicMock()
        mock_leaf_4.actions = ("escalate", "escalate", "escalate")

        mock_payoff_matrix.payoff_leaves = [
            mock_leaf_1,
            mock_leaf_2,
            mock_leaf_3,
            mock_leaf_4,
        ]
        mock_payoff_matrix.ordered_payoff_leaves = [
            [
                ("withdraw",),
                ("escalate", "withdraw"),
                ("escalate", "escalate", "withdraw"),
                ("escalate", "escalate", "escalate"),
            ],
            [
                ("withdraw",),
                ("escalate", "withdraw"),
                ("escalate", "escalate", "withdraw"),
                ("escalate", "escalate", "escalate"),
            ],
        ]

        mock_game_config = {
            "scenario_class": MagicMock(),
            "decision_class": MagicMock(),
            "payoff_matrix": mock_payoff_matrix,
        }
        mock_get_game_config.return_value = mock_game_config

        # Create initial state with sequential game structure
        initial_state = self._get_base_initial_state()
        initial_state["game_name"] = "Escalation_Game"
        initial_state["scenario_draft"] = {
            "scenario": "Sequential Test Scenario",
            "description": "Test scenario for sequential games.",
            "behavior_choices": {
                "withdraw": "Choose to withdraw",
                "escalate": "Choose to escalate",
            },
            "payoff_matrix_description": {
                "P1: withdraw": [
                    "P1 gets peaceful outcome",
                    "P2 gets peaceful outcome",
                ],
                "P1: escalate , P2: withdraw": [
                    "P1 gets advantage from escalation",
                    "P2 gets disadvantage from backing down",
                ],
                "P1: escalate , P2: escalate , P1: withdraw": [
                    "P1 gets disadvantage from backing down after escalation",
                    "P2 gets advantage from counter-escalation",
                ],
                "P1: escalate , P2: escalate , P1: escalate": [
                    "P1 gets mutually destructive outcome",
                    "P2 gets mutually destructive outcome",
                ],
            },
        }

        # Use the actual function for testing
        from data_creation.scenario_creation.langgraph_creation.scenario_creation_graph import (
            verify_pay_off,
        )

        # Act
        updated_state = verify_pay_off(initial_state)

        # Assert
        mock_get_game_config.assert_called_once_with("Escalation_Game")
        mock_get_llm.assert_called_once_with(temperature=0.1, json_mode=True)
        mock_llm_instance.invoke.assert_called_once()

        # Verify that the function handled sequential game structure correctly
        # Capture the arguments passed to invoke
        invoke_args, _ = mock_llm_instance.invoke.call_args
        messages = invoke_args[0]
        human_message = next((msg for msg in messages if msg.type == "human"), None)
        self.assertIsNotNone(human_message, "HumanMessage not found in LLM call")
        human_message_content = human_message.content

        # Verify the generated questions for sequential game structure
        # For sequential games, players alternate, and action sequences can have different lengths

        # First leaf: only P1 acts (withdraw)
        expected_behavior_1 = "Participant 1 (P1) chooses 'Choose to withdraw'"
        expected_question_1 = PAYOFF_VALIDATION_QUESTION_FORMAT.format(
            behavior_description=expected_behavior_1,
            p1_outcome="P1 gets peaceful outcome",
            p2_outcome="P2 gets peaceful outcome",
        )

        # Second leaf: P1 escalates, P2 withdraws
        expected_behavior_2 = "Participant 1 (P1) chooses 'Choose to escalate' and Participant 2 (P2) chooses 'Choose to withdraw'"
        expected_question_2 = PAYOFF_VALIDATION_QUESTION_FORMAT.format(
            behavior_description=expected_behavior_2,
            p1_outcome="P1 gets advantage from escalation",
            p2_outcome="P2 gets disadvantage from backing down",
        )

        # Third leaf: P1 escalates, P2 escalates, P1 withdraws
        expected_behavior_3 = "Participant 1 (P1) chooses 'Choose to escalate' and Participant 2 (P2) chooses 'Choose to escalate' and Participant 1 (P1) chooses 'Choose to withdraw'"
        expected_question_3 = PAYOFF_VALIDATION_QUESTION_FORMAT.format(
            behavior_description=expected_behavior_3,
            p1_outcome="P1 gets disadvantage from backing down after escalation",
            p2_outcome="P2 gets advantage from counter-escalation",
        )

        # Fourth leaf: P1 escalates, P2 escalates, P1 escalates
        expected_behavior_4 = "Participant 1 (P1) chooses 'Choose to escalate' and Participant 2 (P2) chooses 'Choose to escalate' and Participant 1 (P1) chooses 'Choose to escalate'"
        expected_question_4 = PAYOFF_VALIDATION_QUESTION_FORMAT.format(
            behavior_description=expected_behavior_4,
            p1_outcome="P1 gets mutually destructive outcome",
            p2_outcome="P2 gets mutually destructive outcome",
        )

        # Check that the prompt contains the correctly formatted questions
        self.assertIn(expected_question_1, human_message_content)
        self.assertIn(expected_question_2, human_message_content)
        self.assertIn(expected_question_3, human_message_content)
        self.assertIn(expected_question_4, human_message_content)

        # Check state update
        self.assertEqual(
            updated_state["payoff_feedback"],
            [
                "YES, plausible.",
                "YES, plausible.",
                "YES, plausible.",
                "YES, plausible.",
            ],
        )
        self.assertTrue(updated_state["payoff_converged"])

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


class TestScenarioCreationPerformance(unittest.TestCase):
    """Performance tests to catch potential timeout issues."""

    def test_state_copying_performance(self):
        """Test that state copying operations are fast enough."""
        # Create a large state
        large_state = {
            "game_name": "Test_Game",
            "participants": ["Alice", "Bob"],
            "participant_jobs": ["teacher", "doctor"],
            "scenario_draft": {
                "description": "A" * 10000,
                "behavior_choices": {
                    f"action_{i}": f"Description {i}" for i in range(1000)
                },
                "payoff_matrix_description": {
                    f"outcome_{i}": [f"result_{i}_1", f"result_{i}_2"]
                    for i in range(1000)
                },
            },
            "narrative_feedback": [f"feedback_{i}" for i in range(100)],
            "preference_feedback": [f"feedback_{i}" for i in range(100)],
            "payoff_feedback": [f"feedback_{i}" for i in range(100)],
            "iteration_count": 5,
            "final_scenario": None,
            "narrative_converged": False,
            "preference_converged": False,
            "payoff_converged": False,
            "all_converged": None,
            "auto_save_path": None,
        }

        start_time = time.time()
        copied_state = {**large_state}  # Simulate state copying in the graph
        end_time = time.time()

        copying_time = end_time - start_time
        self.assertLess(
            copying_time, 0.1, f"State copying took too long: {copying_time} seconds"
        )
        self.assertEqual(copied_state["game_name"], "Test_Game")

    def test_large_feedback_processing_performance(self):
        """Test processing of large feedback arrays."""
        # Create large feedback arrays to test performance
        large_feedback = [
            f"feedback_{i}" * 100 for i in range(500)
        ]  # Very large feedback

        start_time = time.time()
        # Simulate joining feedback like the script does
        combined_feedback = "\n".join(large_feedback)
        processing_result = len(combined_feedback)  # Simulate processing
        end_time = time.time()

        processing_time = end_time - start_time
        self.assertLess(
            processing_time,
            1.0,
            f"Large feedback processing took too long: {processing_time} seconds",
        )

    def test_graph_execution_with_mocked_components_performance(self):
        """Test simplified graph performance without actually executing create_scenario."""
        # Test just the graph building performance
        start_time = time.time()
        graph = build_scenario_creation_graph(debug_mode=False)
        end_time = time.time()

        # Graph building should be fast
        build_time = end_time - start_time
        self.assertLess(
            build_time, 5.0, f"Graph building took too long: {build_time} seconds"
        )
        self.assertIsNotNone(graph)

    def test_timeout_protection_scenarios(self):
        """Test various scenarios that could lead to timeouts."""
        from data_creation.scenario_creation.langgraph_creation.scenario_creation_graph import (
            aggregate_verification,
            finalize_scenario,
            should_continue,
        )

        # Test with various problematic states
        test_states = [
            # Very high iteration count
            {"iteration_count": 100, "all_converged": False},
            # Missing all convergence flags (add all_converged)
            {"iteration_count": 1, "all_converged": False},
            # All None values (add all_converged)
            {
                "iteration_count": 1,
                "narrative_converged": None,
                "preference_converged": None,
                "payoff_converged": None,
                "all_converged": False,
            },
        ]

        for i, test_state in enumerate(test_states):
            with self.subTest(state=i):
                state = {
                    "game_name": "Test_Game",
                    "participants": ["Alice", "Bob"],
                    "participant_jobs": ["teacher", "doctor"],
                    "scenario_draft": {"scenario": "Test"},
                    "narrative_feedback": [],
                    "preference_feedback": [],
                    "payoff_feedback": [],
                    "final_scenario": None,
                    "auto_save_path": None,
                    **test_state,
                }

                start_time = time.time()

                # These operations should complete quickly even with problematic states
                should_continue_result = should_continue(state)
                aggregate_result = aggregate_verification(state)

                # Add all_converged to the state for finalize_scenario
                state_for_finalize = state.copy()
                state_for_finalize["all_converged"] = aggregate_result.get(
                    "all_converged", False
                )
                finalize_result = finalize_scenario(state_for_finalize)

                end_time = time.time()

                # Should complete in under 1 second
                processing_time = end_time - start_time
                self.assertLess(
                    processing_time,
                    1.0,
                    f"Operations took too long: {processing_time} seconds",
                )

                # Results should be valid
                self.assertIn(should_continue_result, ["finalize", "refine"])
                self.assertIn("all_converged", aggregate_result)
                self.assertIn("final_scenario", finalize_result)


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
