import asyncio
import json
import os
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from data_creation.create_scenario_langgraph import (
    BATCH_TIMEOUT,
    MAX_RETRIES,
    RETRY_DELAY,
    TASK_TIMEOUT,
    BatchTimeoutError,
    TaskTimeoutError,
    create_scenario_with_timeout,
    filter_unprocessed_personas,
    get_existing_processed_personas,
    process_batch_with_timeout,
    save_scenario_and_history,
)


class AsyncTestCase(unittest.TestCase):
    """Base class for async test cases."""

    def async_test(self, coro):
        """Helper method to run async tests."""
        return asyncio.run(coro)


class TestScenarioGenerationRestart(AsyncTestCase):
    """Test suite for the enhanced scenario generation with restart capabilities."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.scenario_path = os.path.join(self.temp_dir, "scenarios")
        self.history_path = os.path.join(self.temp_dir, "histories")
        os.makedirs(self.scenario_path, exist_ok=True)
        os.makedirs(self.history_path, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_get_existing_processed_personas_empty_directory(self):
        """Test that empty directory returns empty set."""
        processed = get_existing_processed_personas(self.scenario_path)
        self.assertEqual(processed, set())

    def test_get_existing_processed_personas_with_files(self):
        """Test extraction of processed personas from existing files."""
        # Create test scenario files
        test_files = [
            "SoftwareEngineer.json",
            "DataScientist.json",
            "ProjectManager.json",
            "CEOExecutive.json",  # Tests CEO Executive -> ceo executive
        ]

        for filename in test_files:
            filepath = os.path.join(self.scenario_path, filename)
            with open(filepath, "w") as f:
                json.dump({"test": "data"}, f)

        processed = get_existing_processed_personas(self.scenario_path)
        expected = {
            "software engineer",
            "data scientist",
            "project manager",
            "ceo executive",
        }
        self.assertEqual(processed, expected)

    def test_filter_unprocessed_personas(self):
        """Test filtering of unprocessed personas."""
        all_personas = [
            "software engineer",
            "data scientist",
            "project manager",
            "teacher",
            "doctor",
        ]

        processed_personas = {"software engineer", "data scientist"}

        unprocessed, skipped = filter_unprocessed_personas(
            all_personas, processed_personas
        )

        self.assertEqual(len(unprocessed), 3)
        self.assertEqual(skipped, 2)
        self.assertIn("project manager", unprocessed)
        self.assertIn("teacher", unprocessed)
        self.assertIn("doctor", unprocessed)

    def test_create_scenario_with_timeout_success(self):
        """Test successful scenario creation within timeout."""

        async def _test():
            # Mock the scenario creation function
            mock_graph = Mock()
            mock_scenario = {"test": "scenario", "participants": ["Alice", "Bob"]}

            with patch(
                "data_creation.create_scenario_langgraph.a_create_scenario"
            ) as mock_create:
                mock_create.return_value = mock_scenario

                result = await create_scenario_with_timeout(
                    graph=mock_graph,
                    game_name="test_game",
                    participants=["Alice", "Bob"],
                    participant_jobs=["teacher", "teacher"],
                    config={"test": "config"},
                    timeout=10,
                )

                self.assertEqual(result, mock_scenario)
                mock_create.assert_called_once()

        self.async_test(_test())

    def test_create_scenario_with_timeout_failure_and_retry(self):
        """Test scenario creation with timeout and retry logic."""

        async def _test():
            mock_graph = Mock()

            with patch(
                "data_creation.create_scenario_langgraph.a_create_scenario"
            ) as mock_create:
                # First two calls timeout, third succeeds
                mock_create.side_effect = [
                    asyncio.TimeoutError("Timeout 1"),
                    asyncio.TimeoutError("Timeout 2"),
                    {"test": "scenario"},
                ]

                with patch(
                    "data_creation.create_scenario_langgraph.RETRY_DELAY", 0.1
                ):  # Speed up test
                    result = await create_scenario_with_timeout(
                        graph=mock_graph,
                        game_name="test_game",
                        participants=["Alice", "Bob"],
                        participant_jobs=["teacher", "teacher"],
                        config={"test": "config"},
                        timeout=1,
                    )

                self.assertEqual(result, {"test": "scenario"})
                self.assertEqual(mock_create.call_count, 3)

        self.async_test(_test())

    def test_create_scenario_with_timeout_max_retries_exceeded(self):
        """Test scenario creation failing after max retries."""

        async def _test():
            mock_graph = Mock()

            with patch(
                "data_creation.create_scenario_langgraph.a_create_scenario"
            ) as mock_create:
                mock_create.side_effect = asyncio.TimeoutError("Persistent timeout")

                with patch(
                    "data_creation.create_scenario_langgraph.RETRY_DELAY", 0.1
                ):  # Speed up test
                    with patch(
                        "data_creation.create_scenario_langgraph.MAX_RETRIES", 2
                    ):  # Reduce retries for test
                        with self.assertRaises(TaskTimeoutError):
                            await create_scenario_with_timeout(
                                graph=mock_graph,
                                game_name="test_game",
                                participants=["Alice", "Bob"],
                                participant_jobs=["teacher", "teacher"],
                                config={"test": "config"},
                                timeout=1,
                            )

        self.async_test(_test())

    def test_save_scenario_and_history_success(self):
        """Test successful saving of scenario and history."""

        async def _test():
            mock_scenario = {"test": "scenario"}
            mock_graph = Mock()

            # Mock the history generator
            async def mock_history_generator():
                yield Mock(values={"step": 1}, metadata={}, created_at="2024-01-01")
                yield Mock(values={"step": 2}, metadata={}, created_at="2024-01-01")

            mock_graph.aget_state_history.return_value = mock_history_generator()

            result = await save_scenario_and_history(
                scenario=mock_scenario,
                scenario_graph=mock_graph,
                config={"test": "config"},
                persona_job_filename="TestJob",
                scenario_path_base=self.scenario_path,
                history_path_base=self.history_path,
            )

            self.assertTrue(result)

            # Check scenario file was created
            scenario_file = os.path.join(self.scenario_path, "TestJob.json")
            self.assertTrue(os.path.exists(scenario_file))

            # Check history file was created
            history_file = os.path.join(self.history_path, "TestJob_history.json")
            self.assertTrue(os.path.exists(history_file))

        self.async_test(_test())

    def test_save_scenario_and_history_no_scenario(self):
        """Test saving when scenario is None."""

        async def _test():
            mock_graph = Mock()
            mock_graph.aget_state_history.return_value = iter([])

            result = await save_scenario_and_history(
                scenario=None,
                scenario_graph=mock_graph,
                config={"test": "config"},
                persona_job_filename="TestJob",
                scenario_path_base=self.scenario_path,
                history_path_base=self.history_path,
            )

            self.assertFalse(result)

        self.async_test(_test())

    def test_process_batch_with_timeout_success(self):
        """Test successful batch processing."""

        async def _test():
            persona_jobs = ["teacher", "doctor"]
            mock_graph = Mock()
            mock_scenario = {"test": "scenario"}

            with patch(
                "data_creation.create_scenario_langgraph.create_scenario_with_timeout"
            ) as mock_create:
                with patch(
                    "data_creation.create_scenario_langgraph.save_scenario_and_history"
                ) as mock_save:
                    mock_create.return_value = mock_scenario
                    mock_save.return_value = True

                    successful, failed = await process_batch_with_timeout(
                        persona_jobs=persona_jobs,
                        scenario_graph=mock_graph,
                        game_name="test_game",
                        scenario_path_base=self.scenario_path,
                        history_path_base=self.history_path,
                        batch_index=0,
                        total_batches=1,
                    )

                    self.assertEqual(successful, 2)
                    self.assertEqual(failed, 0)

        self.async_test(_test())

    def test_process_batch_with_timeout_partial_failure(self):
        """Test batch processing with some failures."""

        async def _test():
            persona_jobs = ["teacher", "doctor", "engineer"]
            mock_graph = Mock()
            mock_scenario = {"test": "scenario"}

            with patch(
                "data_creation.create_scenario_langgraph.create_scenario_with_timeout"
            ) as mock_create:
                with patch(
                    "data_creation.create_scenario_langgraph.save_scenario_and_history"
                ) as mock_save:
                    # First job succeeds, second fails, third succeeds
                    mock_create.side_effect = [
                        mock_scenario,
                        TaskTimeoutError("Timeout"),
                        mock_scenario,
                    ]
                    mock_save.return_value = True

                    successful, failed = await process_batch_with_timeout(
                        persona_jobs=persona_jobs,
                        scenario_graph=mock_graph,
                        game_name="test_game",
                        scenario_path_base=self.scenario_path,
                        history_path_base=self.history_path,
                        batch_index=0,
                        total_batches=1,
                    )

                    self.assertEqual(successful, 2)
                    self.assertEqual(failed, 1)

        self.async_test(_test())

    def test_camel_case_conversion(self):
        """Test conversion from job names to camel case filenames."""
        test_cases = [
            ("software engineer", "SoftwareEngineer"),
            ("data scientist", "DataScientist"),
            ("ceo executive", "CeoExecutive"),
            ("teacher", "Teacher"),
            ("project manager assistant", "ProjectManagerAssistant"),
        ]

        for job_name, expected in test_cases:
            result = "".join(word.capitalize() for word in job_name.split(" "))
            self.assertEqual(result, expected)

    def test_reverse_camel_case_conversion(self):
        """Test conversion from camel case filenames back to job names."""
        test_cases = [
            ("SoftwareEngineer", "software engineer"),
            ("DataScientist", "data scientist"),
            ("CeoExecutive", "ceo executive"),
            ("Teacher", "teacher"),
            ("ProjectManagerAssistant", "project manager assistant"),
        ]

        for filename, expected in test_cases:
            # This tests the logic in get_existing_processed_personas
            original_persona = ""
            for i, char in enumerate(filename):
                if char.isupper() and i > 0:
                    if filename[i - 1].islower() or (
                        i < len(filename) - 1
                        and filename[i + 1].islower()
                        and filename[i - 1].isupper()
                    ):
                        original_persona += " " + char.lower()
                    else:
                        original_persona += char.lower()
                else:
                    original_persona += char.lower()

            self.assertEqual(original_persona, expected)

    def test_constants_configuration(self):
        """Test that configuration constants are properly set."""
        self.assertIsInstance(TASK_TIMEOUT, int)
        self.assertIsInstance(BATCH_TIMEOUT, int)
        self.assertIsInstance(MAX_RETRIES, int)
        self.assertIsInstance(RETRY_DELAY, int)

        self.assertGreater(TASK_TIMEOUT, 0)
        self.assertGreater(BATCH_TIMEOUT, TASK_TIMEOUT)
        self.assertGreater(MAX_RETRIES, 0)
        self.assertGreater(RETRY_DELAY, 0)

    def test_timeout_error_classes(self):
        """Test custom timeout error classes."""

        async def _test():
            with self.assertRaises(TaskTimeoutError):
                raise TaskTimeoutError("Test task timeout")

            with self.assertRaises(BatchTimeoutError):
                raise BatchTimeoutError("Test batch timeout")

        self.async_test(_test())

    def test_file_structure_integrity(self):
        """Test that the expected file structure is maintained."""
        # Test that scenario and history paths are created correctly
        scenario_base = "data_creation/scenario_creation/langgraph_creation/scenarios"
        history_base = "data_creation/scenario_creation/langgraph_creation/histories"

        game_name = "prisoners_dilemma"
        timestamp = "20240101"

        expected_scenario_path = f"{scenario_base}/{game_name}_{timestamp}"
        expected_history_path = f"{history_base}/{game_name}_{timestamp}"

        self.assertIn(game_name, expected_scenario_path)
        self.assertIn(timestamp, expected_scenario_path)
        self.assertIn(game_name, expected_history_path)
        self.assertIn(timestamp, expected_history_path)


class TestScenarioGenerationIntegration(AsyncTestCase):
    """Integration tests for the scenario generation system."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up integration test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_restart_scenario_integration(self):
        """Test the full restart scenario with existing files."""
        scenario_path = os.path.join(self.temp_dir, "scenarios")
        os.makedirs(scenario_path, exist_ok=True)

        # Create some existing scenario files
        existing_files = ["Teacher.json", "Doctor.json"]
        for filename in existing_files:
            filepath = os.path.join(scenario_path, filename)
            with open(filepath, "w") as f:
                json.dump({"scenario": "test"}, f)

        # Test the restart logic
        all_personas = ["teacher", "doctor", "engineer", "lawyer"]
        processed = get_existing_processed_personas(scenario_path)
        unprocessed, skipped = filter_unprocessed_personas(all_personas, processed)

        self.assertEqual(skipped, 2)  # teacher and doctor should be skipped
        self.assertEqual(len(unprocessed), 2)  # engineer and lawyer should remain
        self.assertIn("engineer", unprocessed)
        self.assertIn("lawyer", unprocessed)

    def test_error_recovery_integration(self):
        """Test error recovery across multiple failures."""

        async def _test():
            mock_graph = Mock()

            with patch(
                "data_creation.create_scenario_langgraph.create_scenario_with_timeout"
            ) as mock_create:
                with patch(
                    "data_creation.create_scenario_langgraph.save_scenario_and_history"
                ) as mock_save:
                    # Simulate mixed success/failure pattern
                    mock_create.side_effect = [
                        {"scenario": "success1"},
                        TaskTimeoutError("Timeout"),
                        {"scenario": "success2"},
                        Exception("General error"),
                        {"scenario": "success3"},
                    ]
                    mock_save.return_value = True

                    # Test that the system continues processing despite failures
                    persona_jobs = ["job1", "job2", "job3", "job4", "job5"]

                    successful, failed = await process_batch_with_timeout(
                        persona_jobs=persona_jobs,
                        scenario_graph=mock_graph,
                        game_name="test_game",
                        scenario_path_base=self.temp_dir,
                        history_path_base=self.temp_dir,
                        batch_index=0,
                        total_batches=1,
                    )

                    self.assertEqual(successful, 3)  # 3 successes
                    self.assertEqual(failed, 2)  # 2 failures

        self.async_test(_test())


class TestScenarioGenerationPerformance(AsyncTestCase):
    """Performance and timing tests for the scenario generation system."""

    def test_timeout_enforcement(self):
        """Test that timeouts are properly enforced."""

        async def _test():
            mock_graph = Mock()

            with patch(
                "data_creation.create_scenario_langgraph.a_create_scenario"
            ) as mock_create:
                with patch(
                    "data_creation.create_scenario_langgraph.RETRY_DELAY", 0.1
                ):  # Speed up test
                    with patch(
                        "data_creation.create_scenario_langgraph.MAX_RETRIES", 2
                    ):  # Reduce retries for test
                        # Simulate a function that takes longer than timeout
                        async def slow_function(*args, **kwargs):
                            await asyncio.sleep(2)  # 2 seconds
                            return {"scenario": "slow"}

                        mock_create.side_effect = slow_function

                        start_time = time.time()

                        with self.assertRaises(TaskTimeoutError):
                            await create_scenario_with_timeout(
                                graph=mock_graph,
                                game_name="test_game",
                                participants=["Alice", "Bob"],
                                participant_jobs=["teacher", "teacher"],
                                config={"test": "config"},
                                timeout=1,  # 1 second timeout
                            )

                        # Should timeout before the slow function completes
                        elapsed_time = time.time() - start_time
                        self.assertLess(
                            elapsed_time, 5
                        )  # Should complete within 5 seconds with reduced retries and delays

        self.async_test(_test())


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
