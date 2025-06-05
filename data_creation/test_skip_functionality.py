import json
import tempfile
import unittest
from pathlib import Path

from data_creation.create_scenario_langgraph import (
    filter_unprocessed_personas,
    get_existing_processed_personas,
)


class TestSkipFunctionality(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.scenario_path = Path(self.temp_dir) / "scenarios"
        self.scenario_path.mkdir(parents=True, exist_ok=True)

        # Sample persona jobs
        self.sample_persona_jobs = [
            "software engineer",
            "data scientist",
            "marketing manager",
            "product manager",
            "sales representative",
        ]

    def tearDown(self):
        """Clean up after each test method."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_get_existing_processed_personas_empty_directory(self):
        """Test get_existing_processed_personas with empty directory."""
        processed = get_existing_processed_personas(str(self.scenario_path))
        self.assertEqual(processed, set())

    def test_get_existing_processed_personas_with_files(self):
        """Test get_existing_processed_personas with existing scenario files."""
        # Create some mock scenario files
        test_files = [
            "SoftwareEngineer.json",
            "DataScientist.json",
            "MarketingManager.json",
        ]

        for filename in test_files:
            file_path = self.scenario_path / filename
            with open(file_path, "w") as f:
                json.dump({"test": "data"}, f)

        processed = get_existing_processed_personas(str(self.scenario_path))
        expected = {"software engineer", "data scientist", "marketing manager"}
        self.assertEqual(processed, expected)

    def test_get_existing_processed_personas_nonexistent_directory(self):
        """Test get_existing_processed_personas with non-existent directory."""
        nonexistent_path = str(Path(self.temp_dir) / "nonexistent")
        processed = get_existing_processed_personas(nonexistent_path)
        self.assertEqual(processed, set())

    def test_filter_unprocessed_personas_no_processed(self):
        """Test filter_unprocessed_personas when no personas are processed."""
        processed_personas = set()
        unprocessed, skipped = filter_unprocessed_personas(
            self.sample_persona_jobs, processed_personas
        )

        self.assertEqual(unprocessed, self.sample_persona_jobs)
        self.assertEqual(skipped, 0)

    def test_filter_unprocessed_personas_some_processed(self):
        """Test filter_unprocessed_personas when some personas are processed."""
        processed_personas = {"software engineer", "data scientist"}
        unprocessed, skipped = filter_unprocessed_personas(
            self.sample_persona_jobs, processed_personas
        )

        expected_unprocessed = [
            "marketing manager",
            "product manager",
            "sales representative",
        ]
        self.assertEqual(unprocessed, expected_unprocessed)
        self.assertEqual(skipped, 2)

    def test_filter_unprocessed_personas_all_processed(self):
        """Test filter_unprocessed_personas when all personas are processed."""
        processed_personas = set(self.sample_persona_jobs)
        unprocessed, skipped = filter_unprocessed_personas(
            self.sample_persona_jobs, processed_personas
        )

        self.assertEqual(unprocessed, [])
        self.assertEqual(skipped, len(self.sample_persona_jobs))

    def test_camel_case_conversion(self):
        """Test that CamelCase filenames are correctly converted back to original format."""
        # Create scenario files with various CamelCase patterns
        test_cases = [
            ("SoftwareEngineer.json", "software engineer"),
            ("DataScientist.json", "data scientist"),
            ("SeniorProductManager.json", "senior product manager"),
            (
                "CEOExecutive.json",
                "ceo executive",
            ),  # Consecutive capitals handled properly
            ("Manager.json", "manager"),  # Single word
        ]

        for filename, expected_persona in test_cases:
            file_path = self.scenario_path / filename
            with open(file_path, "w") as f:
                json.dump({"test": "data"}, f)

        processed = get_existing_processed_personas(str(self.scenario_path))
        expected_personas = {expected for _, expected in test_cases}
        self.assertEqual(processed, expected_personas)

    def test_case_insensitive_matching(self):
        """Test that persona matching is case insensitive."""
        # Create a file
        file_path = self.scenario_path / "SoftwareEngineer.json"
        with open(file_path, "w") as f:
            json.dump({"test": "data"}, f)

        processed_personas = get_existing_processed_personas(str(self.scenario_path))

        # Test with different case variations
        test_jobs = ["Software Engineer", "SOFTWARE ENGINEER", "software engineer"]
        unprocessed, skipped = filter_unprocessed_personas(
            test_jobs, processed_personas
        )

        # All should be skipped due to case-insensitive matching
        self.assertEqual(len(unprocessed), 0)
        self.assertEqual(skipped, 3)

    def test_integration_workflow(self):
        """Test the complete workflow integration."""
        # Create some existing scenario files
        existing_files = ["SoftwareEngineer.json", "DataScientist.json"]
        for filename in existing_files:
            file_path = self.scenario_path / filename
            with open(file_path, "w") as f:
                json.dump({"test": "data"}, f)

        # Get processed personas
        processed = get_existing_processed_personas(str(self.scenario_path))

        # Filter the job list
        unprocessed, skipped = filter_unprocessed_personas(
            self.sample_persona_jobs, processed
        )

        # Verify results
        self.assertEqual(skipped, 2)
        self.assertEqual(len(unprocessed), 3)
        self.assertNotIn("software engineer", unprocessed)
        self.assertNotIn("data scientist", unprocessed)
        self.assertIn("marketing manager", unprocessed)


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
