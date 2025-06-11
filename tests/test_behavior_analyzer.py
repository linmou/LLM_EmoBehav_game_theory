import json
import os

# Add the parent directory to sys.path to allow direct import of behavior_analyzer
import sys
import unittest
from collections import Counter
from unittest.mock import MagicMock, patch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

import result_analysis.behavior_predicate_analyzer as behavior_predicate_analyzer  # Now behavior_analyzer can be imported

# Sample data for testing (mimicking the structure of Prisoners_Dilemma_all_data_samples.json)
SAMPLE_SCENARIO_DATA = [
    {
        "scenario": "Test Scenario 1",
        "behavior_choices": {
            "cooperate": "Player decides to cooperate fully with partner.",
            "defect": "Player chooses to defect and betray partner.",
        },
    },
    {
        "scenario": "Test Scenario 2",
        "behavior_choices": {
            "cooperate": "Share resources and work together.",
            "defect": "Keep all resources secret and act alone.",
        },
    },
    {
        "scenario": "Test Scenario 3 - Only Cooperate",
        "behavior_choices": {"cooperate": "Trust and support."},
    },
    {
        "scenario": "Test Scenario 4 - Only Defect",
        "behavior_choices": {"defect": "Backstab and deceive."},
    },
    {"scenario": "Test Scenario 5 - Empty Choices", "behavior_choices": {}},
]


class TestBehaviorAnalyzer(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.test_json_path = "test_data.json"
        with open(self.test_json_path, "w") as f:
            json.dump(SAMPLE_SCENARIO_DATA, f)

        # Ensure output files from previous tests are cleaned up
        self.test_csv_path = "test_predicates.csv"
        self.test_coop_pie_path = "test_coop_pie.png"
        self.test_defect_pie_path = "test_defect_pie.png"
        for path in [
            self.test_csv_path,
            self.test_coop_pie_path,
            self.test_defect_pie_path,
        ]:
            if os.path.exists(path):
                os.remove(path)

    def tearDown(self):
        """Tear down after test methods."""
        if os.path.exists(self.test_json_path):
            os.remove(self.test_json_path)
        for path in [
            self.test_csv_path,
            self.test_coop_pie_path,
            self.test_defect_pie_path,
        ]:
            if os.path.exists(path):
                os.remove(path)

    def test_load_data(self):
        """Test loading data from JSON."""
        data = behavior_predicate_analyzer.load_data(self.test_json_path)
        self.assertEqual(len(data), len(SAMPLE_SCENARIO_DATA))
        self.assertEqual(data[0]["scenario"], "Test Scenario 1")

        # Test file not found
        data_not_found = behavior_predicate_analyzer.load_data("non_existent_file.json")
        self.assertEqual(data_not_found, [])

        # Test invalid JSON
        with open("invalid.json", "w") as f:
            f.write("this is not json")
        data_invalid = behavior_predicate_analyzer.load_data("invalid.json")
        self.assertEqual(data_invalid, [])
        os.remove("invalid.json")

    @patch("behavior_analyzer.AzureOpenAI")  # Mock the AzureOpenAI client
    def test_extract_predicates_from_text_mock_on(self, MockAzureOpenAI):
        """Test predicate extraction with MOCK_API_CALLS=True."""
        # Ensure client is not actually used when MOCK_API_CALLS is true
        mock_client_instance = MockAzureOpenAI.return_value

        with patch.dict(os.environ, {"MOCK_API_CALLS": "True"}):
            # Re-import or reload behavior_analyzer if MOCK_API_CALLS is read at import time
            # For simplicity here, we assume behavior_analyzer.MOCK_API_CALLS is updated or checked dynamically.
            # If it's a module-level constant set on import, this test needs adjustment or the module reloaded.
            # For now, let's directly set the module's variable for testing effect.
            original_mock_flag = behavior_predicate_analyzer.MOCK_API_CALLS
            behavior_predicate_analyzer.MOCK_API_CALLS = True

            predicates_coop = behavior_predicate_analyzer.extract_predicates_from_text(
                "cooperate text", mock_client_instance, "test-deployment"
            )
            self.assertEqual(
                predicates_coop, ["mock_cooperate_action", "mock_share_resource"]
            )

            predicates_defect = (
                behavior_predicate_analyzer.extract_predicates_from_text(
                    "defect text", mock_client_instance, "test-deployment"
                )
            )
            self.assertEqual(
                predicates_defect, ["mock_defect_action", "mock_keep_secret"]
            )

            predicates_generic = (
                behavior_predicate_analyzer.extract_predicates_from_text(
                    "some other text", mock_client_instance, "test-deployment"
                )
            )
            self.assertEqual(predicates_generic, ["mock_generic_action"])

            behavior_predicate_analyzer.MOCK_API_CALLS = (
                original_mock_flag  # Reset flag
            )

    @patch("behavior_analyzer.AzureOpenAI")
    def test_extract_predicates_from_text_api_call_success(self, MockAzureOpenAI):
        """Test predicate extraction with a successful API call (mocked)."""
        mock_client_instance = MockAzureOpenAI.return_value
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = json.dumps(
            ["extracted predicate 1", "extracted predicate 2"]
        )
        mock_client_instance.chat.completions.create.return_value = mock_response

        original_mock_flag = behavior_predicate_analyzer.MOCK_API_CALLS
        behavior_predicate_analyzer.MOCK_API_CALLS = (
            False  # Ensure we are testing the API path
        )

        predicates = behavior_predicate_analyzer.extract_predicates_from_text(
            "real text", mock_client_instance, "test-deployment"
        )
        self.assertEqual(predicates, ["extracted predicate 1", "extracted predicate 2"])
        mock_client_instance.chat.completions.create.assert_called_once()

        behavior_predicate_analyzer.MOCK_API_CALLS = original_mock_flag

    @patch("behavior_analyzer.AzureOpenAI")
    def test_extract_predicates_from_text_api_call_json_error(self, MockAzureOpenAI):
        """Test predicate extraction with API returning invalid JSON."""
        mock_client_instance = MockAzureOpenAI.return_value
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "not a valid json string"
        mock_client_instance.chat.completions.create.return_value = mock_response

        original_mock_flag = behavior_predicate_analyzer.MOCK_API_CALLS
        behavior_predicate_analyzer.MOCK_API_CALLS = False

        predicates = behavior_predicate_analyzer.extract_predicates_from_text(
            "real text", mock_client_instance, "test-deployment"
        )
        self.assertEqual(predicates, [])

        behavior_predicate_analyzer.MOCK_API_CALLS = original_mock_flag

    @patch("behavior_analyzer.AzureOpenAI")
    def test_extract_predicates_from_text_api_call_exception(self, MockAzureOpenAI):
        """Test predicate extraction when API call raises an exception."""
        mock_client_instance = MockAzureOpenAI.return_value
        mock_client_instance.chat.completions.create.side_effect = Exception(
            "API Error"
        )

        original_mock_flag = behavior_predicate_analyzer.MOCK_API_CALLS
        behavior_predicate_analyzer.MOCK_API_CALLS = False

        predicates = behavior_predicate_analyzer.extract_predicates_from_text(
            "real text", mock_client_instance, "test-deployment"
        )
        self.assertEqual(predicates, [])

        behavior_predicate_analyzer.MOCK_API_CALLS = original_mock_flag

    @patch("behavior_analyzer.extract_predicates_from_text")
    def test_process_entry(self, mock_extract_predicates):
        """Test processing a single data entry."""
        # Define return values for cooperate and defect texts
        mock_extract_predicates.side_effect = [
            ["coop_pred1", "coop_pred2"],  # For cooperate text of first call
            ["def_pred1"],  # For defect text of first call
        ]

        entry = SAMPLE_SCENARIO_DATA[0]
        coop_preds, def_preds = behavior_predicate_analyzer.process_entry(
            entry, MagicMock(), "test-deploy"
        )

        self.assertEqual(coop_preds, ["coop_pred1", "coop_pred2"])
        self.assertEqual(def_preds, ["def_pred1"])
        self.assertEqual(mock_extract_predicates.call_count, 2)
        mock_extract_predicates.assert_any_call(
            entry["behavior_choices"]["cooperate"], unittest.mock.ANY, "test-deploy"
        )
        mock_extract_predicates.assert_any_call(
            entry["behavior_choices"]["defect"], unittest.mock.ANY, "test-deploy"
        )

    @patch("behavior_analyzer.extract_predicates_from_text")
    def test_process_entry_missing_choices(self, mock_extract_predicates):
        """Test processing entry with missing cooperate or defect text."""
        mock_extract_predicates.return_value = ["mock_pred"]

        entry_only_coop = SAMPLE_SCENARIO_DATA[2]
        coop_preds, def_preds = behavior_predicate_analyzer.process_entry(
            entry_only_coop, MagicMock(), "test-deploy"
        )
        self.assertEqual(coop_preds, ["mock_pred"])
        self.assertEqual(def_preds, [])
        mock_extract_predicates.assert_called_once_with(
            entry_only_coop["behavior_choices"]["cooperate"],
            unittest.mock.ANY,
            "test-deploy",
        )

        mock_extract_predicates.reset_mock()
        entry_only_defect = SAMPLE_SCENARIO_DATA[3]
        coop_preds, def_preds = behavior_predicate_analyzer.process_entry(
            entry_only_defect, MagicMock(), "test-deploy"
        )
        self.assertEqual(coop_preds, [])
        self.assertEqual(def_preds, ["mock_pred"])
        mock_extract_predicates.assert_called_once_with(
            entry_only_defect["behavior_choices"]["defect"],
            unittest.mock.ANY,
            "test-deploy",
        )

    def test_generate_csv(self):
        """Test CSV generation."""
        cooperate_counts = Counter({"share": 2, "help": 1})
        defect_counts = Counter({"steal": 1, "share": 1})
        behavior_predicate_analyzer.generate_csv(
            cooperate_counts, defect_counts, self.test_csv_path
        )
        self.assertTrue(os.path.exists(self.test_csv_path))

        # Basic check: read and verify header and a row (simplified)
        import pandas as pd

        df = pd.read_csv(self.test_csv_path)
        self.assertListEqual(
            list(df.columns), ["predicate", "cooperate_frequency", "defect_frequency"]
        )
        self.assertIn("share", df["predicate"].values)
        # Further checks can be added for specific values

    @patch(
        "matplotlib.pyplot.savefig"
    )  # Mock savefig to avoid actual file saving during test
    @patch("matplotlib.pyplot.show")  # Mock show if it's called
    @patch("matplotlib.pyplot.close")  # Mock close
    def test_generate_pie_chart(self, mock_close, mock_show, mock_savefig):
        """Test pie chart generation (mocking Matplotlib)."""
        predicate_counts = Counter({"actionA": 10, "actionB": 5, "actionC": 3})
        behavior_predicate_analyzer.generate_pie_chart(
            predicate_counts, "Test Pie", self.test_coop_pie_path
        )
        mock_savefig.assert_called_with(self.test_coop_pie_path)
        # Test with empty counts
        mock_savefig.reset_mock()
        behavior_predicate_analyzer.generate_pie_chart(
            Counter(), "Empty Pie", self.test_defect_pie_path
        )
        mock_savefig.assert_not_called()  # Should not attempt to save if no data

    @patch("behavior_analyzer.get_azure_openai_client")
    @patch("behavior_analyzer.load_data")
    @patch(
        "behavior_analyzer.ThreadPoolExecutor"
    )  # Mock the executor to run sequentially for tests
    @patch("behavior_analyzer.generate_csv")
    @patch("behavior_analyzer.generate_pie_chart")
    def test_main_flow_mock_api(
        self,
        mock_gen_pie,
        mock_gen_csv,
        MockThreadPoolExecutor,
        mock_load_data,
        mock_get_client,
    ):
        """Test the main execution flow with MOCK_API_CALLS=True."""
        # Setup mocks
        mock_get_client.return_value = (
            MagicMock(),
            "test-deployment",
        )  # Mock client and deployment
        mock_load_data.return_value = SAMPLE_SCENARIO_DATA[
            :2
        ]  # Use a subset for faster testing

        # Mock the executor to simulate sequential execution and return values
        class MockFuture:
            def __init__(self, result):
                self._result = result

            def result(self):
                return self._result

        # Simulate what process_entry would return with mock predicates
        mock_process_entry_results = [
            (
                ["mock_cooperate_action", "mock_share_resource"],
                ["mock_defect_action", "mock_keep_secret"],
            ),  # For entry 1
            (
                ["mock_cooperate_action", "mock_share_resource"],
                ["mock_defect_action", "mock_keep_secret"],
            ),  # For entry 2
        ]

        mock_executor_instance = (
            MockThreadPoolExecutor.return_value.__enter__.return_value
        )
        # Make submit return MockFuture with pre-defined results
        mock_executor_instance.submit.side_effect = [
            MockFuture(res) for res in mock_process_entry_results
        ]

        with patch.dict(os.environ, {"MOCK_API_CALLS": "True"}):
            original_mock_flag = behavior_predicate_analyzer.MOCK_API_CALLS
            behavior_predicate_analyzer.MOCK_API_CALLS = True

            behavior_predicate_analyzer.main()

            behavior_predicate_analyzer.MOCK_API_CALLS = original_mock_flag

        mock_load_data.assert_called_with(behavior_predicate_analyzer.INPUT_JSON_PATH)
        self.assertTrue(
            mock_executor_instance.submit.call_count >= 2
        )  # Called for each entry
        mock_gen_csv.assert_called_once()
        self.assertEqual(mock_gen_pie.call_count, 2)  # One for coop, one for defect

    def test_get_azure_openai_client_env_vars_set(self):
        """Test Azure client initialization when all env vars are set."""
        with patch.dict(
            os.environ,
            {
                "AZURE_OPENAI_API_KEY": "test_key",
                "AZURE_OPENAI_ENDPOINT": "test_endpoint",
                "AZURE_OPENAI_API_VERSION": "test_version",
                "AZURE_OPENAI_DEPLOYMENT_NAME": "test_deployment",
            },
        ):
            with patch("behavior_analyzer.AzureOpenAI") as MockAzureOpenAIConstructor:
                mock_client_instance = MockAzureOpenAIConstructor.return_value
                client, deployment = (
                    behavior_predicate_analyzer.get_azure_openai_client()
                )
                MockAzureOpenAIConstructor.assert_called_once_with(
                    api_key="test_key",
                    azure_endpoint="test_endpoint",
                    api_version="test_version",
                )
                self.assertEqual(client, mock_client_instance)
                self.assertEqual(deployment, "test_deployment")

    def test_get_azure_openai_client_env_vars_missing(self):
        """Test Azure client init when env vars are missing (and fallback is used)."""
        # Ensure critical env vars are not set for this test
        temp_env = os.environ.copy()
        if "AZURE_OPENAI_API_KEY" in temp_env:
            del temp_env["AZURE_OPENAI_API_KEY"]
        if "AZURE_OPENAI_ENDPOINT" in temp_env:
            del temp_env["AZURE_OPENAI_ENDPOINT"]
        # Keep AZURE_OPENAI_DEPLOYMENT_NAME as it might be used by fallback
        if "AZURE_OPENAI_DEPLOYMENT_NAME" not in temp_env:
            temp_env["AZURE_OPENAI_DEPLOYMENT_NAME"] = "gpt-4o"  # default if not set

        with patch.dict(os.environ, temp_env, clear=True):
            with patch("behavior_analyzer.AzureOpenAI") as MockAzureOpenAIConstructor:
                # Mock the hardcoded fallback config for predictability in test
                behavior_predicate_analyzer.AZURE_OPENAI_CONFIG = {
                    "api_key": "FALLBACK_KEY",
                    "azure_endpoint": "FALLBACK_ENDPOINT",
                    "api_version": "FALLBACK_VERSION",
                }
                mock_client_instance = MockAzureOpenAIConstructor.return_value
                client, deployment = (
                    behavior_predicate_analyzer.get_azure_openai_client()
                )

                MockAzureOpenAIConstructor.assert_called_once_with(
                    api_key="FALLBACK_KEY",
                    azure_endpoint="FALLBACK_ENDPOINT",
                    api_version="FALLBACK_VERSION",
                )
                self.assertEqual(client, mock_client_instance)
                self.assertEqual(deployment, "gpt-4o")  # Default or env var if set


if __name__ == "__main__":
    unittest.main()
