# Behavior Analyzer Script

This script analyzes behavioral choices from a JSON dataset, specifically designed for scenarios like the Prisoner's Dilemma. It uses Azure GPT-4o to extract action predicates from textual descriptions of 'cooperate' and 'defect' behaviors, then aggregates these predicates to show their distribution.

## Features

-   Loads scenario data from a specified JSON file.
-   Uses Azure GPT-4o to perform linguistic analysis on behavior descriptions.
-   Extracts key action predicates for 'cooperate' and 'defect' choices.
-   Processes entries in parallel to accelerate analysis.
-   Generates a CSV file summarizing the frequency of each predicate for both cooperation and defection.
-   Creates pie charts visualizing the distribution of predicates for cooperation and defection.
-   Supports mock API calls for testing without consuming Azure resources.

## Setup

1.  **Python Environment**:
    Ensure you have Python 3.7+ installed. It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  **Install Dependencies**:
    ```bash
    pip install pandas matplotlib openai
    ```
    *(Note: `openai` library version should be compatible with Azure. Typically `pip install openai>=1.0.0`)*

3.  **Azure OpenAI Configuration**:
    The script requires access to an Azure OpenAI deployment (e.g., GPT-4o).
    Set the following environment variables with your Azure OpenAI service credentials:
    -   `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key.
    -   `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI service endpoint (e.g., `https://your-resource-name.openai.azure.com/`).
    -   `AZURE_OPENAI_API_VERSION`: The API version (e.g., `2024-02-01`).
    -   `AZURE_OPENAI_DEPLOYMENT_NAME`: The name of your GPT-4o deployment on Azure.

    Example for `.bashrc` or `.zshrc`:
    ```bash
    export AZURE_OPENAI_API_KEY="your_actual_api_key"
    export AZURE_OPENAI_ENDPOINT="your_actual_endpoint"
    export AZURE_OPENAI_API_VERSION="2024-02-01"
    export AZURE_OPENAI_DEPLOYMENT_NAME="your_gpt_4o_deployment_name"
    ```

4.  **Input Data**:
    The script expects the input JSON file at:
    `data_creation/scenario_creation/langgraph_creation/Prisoners_Dilemma_all_data_samples.json`
    Ensure this file exists and is correctly formatted as a list of scenarios, each containing a `behavior_choices` dictionary with `"cooperate"` and `"defect"` text descriptions.

## Usage

Run the script from the root directory of the project:

```bash
python behavior_analyzer.py
```

### Mock API Calls (for Testing)

To run the script without making actual calls to the Azure GPT-4o API (e.g., for testing the data processing flow or when API access is unavailable), set the `MOCK_API_CALLS` environment variable to `true`:

```bash
export MOCK_API_CALLS=True
python behavior_analyzer.py
```

This will use predefined mock predicates instead of querying the AI model.

## Script Logic (`behavior_analyzer.py`)

1.  **Configuration**: Defines paths for input/output files, Azure client settings, and mock call flag.
2.  **Azure Client Initialization (`get_azure_openai_client`)**: Sets up the `AzureOpenAI` client using environment variables. Handles potential errors if variables are missing.
3.  **Predicate Extraction (`extract_predicates_from_text`)**: 
    -   If `MOCK_API_CALLS` is true, returns predefined mock predicates.
    -   Otherwise, constructs a prompt for GPT-4o, asking it to identify key action predicates from the input behavior text and return them as a JSON list of strings.
    -   Sends the request to the Azure OpenAI model and parses the JSON response.
    -   Includes error handling for API calls and JSON decoding.
4.  **Data Loading (`load_data`)**: Reads and parses the input JSON file.
5.  **Entry Processing (`process_entry`)**: For each scenario in the data:
    -   Retrieves the `cooperate` and `defect` behavior descriptions.
    -   Calls `extract_predicates_from_text` for both descriptions.
    -   Returns the lists of extracted predicates.
6.  **Main Execution (`main`)**:
    -   Initializes the Azure client.
    -   Loads scenario data.
    -   Uses a `ThreadPoolExecutor` to call `process_entry` for all scenarios in parallel, improving performance.
    -   Collects all extracted predicates for `cooperate` and `defect` actions separately.
    -   Uses `collections.Counter` to count the frequency of each predicate.
    -   Prints a summary of the top predicates to the console.
    -   Calls `generate_csv` to save the full frequency data.
    -   Calls `generate_pie_chart` to create and save visualizations for both cooperate and defect predicate distributions.
7.  **CSV Generation (`generate_csv`)**: 
    -   Creates a Pandas DataFrame from the predicate counts.
    -   Saves the DataFrame to a CSV file.
8.  **Pie Chart Generation (`generate_pie_chart`)**: 
    -   Uses Matplotlib to create a pie chart of predicate frequencies.
    -   Groups less frequent predicates into an "Others" category if there are more than `top_n` (default 15) predicates.
    -   Saves the chart as a PNG image.

## Outputs

-   `predicate_analysis.csv`: A CSV file with columns `predicate`, `cooperate_frequency`, `defect_frequency`.
-   `cooperate_predicates_pie.png`: A pie chart showing the distribution of predicates for 'cooperate' behaviors.
-   `defect_predicates_pie.png`: A pie chart showing the distribution of predicates for 'defect' behaviors.

## Unit Tests

Unit tests are located in the `tests/` directory (e.g., `tests/test_behavior_analyzer.py`). They use Python's built-in `unittest` module and focus on testing individual functions with mocked API calls and sample data.

To run tests:
```bash
python -m unittest discover tests
```