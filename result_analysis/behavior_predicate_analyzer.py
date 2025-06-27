import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import pandas as pd
from openai import AzureOpenAI  # Assuming this is how AzureOpenAI is imported

INPUT_JSON_PATH = "data_creation/scenario_creation/langgraph_creation/Prisoners_Dilemma_all_data_samples.json"
OUTPUT_CSV_PATH = "predicate_analysis.csv"
COOPERATE_PIE_CHART_PATH = "cooperate_predicates_pie.png"
DEFECT_PIE_CHART_PATH = "defect_predicates_pie.png"
MAX_WORKERS_DEFAULT = 10  # Default number of parallel workers

from api_configs import AZURE_OPENAI_CONFIG


# --- Azure GPT-4o Analysis Function ---
def get_azure_openai_client():
    """Initializes and returns the AzureOpenAI client."""
    # This is a placeholder. User should adapt this from their azure_test.py
    # or their specific Azure OpenAI setup.
    try:
        # Option 1: Load from environment variables (Recommended)
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv(
            "AZURE_OPENAI_API_VERSION", "2024-02-01"
        )  # Default if not set
        deployment_name = os.getenv(
            "AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"
        )  # Default if not set

        if not all([azure_api_key, azure_endpoint, deployment_name]):
            print(
                "Warning: One or more Azure OpenAI environment variables (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME) are not set."
            )
            print(
                "Attempting to use a hardcoded configuration (not recommended for production)."
            )
            # Option 2: Fallback to a hardcoded config if necessary (replace with actual values or remove)
            # This is just an example structure and should be secured.

            client = AzureOpenAI(
                api_key=AZURE_OPENAI_CONFIG["api_key"],
                azure_endpoint=AZURE_OPENAI_CONFIG["azure_endpoint"],
                api_version=AZURE_OPENAI_CONFIG["api_version"],
            )
            deployment = deployment_name  # Still need deployment name
            return client, deployment

        client = AzureOpenAI(
            api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        return client, deployment_name

    except Exception as e:
        print(f"Error initializing AzureOpenAI client: {e}")
        print(
            "Please ensure your Azure OpenAI environment variables (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT_NAME) are correctly set."
        )
        return None, None


def extract_predicates_from_text(
    text_description: str, client: AzureOpenAI, deployment_name: str
) -> list[str]:
    """
    Analyzes the behavior description using Azure GPT-4o to extract predicates.
    """

    if not client or not deployment_name:
        print("Azure OpenAI client not initialized. Skipping API call.")
        return []

    prompt = f"""You are an expert in linguistic analysis and game theory. Analyze the following player's behavior choice in a Prisoner's Dilemma scenario. Extract the core action predicates that describe the player's decision. Return these predicates as a JSON list of strings. e.g. {{'action_predicates': ['advise']}},  Behavior: '{text_description}'"""

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in linguistic analysis and game theory.",
                },
                {"role": "user", "content": prompt},
            ],
            model=deployment_name,
            response_format={"type": "json_object"},
            max_tokens=150,
        )
        # Assuming the response structure contains the JSON string in choices[0].message.content
        # You might need to adjust this based on the actual API response structure
        result_json_str = response.choices[0].message.content
        predicates = eval(result_json_str)["action_predicates"]
        if isinstance(predicates, list) and all(isinstance(p, str) for p in predicates):
            return predicates
        else:
            print(
                f"Warning: GPT-4o returned an unexpected format for predicates: {predicates}"
            )
            return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from GPT-4o response: {e}")
        print(f"Raw response content: {result_json_str}")
        return []
    except Exception as e:
        print(f"Error calling Azure GPT-4o: {e}")
        return []


# --- Data Loading and Processing ---
def load_data(json_path: str) -> list:
    """Loads data from the specified JSON file."""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        return []


def process_entry(entry_data, client, deployment_name):
    """Processes a single entry from the JSON data to extract predicates."""
    scenario_name = entry_data.get("scenario", "Unknown Scenario")
    behavior_choices = entry_data.get("behavior_choices", {})

    cooperate_text = behavior_choices.get("cooperate")
    defect_text = behavior_choices.get("defect")

    cooperate_predicates = []
    defect_predicates = []

    if cooperate_text:
        print(f"Analyzing COOPERATE for '{scenario_name}': {cooperate_text[:50]}...")
        cooperate_predicates = extract_predicates_from_text(
            cooperate_text, client, deployment_name
        )
    if defect_text:
        print(f"Analyzing DEFECT for '{scenario_name}': {defect_text[:50]}...")
        defect_predicates = extract_predicates_from_text(
            defect_text, client, deployment_name
        )

    return cooperate_predicates, defect_predicates


# --- Aggregation and Output ---
def generate_csv(cooperate_counts: Counter, defect_counts: Counter, csv_path: str):
    """Generates a CSV file summarizing predicate frequencies."""
    all_predicates = sorted(
        list(set(cooperate_counts.keys()) | set(defect_counts.keys()))
    )

    df_data = []
    for pred in all_predicates:
        df_data.append(
            {
                "predicate": pred,
                "cooperate_frequency": cooperate_counts.get(pred, 0),
                "defect_frequency": defect_counts.get(pred, 0),
            }
        )

    df = pd.DataFrame(df_data)
    try:
        df.to_csv(csv_path, index=False)
        print(f"Successfully generated CSV: {csv_path}")
    except Exception as e:
        print(f"Error writing CSV to {csv_path}: {e}")


def generate_pie_chart(
    predicate_counts: Counter, title: str, output_path: str, top_n=15
):
    """Generates a pie chart for predicate distribution."""
    if not predicate_counts:
        print(f"No data to generate pie chart: {title}")
        return

    # Select top N predicates and group others into "Others"
    if len(predicate_counts) > top_n:
        top_items = predicate_counts.most_common(top_n)
        others_count = sum(count for _, count in predicate_counts.most_common()[top_n:])
        labels = [item[0] for item in top_items] + ["Others"]
        sizes = [item[1] for item in top_items] + [others_count]
    else:
        labels = list(predicate_counts.keys())
        sizes = list(predicate_counts.values())

    if not sizes or sum(sizes) == 0:
        print(f"No valid data (all zero counts) to generate pie chart: {title}")
        return

    plt.figure(figsize=(10, 8))
    plt.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        textprops={"fontsize": 8},
    )
    plt.title(title)
    plt.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    try:
        plt.savefig(output_path)
        print(f"Successfully generated pie chart: {output_path}")
    except Exception as e:
        print(f"Error saving pie chart to {output_path}: {e}")
    plt.close()


# --- Main Execution ---
def main():
    print("Starting behavior analysis script...")

    azure_client, deployment_name = get_azure_openai_client()

    scenarios_data = load_data(INPUT_JSON_PATH)
    if not scenarios_data:
        print("No data loaded. Exiting.")
        return

    print(f"Loaded {len(scenarios_data)} scenarios from {INPUT_JSON_PATH}")

    all_cooperate_predicates = []
    all_defect_predicates = []

    # Determine number of workers, ensuring it's not more than available CPU cores or a sensible max
    max_workers = min(MAX_WORKERS_DEFAULT, os.cpu_count() or 1)

    print(f"Using up to {max_workers} parallel workers for API calls.")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map entries to futures
        futures = [
            executor.submit(process_entry, entry, azure_client, deployment_name)
            for entry in scenarios_data
        ]

        for i, future in enumerate(futures):
            try:
                coop_preds, def_preds = future.result()  # Wait for the result
                all_cooperate_predicates.extend(coop_preds)
                all_defect_predicates.extend(def_preds)
                if (i + 1) % 10 == 0 or (i + 1) == len(futures):  # Log progress
                    print(f"Processed {i+1}/{len(futures)} scenarios...")
            except Exception as e:
                print(f"Error processing an entry: {e}")

    print("All scenarios processed.")

    cooperate_counts = Counter(all_cooperate_predicates)
    defect_counts = Counter(all_defect_predicates)

    print("--- Analysis Summary ---")
    print(f"Total unique 'cooperate' predicates: {len(cooperate_counts)}")
    print(f"Total unique 'defect' predicates: {len(defect_counts)}")

    print("Top 5 Cooperate Predicates:")
    for pred, count in cooperate_counts.most_common(5):
        print(f"- {pred}: {count}")

    print("Top 5 Defect Predicates:")
    for pred, count in defect_counts.most_common(5):
        print(f"- {pred}: {count}")

    generate_csv(cooperate_counts, defect_counts, OUTPUT_CSV_PATH)
    generate_pie_chart(
        cooperate_counts,
        "Distribution of Cooperate Predicates",
        COOPERATE_PIE_CHART_PATH,
    )
    generate_pie_chart(
        defect_counts, "Distribution of Defect Predicates", DEFECT_PIE_CHART_PATH
    )

    print("Analysis complete. Outputs generated:")
    print(f"- CSV: {OUTPUT_CSV_PATH}")
    print(f"- Cooperate Pie Chart: {COOPERATE_PIE_CHART_PATH}")
    print(f"- Defect Pie Chart: {DEFECT_PIE_CHART_PATH}")


if __name__ == "__main__":
    main()
