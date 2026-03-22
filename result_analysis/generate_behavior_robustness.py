import pandas as pd
import numpy as np
from pathlib import Path

# Directory containing behavior robustness CSV files
heatmaps_dir = Path("results/game_theory/non-thinking/heatmaps")
output_dir = Path("results/game_theory/non-thinking/heatmaps")

# Find all behavior_robustness CSV files
behavior_files = sorted(heatmaps_dir.glob("behavior_robustness__*.csv"))

# Initialize list to store all data
all_data = []

# Model metadata mapping (architecture and parameters)
model_metadata = {
    "Llama-3.2-1B-Instruct": {"architecture": "Llama", "parameters": "1B", "model_id": "Llama 1B"},
    "Llama-3.2-3B-Instruct": {"architecture": "Llama", "parameters": "3B", "model_id": "Llama 3B"},
    "Phi-3.5-mini-instruct": {"architecture": "Phi", "parameters": "3.5-mini", "model_id": "Phi 3.5-mini"},
    "Phi-4-mini-instruct": {"architecture": "Phi", "parameters": "4-mini", "model_id": "Phi 4-mini"},
    "Qwen2.5-0.5B-Instruct": {"architecture": "Qwen", "parameters": "0.5B", "model_id": "Qwen 0.5B"},
    "Qwen2.5-1.5B-Instruct": {"architecture": "Qwen", "parameters": "1.5B", "model_id": "Qwen 1.5B"},
    "Qwen2.5-3B-Instruct": {"architecture": "Qwen", "parameters": "3B", "model_id": "Qwen 3B"},
    "Qwen3-0.6B": {"architecture": "Qwen", "parameters": "0.6B", "model_id": "Qwen 0.6B"},
    "Qwen3-4B": {"architecture": "Qwen", "parameters": "4B", "model_id": "Qwen 4B"},
    "gemma-3-1b-it": {"architecture": "Gemma", "parameters": "1b", "model_id": "Gemma 1b"},
    "gemma-3-4b-it": {"architecture": "Gemma", "parameters": "4b", "model_id": "Gemma 4b"},
    "gemma-3-270m-it": {"architecture": "gemma-3-270m-it", "parameters": "default", "model_id": "gemma-3-270m-it default"},
}

# Process each game file
for file in behavior_files:
    game_name = file.stem.replace("behavior_robustness__", "")
    print(f"Processing {game_name}...")
    
    df = pd.read_csv(file)
    
    # Calculate robustness = 1 - item_change_rate
    df['robustness'] = 1 - df['item_change_rate']
    
    # Group by model and calculate mean and std of robustness across emotions/behaviors
    grouped = df.groupby('model')['robustness'].agg(['std', 'mean']).reset_index()
    grouped.columns = ['model', 'std_robustness', 'mean_robustness']
    
    # Add metadata
    grouped['game'] = game_name
    grouped['architecture'] = grouped['model'].map(lambda x: model_metadata.get(x, {}).get('architecture', x))
    grouped['parameters'] = grouped['model'].map(lambda x: model_metadata.get(x, {}).get('parameters', 'default'))
    grouped['model_id'] = grouped['model'].map(lambda x: model_metadata.get(x, {}).get('model_id', x))
    
    all_data.append(grouped)

# Combine all data
data_df = pd.concat(all_data, ignore_index=True)

# Reorder columns to match emotion_robustness_data.csv
data_df = data_df[['model', 'architecture', 'parameters', 'game', 'std_robustness', 'mean_robustness', 'model_id']]

# Save data CSV
output_data_file = output_dir / "behavior_robustness_data.csv"
data_df.to_csv(output_data_file, index=False)
print(f"\nSaved: {output_data_file}")

# Create table CSV (pivot table with std_robustness values)
table_df = data_df.pivot_table(
    index=['architecture', 'parameters'],
    columns='game',
    values='std_robustness',
    aggfunc='first'
)

# Reset index to make architecture and parameters regular columns
table_df = table_df.reset_index()

# Sort games alphabetically in columns
game_columns = sorted([col for col in table_df.columns if col not in ['architecture', 'parameters']])
table_df = table_df[['architecture', 'parameters'] + game_columns]

# Save table CSV
output_table_file = output_dir / "behavior_robustness_table.csv"
table_df.to_csv(output_table_file, index=False)
print(f"Saved: {output_table_file}")

print("\nData summary:")
print(f"Total rows in data CSV: {len(data_df)}")
print(f"Total rows in table CSV: {len(table_df)}")
print(f"Games processed: {sorted(data_df['game'].unique())}")
