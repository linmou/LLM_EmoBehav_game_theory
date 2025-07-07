#!/usr/bin/env python3
"""
Debug the data structure to understand why matching isn't working.
"""

import json
from collections import Counter

def load_results(file_path: str):
    """Load the JSON results file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def main():
    file_path = "results/Choice_Selection/choice_selection_choice_selection_context_activation_test_prisoners_dilemma_Qwen2.5-3B-Instruct_20250618_223246/choice_selection_results.json"
    
    print("Loading results...")
    results = load_results(file_path)
    
    # Check first few entries
    print("\n=== First 3 entries ===")
    for i, result in enumerate(results[:3]):
        print(f"\nEntry {i}:")
        print(f"  Condition: {result['condition_name']}")
        print(f"  Scenario: {result['scenario'][:50]}...")
        print(f"  Batch idx: {result['batch_idx']}")
        print(f"  Emotion: {result['activation_emotion']}")
        print(f"  Choice: {result['chosen_option_id']}")
    
    # Check unique scenarios
    scenarios = set(r['scenario'] for r in results)
    print(f"\n=== Unique scenarios: {len(scenarios)} ===")
    
    # Check batch indices
    batch_indices = Counter(r['batch_idx'] for r in results)
    print(f"\n=== Batch index distribution ===")
    print(f"Min batch_idx: {min(batch_indices.keys())}")
    print(f"Max batch_idx: {max(batch_indices.keys())}")
    print(f"Number of unique batch indices: {len(batch_indices)}")
    
    # Check how many times each batch_idx appears
    batch_count_dist = Counter(batch_indices.values())
    print(f"\n=== Batch count distribution ===")
    for count, freq in sorted(batch_count_dist.items()):
        print(f"  Batch indices appearing {count} times: {freq}")
    
    # Check specific batch_idx=0 across conditions
    print(f"\n=== Entries with batch_idx=0 ===")
    batch_0_entries = [r for r in results if r['batch_idx'] == 0]
    for entry in batch_0_entries[:8]:  # Show first 8
        print(f"  {entry['condition_name']:25} | {entry['scenario'][:40]}... | Choice: {entry['chosen_option_id']}")
    
    # Check if scenarios repeat across conditions
    print(f"\n=== Checking scenario repetition across conditions ===")
    scenario_condition_map = {}
    for r in results:
        key = (r['scenario'], r['batch_idx'])
        if key not in scenario_condition_map:
            scenario_condition_map[key] = []
        scenario_condition_map[key].append(r['condition_name'])
    
    # Count how many scenarios appear in multiple conditions
    multi_condition_count = 0
    for key, conditions in scenario_condition_map.items():
        if len(set(conditions)) > 1:
            multi_condition_count += 1
    
    print(f"Scenarios appearing in multiple conditions: {multi_condition_count}")
    
    # Show an example
    for key, conditions in scenario_condition_map.items():
        if 'activation_only' in conditions and 'context_and_activation' in conditions:
            print(f"\nExample scenario in both conditions:")
            print(f"  Scenario: {key[0][:60]}...")
            print(f"  Batch idx: {key[1]}")
            print(f"  Conditions: {set(conditions)}")
            break

if __name__ == "__main__":
    main()