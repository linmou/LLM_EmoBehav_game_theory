#!/usr/bin/env python3
"""
Analyze Choice Selection patterns to understand how choices differ between conditions.
"""

import json
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

def load_results(file_path: str) -> List[Dict]:
    """Load the JSON results file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_choice_patterns(results: List[Dict]) -> None:
    """Analyze overall choice patterns between conditions."""
    
    # Group by condition
    by_condition = defaultdict(list)
    for result in results:
        by_condition[result['condition_name']].append(result)
    
    print("\n=== Overall Statistics ===")
    print(f"Total results: {len(results)}")
    for condition, items in by_condition.items():
        print(f"{condition}: {len(items)} results")
    
    # Analyze choice distribution by condition
    print("\n=== Choice Distribution by Condition ===")
    for condition, items in by_condition.items():
        choices = Counter([item['chosen_option_id'] for item in items if item['chosen_option_id'] is not None])
        print(f"\n{condition}:")
        for choice, count in sorted(choices.items()):
            percentage = (count / len(items)) * 100
            print(f"  Option {choice}: {count} ({percentage:.1f}%)")
    
    # Analyze by emotion within each condition
    print("\n=== Choice Distribution by Emotion and Condition ===")
    emotion_condition_choices = defaultdict(lambda: defaultdict(Counter))
    
    for result in results:
        emotion = result['activation_emotion']
        condition = result['condition_name']
        choice = result['chosen_option_id']
        if choice is not None:
            emotion_condition_choices[emotion][condition][choice] += 1
    
    for emotion in sorted(emotion_condition_choices.keys()):
        print(f"\n{emotion}:")
        for condition in sorted(emotion_condition_choices[emotion].keys()):
            choices = emotion_condition_choices[emotion][condition]
            total = sum(choices.values())
            print(f"  {condition}:")
            for choice in sorted(choices.keys()):
                count = choices[choice]
                percentage = (count / total) * 100
                print(f"    Option {choice}: {count} ({percentage:.1f}%)")

def find_switching_patterns(results: List[Dict]) -> pd.DataFrame:
    """Find cases where the same scenario switches choices between conditions."""
    
    # Group by scenario, emotion, and batch_idx
    grouped = defaultdict(dict)
    
    for result in results:
        key = (result['scenario'], result['activation_emotion'], result['batch_idx'])
        condition = result['condition_name']
        grouped[key][condition] = result
    
    # Find switches
    switches = []
    
    for key, conditions in grouped.items():
        if len(conditions) < 2:
            continue
        
        # Get choices for each condition
        choices = {cond: res['chosen_option_id'] for cond, res in conditions.items()}
        
        # Check for any differences
        if len(set(choices.values())) > 1:
            scenario, emotion, batch_idx = key
            switches.append({
                'scenario': scenario,
                'emotion': emotion,
                'batch_idx': batch_idx,
                **{f'{cond}_choice': choice for cond, choice in choices.items()}
            })
    
    return pd.DataFrame(switches)

def analyze_specific_switches(results: List[Dict]) -> None:
    """Analyze specific switching patterns between activation_only and context_and_activation."""
    
    # Group by scenario, emotion, and batch_idx
    grouped = defaultdict(dict)
    
    for result in results:
        if result['condition_name'] in ['activation_only', 'context_and_activation']:
            key = (result['scenario'], result['activation_emotion'], result['batch_idx'])
            condition = result['condition_name']
            grouped[key][condition] = result
    
    # Count different switching patterns
    switch_patterns = Counter()
    switch_details = []
    
    for key, conditions in grouped.items():
        if 'activation_only' in conditions and 'context_and_activation' in conditions:
            ao_choice = conditions['activation_only']['chosen_option_id']
            ca_choice = conditions['context_and_activation']['chosen_option_id']
            
            pattern = f"activation_only:{ao_choice} -> context_and_activation:{ca_choice}"
            switch_patterns[pattern] += 1
            
            if ao_choice != ca_choice:
                scenario, emotion, batch_idx = key
                switch_details.append({
                    'scenario': scenario,
                    'emotion': emotion,
                    'batch_idx': batch_idx,
                    'ao_choice': ao_choice,
                    'ca_choice': ca_choice,
                    'pattern': pattern
                })
    
    print("\n=== Switching Patterns: activation_only vs context_and_activation ===")
    for pattern, count in sorted(switch_patterns.items(), key=lambda x: -x[1]):
        print(f"{pattern}: {count} cases")
    
    # Analyze switches by emotion
    if switch_details:
        df = pd.DataFrame(switch_details)
        print("\n=== Switches by Emotion ===")
        emotion_switches = df['emotion'].value_counts()
        for emotion, count in emotion_switches.items():
            print(f"{emotion}: {count} switches")
        
        # Show some examples
        print("\n=== Example Switches ===")
        for pattern in ['activation_only:1 -> context_and_activation:2', 
                       'activation_only:2 -> context_and_activation:1']:
            examples = df[df['pattern'] == pattern]
            if len(examples) > 0:
                print(f"\n{pattern} ({len(examples)} total):")
                for idx, row in examples.head(2).iterrows():
                    print(f"  - {row['scenario']} | {row['emotion']} | batch {row['batch_idx']}")

def main():
    # File path
    file_path = "results/Choice_Selection/choice_selection_choice_selection_context_activation_test_prisoners_dilemma_Qwen2.5-3B-Instruct_20250618_223246/choice_selection_results.json"
    
    print("Loading results...")
    results = load_results(file_path)
    
    # Overall analysis
    analyze_choice_patterns(results)
    
    # Find switching patterns
    print("\n" + "="*50)
    switches_df = find_switching_patterns(results)
    print(f"\nFound {len(switches_df)} cases where choices switch between conditions")
    
    if len(switches_df) > 0:
        switches_df.to_csv("choice_switches_all_conditions.csv", index=False)
        print("Saved detailed switches to choice_switches_all_conditions.csv")
    
    # Specific analysis for activation_only vs context_and_activation
    print("\n" + "="*50)
    analyze_specific_switches(results)

if __name__ == "__main__":
    main()