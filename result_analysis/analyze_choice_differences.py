#!/usr/bin/env python3
"""
Analyze Choice Selection results to find cases where:
- activation_only condition selects option 1
- context_and_activation condition selects option 2
"""

import json
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple

def load_results(file_path: str) -> List[Dict]:
    """Load the JSON results file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def group_by_scenario_and_emotion(results: List[Dict]) -> Dict[Tuple[str, str], Dict[str, List[Dict]]]:
    """Group results by scenario and emotion, then by condition."""
    grouped = defaultdict(lambda: defaultdict(list))
    
    for result in results:
        scenario = result['scenario']
        emotion = result['activation_emotion']
        condition = result['condition_name']
        
        grouped[(scenario, emotion)][condition].append(result)
    
    return grouped

def analyze_choice_differences(results: List[Dict]) -> pd.DataFrame:
    """Find cases where activation_only chooses option 1 and context_and_activation chooses option 2."""
    grouped = group_by_scenario_and_emotion(results)
    
    differences = []
    
    for (scenario, emotion), conditions in grouped.items():
        # Skip if we don't have both conditions
        if 'activation_only' not in conditions or 'context_and_activation' not in conditions:
            continue
        
        # Check each batch
        for batch_idx in range(len(conditions['activation_only'])):
            activation_only = conditions['activation_only'][batch_idx]
            
            # Find corresponding context_and_activation result
            context_activation = None
            for ca in conditions['context_and_activation']:
                if ca['batch_idx'] == activation_only['batch_idx']:
                    context_activation = ca
                    break
            
            if context_activation is None:
                continue
            
            # Check if activation_only chose option 1 and context_and_activation chose option 2
            if (activation_only['chosen_option_id'] == 1 and 
                context_activation['chosen_option_id'] == 2):
                
                differences.append({
                    'scenario': scenario,
                    'emotion': emotion,
                    'intensity': activation_only['activation_intensity'],
                    'batch_idx': batch_idx,
                    'activation_only_choice': activation_only['chosen_option_id'],
                    'activation_only_text': activation_only['chosen_option_text'],
                    'activation_only_rational': activation_only['generated_text'].split("'rational': '")[1].split("',")[0] if "'rational': '" in activation_only['generated_text'] else "N/A",
                    'context_activation_choice': context_activation['chosen_option_id'],
                    'context_activation_text': context_activation['chosen_option_text'],
                    'context_activation_rational': context_activation['generated_text'].split("'rational': '")[1].split("',")[0] if "'rational': '" in context_activation['generated_text'] else "N/A"
                })
    
    return pd.DataFrame(differences)

def main():
    # File path
    file_path = "results/Choice_Selection/choice_selection_choice_selection_context_activation_test_prisoners_dilemma_Qwen2.5-3B-Instruct_20250618_223246/choice_selection_results.json"
    
    print("Loading results...")
    results = load_results(file_path)
    print(f"Loaded {len(results)} results")
    
    # Analyze differences
    print("\nAnalyzing choice differences...")
    differences_df = analyze_choice_differences(results)
    
    if len(differences_df) == 0:
        print("No cases found where activation_only chose option 1 and context_and_activation chose option 2")
        return
    
    print(f"\nFound {len(differences_df)} cases where choices differ as specified")
    
    # Group by emotion
    print("\nBreakdown by emotion:")
    emotion_counts = differences_df['emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        print(f"  {emotion}: {count} cases")
    
    # Group by scenario
    print("\nBreakdown by scenario:")
    scenario_counts = differences_df['scenario'].value_counts()
    for scenario, count in scenario_counts.items():
        print(f"  {scenario}: {count} cases")
    
    # Save detailed results
    output_file = "choice_difference_analysis.csv"
    differences_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to {output_file}")
    
    # Show a few examples
    print("\nExample cases:")
    for idx, row in differences_df.head(3).iterrows():
        print(f"\n--- Example {idx + 1} ---")
        print(f"Scenario: {row['scenario']}")
        print(f"Emotion: {row['emotion']} (intensity: {row['intensity']})")
        print(f"Activation only chose: {row['activation_only_text']}")
        print(f"  Rational: {row['activation_only_rational'][:100]}...")
        print(f"Context+Activation chose: {row['context_activation_text']}")
        print(f"  Rational: {row['context_activation_rational'][:100]}...")

if __name__ == "__main__":
    main()