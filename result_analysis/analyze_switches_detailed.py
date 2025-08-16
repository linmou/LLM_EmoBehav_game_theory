#!/usr/bin/env python3
"""
Detailed analysis of choice switches between activation_only and context_and_activation conditions.
"""

import json
import pandas as pd
from collections import defaultdict, Counter

def load_results(file_path: str):
    """Load the JSON results file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def main():
    file_path = "results/Choice_Selection/choice_selection_choice_selection_context_activation_test_prisoners_dilemma_Qwen2.5-3B-Instruct_20250703_133259/choice_selection_results.json"
    
    print("Loading results...")
    results = load_results(file_path)
    
    # Create a mapping of (scenario, batch_idx) -> condition -> result
    scenario_map = defaultdict(dict)
    
    for result in results:
        if result['condition_name'] in ['activation_only', 'context_and_activation']:
            # Use scenario description to ensure proper matching
            key = (result['scenario'], result['batch_idx'])
            scenario_map[key][result['condition_name']] = result
    
    # Find all pairs and analyze switches
    total_pairs = 0
    switches = []
    switch_types = Counter()
    
    for key, conditions in scenario_map.items():
        if 'activation_only' in conditions and 'context_and_activation' in conditions:
            total_pairs += 1
            
            ao = conditions['activation_only']
            ca = conditions['context_and_activation']
            
            ao_choice = ao['chosen_option_id']
            ca_choice = ca['chosen_option_id']
            
            # Skip if either choice is None
            if ao_choice is None or ca_choice is None:
                continue
            
            # Record the switch pattern
            pattern = f"{ao_choice} -> {ca_choice}"
            switch_types[pattern] += 1
            
            # If choices differ, record the switch
            if ao_choice != ca_choice:
                switches.append({
                    'scenario': ao['scenario'],
                    'batch_idx': ao['batch_idx'],
                    'emotion': ao['activation_emotion'],
                    'intensity': ao['activation_intensity'],
                    'ao_choice': ao_choice,
                    'ca_choice': ca_choice,
                    'switch_type': pattern,
                    'ao_text': ao['chosen_option_text'],
                    'ca_text': ca['chosen_option_text'],
                    'ao_prompt': ao.get('prompt', ''),
                    'ca_prompt': ca.get('prompt', ''),
                    'generated_text_ao': ao.get('generated_text', ''),
                    'generated_text_ca': ca.get('generated_text', '')
                })
    
    print(f"\nTotal scenario pairs analyzed: {total_pairs}")
    print(f"\nSwitch patterns:")
    for pattern, count in sorted(switch_types.items()):
        percentage = (count / total_pairs) * 100
        print(f"  {pattern}: {count} ({percentage:.1f}%)")
    
    print(f"\nTotal switches (where choice changed): {len(switches)}")
    
    if switches:
        # Analyze switch directions
        switch_1_to_2 = [s for s in switches if s['switch_type'] == '1 -> 2']
        switch_2_to_1 = [s for s in switches if s['switch_type'] == '2 -> 1']
        
        print(f"\nSwitch directions:")
        print(f"  1 -> 2 (cooperate to defect): {len(switch_1_to_2)}")
        print(f"  2 -> 1 (defect to cooperate): {len(switch_2_to_1)}")
        
        # Save detailed results
        df = pd.DataFrame(switches)
        df.to_csv("detailed_switches_with_prompts.csv", index=False)
        print(f"\nDetailed switches saved to detailed_switches_with_prompts.csv")
        
        # Show examples of 1->2 switches
        if switch_1_to_2:
            print(f"\n=== Examples of 1->2 switches (cooperate to defect) ===")
            for i, switch in enumerate(switch_1_to_2[:3]):
                print(f"\nExample {i+1}:")
                print(f"  Scenario: {switch['scenario']}")
                print(f"  Emotion: {switch['emotion']} (intensity: {switch['intensity']})")
                print(f"  Activation only chose: {switch['ao_text']}")
                print(f"  Context+Activation chose: {switch['ca_text']}")
        
        # Show examples of 2->1 switches
        if switch_2_to_1:
            print(f"\n=== Examples of 2->1 switches (defect to cooperate) ===")
            for i, switch in enumerate(switch_2_to_1[:3]):
                print(f"\nExample {i+1}:")
                print(f"  Scenario: {switch['scenario']}")
                print(f"  Emotion: {switch['emotion']} (intensity: {switch['intensity']})")
                print(f"  Activation only chose: {switch['ao_text']}")
                print(f"  Context+Activation chose: {switch['ca_text']}")
    
    # Additional analysis: check consistency
    print("\n=== Data Consistency Check ===")
    
    # Count conditions
    condition_counts = Counter(r['condition_name'] for r in results)
    for cond, count in condition_counts.items():
        print(f"{cond}: {count} results")
    
    # Check emotion distribution
    emotion_counts = Counter()
    for r in results:
        if r['condition_name'] in ['activation_only', 'context_and_activation']:
            emotion_counts[r['activation_emotion']] += 1
    print(f"\nEmotion distribution in activation conditions:")
    for emotion, count in emotion_counts.items():
        print(f"  {emotion}: {count}")

if __name__ == "__main__":
    main()