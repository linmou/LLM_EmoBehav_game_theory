#!/usr/bin/env python3
"""
Script to analyze the ratio of samples containing '</think>' in 'output' 
that have category 2 in exp_results.json files.
"""

import json
import glob
from pathlib import Path
from typing import List, Dict, Any

def analyze_thinking_mode_ratio(pattern: str) -> Dict[str, Any]:
    """
    Analyze the ratio of samples with '</think>' in output that have category 2.
    
    Args:
        pattern: Glob pattern to find exp_results.json files
        
    Returns:
        Dictionary with analysis results
    """
    
    # Find all exp_results.json files matching the pattern
    json_files = glob.glob(pattern)
    
    if not json_files:
        print(f"No files found matching pattern: {pattern}")
        return {}
    
    print(f"Found {len(json_files)} files to analyze:")
    for file in json_files:
        print(f"  - {file}")
    print()
    
    # Aggregate results across all files
    total_samples = 0
    thinking_samples = 0
    thinking_category_2 = 0
    
    file_results = {}
    
    for json_file in json_files:
        print(f"Analyzing: {json_file}")
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            file_total = len(data)
            file_thinking = 0
            file_thinking_cat2 = 0
            
            for sample in data:
                total_samples += 1
                
                # Check if output contains '</think>'
                output = sample.get('output', '')
                if '</think>' in output:
                    thinking_samples += 1
                    file_thinking += 1
                    
                    # Check if category is 2
                    category = sample.get('category')
                    if category == 2:
                        thinking_category_2 += 1
                        file_thinking_cat2 += 1
            
            # Calculate ratio for this file
            file_ratio = file_thinking_cat2 / file_thinking if file_thinking > 0 else 0
            
            file_results[json_file] = {
                'total_samples': file_total,
                'thinking_samples': file_thinking,
                'thinking_category_2': file_thinking_cat2,
                'ratio': file_ratio
            }
            
            print(f"  Total samples: {file_total}")
            print(f"  Samples with </think>: {file_thinking}")
            print(f"  Thinking samples with category 2: {file_thinking_cat2}")
            print(f"  Ratio (category 2 / thinking): {file_ratio:.4f}")
            print()
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    # Calculate overall ratio
    overall_ratio = thinking_category_2 / thinking_samples if thinking_samples > 0 else 0
    
    results = {
        'total_files': len(json_files),
        'total_samples': total_samples,
        'thinking_samples': thinking_samples,
        'thinking_category_2': thinking_category_2,
        'overall_ratio': overall_ratio,
        'file_results': file_results
    }
    
    return results

def main():
    # Pattern to match the specified files
    pattern = "results/Qwen3_Thinking_Mode_Comparison_Baseline/Qwen3_Thinking_Mode_Comparison_Baseline_Prisoners_Dilemma_Qwen3-*/exp_results.json"
    
    print("=" * 80)
    print("ANALYZING THINKING MODE RATIO")
    print("=" * 80)
    print(f"Pattern: {pattern}")
    print()
    
    results = analyze_thinking_mode_ratio(pattern)
    
    if results:
        print("=" * 80)
        print("SUMMARY RESULTS")
        print("=" * 80)
        print(f"Total files analyzed: {results['total_files']}")
        print(f"Total samples across all files: {results['total_samples']}")
        print(f"Samples containing '</think>': {results['thinking_samples']}")
        print(f"Thinking samples with category 2: {results['thinking_category_2']}")
        print(f"Overall ratio (category 2 / thinking): {results['overall_ratio']:.4f}")
        print(f"Percentage: {results['overall_ratio'] * 100:.2f}%")
        print()
        
        # Save results to JSON file
        output_file = "thinking_mode_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()