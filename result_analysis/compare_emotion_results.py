#!/usr/bin/env python3
"""
Compare results from happiness and anger emotion experiments.
This script analyzes cooperation rates and performs statistical tests.
"""

import json
import sys
from pathlib import Path
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(results_file):
    """Load results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def analyze_cooperation_rates(results):
    """Calculate cooperation rate from results."""
    total = len(results)
    cooperate_count = sum(1 for r in results if r.get('category') == 'cooperate')
    defect_count = sum(1 for r in results if r.get('category') == 'defect')
    
    return {
        'total': total,
        'cooperate': cooperate_count,
        'defect': defect_count,
        'cooperation_rate': cooperate_count / total if total > 0 else 0,
        'defection_rate': defect_count / total if total > 0 else 0
    }


def compare_emotions(happiness_file, anger_file, output_dir=None):
    """Compare happiness vs anger results with statistical tests."""
    
    # Load results
    print("Loading results...")
    happiness_results = load_results(happiness_file)
    anger_results = load_results(anger_file)
    
    # Analyze cooperation rates
    happiness_stats = analyze_cooperation_rates(happiness_results)
    anger_stats = analyze_cooperation_rates(anger_results)
    
    print("\n=== EMOTION COMPARISON RESULTS ===")
    print(f"\nHappiness Condition:")
    print(f"  Total decisions: {happiness_stats['total']}")
    print(f"  Cooperation rate: {happiness_stats['cooperation_rate']:.2%}")
    print(f"  Defection rate: {happiness_stats['defection_rate']:.2%}")
    
    print(f"\nAnger Condition:")
    print(f"  Total decisions: {anger_stats['total']}")
    print(f"  Cooperation rate: {anger_stats['cooperation_rate']:.2%}")
    print(f"  Defection rate: {anger_stats['defection_rate']:.2%}")
    
    # Chi-square test for independence
    observed = [
        [happiness_stats['cooperate'], happiness_stats['defect']],
        [anger_stats['cooperate'], anger_stats['defect']]
    ]
    chi2, p_value, dof, expected = stats.chi2_contingency(observed)
    
    # Effect size (Cramer's V)
    n = sum(sum(row) for row in observed)
    cramers_v = (chi2 / n) ** 0.5
    
    print(f"\n=== STATISTICAL ANALYSIS ===")
    print(f"Chi-square test:")
    print(f"  χ² = {chi2:.3f}")
    print(f"  p-value = {p_value:.6f}")
    print(f"  Effect size (Cramer's V) = {cramers_v:.3f}")
    
    if p_value < 0.05:
        print(f"  Result: SIGNIFICANT difference (p < 0.05)")
        if happiness_stats['cooperation_rate'] > anger_stats['cooperation_rate']:
            print(f"  Direction: Happiness → MORE cooperation")
        else:
            print(f"  Direction: Anger → MORE cooperation")
    else:
        print(f"  Result: No significant difference (p >= 0.05)")
    
    # Create visualization
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        conditions = ['Happiness', 'Anger']
        cooperation_rates = [happiness_stats['cooperation_rate'], anger_stats['cooperation_rate']]
        
        bars = ax.bar(conditions, cooperation_rates, color=['#4CAF50', '#F44336'])
        ax.set_ylabel('Cooperation Rate', fontsize=12)
        ax.set_title('Cooperation Rates: Happiness vs Anger', fontsize=14)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, rate in zip(bars, cooperation_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{rate:.1%}', ha='center', va='bottom')
        
        # Add significance annotation
        if p_value < 0.05:
            ax.text(0.5, 0.9, f'p = {p_value:.4f} (significant)', 
                   transform=ax.transAxes, ha='center', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_path / 'emotion_comparison.png', dpi=300)
        print(f"\nVisualization saved to: {output_path / 'emotion_comparison.png'}")
    
    # Save detailed results
    if output_dir:
        results_summary = {
            'happiness': happiness_stats,
            'anger': anger_stats,
            'statistical_test': {
                'chi_square': float(chi2),
                'p_value': float(p_value),
                'cramers_v': float(cramers_v),
                'significant': bool(p_value < 0.05)
            }
        }
        
        with open(output_path / 'comparison_summary.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"Summary saved to: {output_path / 'comparison_summary.json'}")
    
    return {
        'happiness_stats': happiness_stats,
        'anger_stats': anger_stats,
        'chi2': chi2,
        'p_value': p_value,
        'cramers_v': cramers_v
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_emotion_results.py <happiness_results.json> <anger_results.json> [output_dir]")
        sys.exit(1)
    
    happiness_file = sys.argv[1]
    anger_file = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    compare_emotions(happiness_file, anger_file, output_dir)