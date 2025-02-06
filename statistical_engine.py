import json
import numpy as np
from scipy import stats
import pandas as pd
from typing import List, Dict, Tuple
from pathlib import Path
from itertools import combinations

class BehaviorAnalyzer:
    def __init__(self):
        pass
    
    def load_json_data(self, file_path: str) -> List[Dict]:
        """Load data from a JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def extract_behavior_counts(self, data: List[Dict]) -> Dict[str, int]:
        """Extract counts of cooperative and defective behaviors."""
        behavior_counts = {'cooperate': 0, 'defect': 0}
        for item in data:
            category = item.get('category', '').lower()
            if category in behavior_counts:
                behavior_counts[category] += 1
        return behavior_counts
    
    def calculate_behavior_ratio(self, counts: Dict[str, int]) -> Tuple[float, float]:
        """Calculate the ratio of cooperative to defective behaviors."""
        total = sum(counts.values())
        if total == 0:
            return 0.0, 0.0
        coop_ratio = counts['cooperate'] / total
        defect_ratio = counts['defect'] / total
        return coop_ratio, defect_ratio
    
    def chi_square_test(self, condition1_counts: Dict[str, int], 
                       condition2_counts: Dict[str, int]) -> Tuple[float, float]:
        """Perform chi-square test of independence between two conditions."""
        contingency = np.array([
            [condition1_counts['cooperate'], condition1_counts['defect']],
            [condition2_counts['cooperate'], condition2_counts['defect']]
        ])
        chi2, p_value = stats.chi2_contingency(contingency)[:2]
        return chi2, p_value
    
    def analyze_conditions(self, file_path1: str, file_path2: str) -> Dict:
        """
        Analyze behavioral differences between two conditions.
        
        Args:
            file_path1: Path to first condition's JSON file
            file_path2: Path to second condition's JSON file
            
        Returns:
            Dictionary containing statistical analysis results
        """
        # Load data
        data1 = self.load_json_data(file_path1)
        data2 = self.load_json_data(file_path2)
        
        # Get behavior counts
        counts1 = self.extract_behavior_counts(data1)
        counts2 = self.extract_behavior_counts(data2)
        
        # Calculate ratios
        coop_ratio1, defect_ratio1 = self.calculate_behavior_ratio(counts1)
        coop_ratio2, defect_ratio2 = self.calculate_behavior_ratio(counts2)
        
        # Perform statistical test
        chi2, p_value = self.chi_square_test(counts1, counts2)
        
        # Prepare results
        results = {
            'condition1': {
                'counts': counts1,
                'cooperative_ratio': coop_ratio1,
                'defective_ratio': defect_ratio1
            },
            'condition2': {
                'counts': counts2,
                'cooperative_ratio': coop_ratio2,
                'defective_ratio': defect_ratio2
            },
            'statistical_test': {
                'chi_square': chi2,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        }
        
        return results

    def generate_text_description(self, results: Dict) -> str:
        """
        Generate a human-readable description of the statistical analysis results.
        
        Args:
            results: Dictionary containing the statistical analysis results
            
        Returns:
            A formatted string describing the results
        """
        description = []
        
        # Overall summary
        description.append("Statistical Analysis Summary:")
        description.append("=" * 30)
        
        # Individual conditions
        description.append("\n1. Individual Emotion Conditions:")
        for emotion, data in results['individual_conditions'].items():
            description.append(f"\n{emotion.capitalize()}:")
            description.append(f"- Cooperative decisions: {data['counts']['cooperate']} ({data['cooperative_ratio']*100:.1f}%)")
            description.append(f"- Defective decisions: {data['counts']['defect']} ({data['defective_ratio']*100:.1f}%)")
        
        # Overall test results
        description.append("\n2. Overall Statistical Test:")
        overall_test = results['overall_test']
        description.append(f"Chi-square value: {overall_test['chi_square']:.2f}")
        description.append(f"P-value: {overall_test['p_value']:.4f}")
        
        if overall_test['significant']:
            description.append("Conclusion: There are statistically significant differences in behavior across emotions (p < 0.05)")
        else:
            description.append("Conclusion: No statistically significant differences in behavior across emotions (p ≥ 0.05)")
        
        # Pairwise comparisons
        description.append("\n3. Pairwise Comparisons:")
        significant_pairs = []
        non_significant_pairs = []
        
        for comparison, stats in results['pairwise_comparisons'].items():
            emotions = comparison.replace('_vs_', ' vs ')
            if stats['significant']:
                significant_pairs.append(f"- {emotions} (p={stats['p_value']:.4f})")
            else:
                non_significant_pairs.append(f"- {emotions} (p={stats['p_value']:.4f})")
        
        if significant_pairs:
            description.append("\nSignificant differences found between:")
            description.extend(significant_pairs)
        
        if non_significant_pairs:
            description.append("\nNo significant differences found between:")
            description.extend(non_significant_pairs)
        
        return "\n".join(description)

    def analyze_multiple_conditions(self, emotion_files: Dict[str, str]) -> Dict:
        """
        Analyze behavioral differences across multiple emotion conditions.
        
        Args:
            emotion_files: Dictionary mapping emotion names to their JSON file paths
                         e.g., {'happy': 'happy.json', 'angry': 'angry.json', ...}
            
        Returns:
            Dictionary containing comprehensive statistical analysis results
        """
        # Store data for each emotion
        emotion_data = {}
        for emotion, file_path in emotion_files.items():
            data = self.load_json_data(file_path)
            counts = self.extract_behavior_counts(data)
            coop_ratio, defect_ratio = self.calculate_behavior_ratio(counts)
            
            emotion_data[emotion] = {
                'counts': counts,
                'cooperative_ratio': coop_ratio,
                'defective_ratio': defect_ratio
            }
        
        # Perform overall chi-square test
        contingency_matrix = []
        for emotion in emotion_data:
            counts = emotion_data[emotion]['counts']
            contingency_matrix.append([counts['cooperate'], counts['defect']])
        
        overall_chi2, overall_p_value = stats.chi2_contingency(np.array(contingency_matrix))[:2]
        
        # Perform pairwise comparisons
        pairwise_comparisons = {}
        for (emotion1, emotion2) in combinations(emotion_data.keys(), 2):
            chi2, p_value = self.chi_square_test(
                emotion_data[emotion1]['counts'],
                emotion_data[emotion2]['counts']
            )
            pairwise_comparisons[f"{emotion1}_vs_{emotion2}"] = {
                'chi_square': chi2,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        # Prepare comprehensive results
        results = {
            'individual_conditions': emotion_data,
            'overall_test': {
                'chi_square': overall_chi2,
                'p_value': overall_p_value,
                'significant': overall_p_value < 0.05
            },
            'pairwise_comparisons': pairwise_comparisons
        }
        
        # Add text description
        results['text_description'] = self.generate_text_description(results)
        
        return results

def compare_behavior_conditions(file_path1: str, file_path2: str) -> Dict:
    """
    Convenience function to compare behavior between two conditions.
    
    Args:
        file_path1: Path to first condition's JSON file
        file_path2: Path to second condition's JSON file
        
    Returns:
        Dictionary containing statistical analysis results
    """
    analyzer = BehaviorAnalyzer()
    return analyzer.analyze_conditions(file_path1, file_path2)

def compare_multiple_emotions(emotion_files: Dict[str, str]) -> Dict:
    """
    Convenience function to compare behavior across multiple emotion conditions.
    
    Args:
        emotion_files: Dictionary mapping emotion names to their JSON file paths
                      e.g., {'happy': 'happy.json', 'angry': 'angry.json', ...}
        
    Returns:
        Dictionary containing comprehensive statistical analysis results
    """
    analyzer = BehaviorAnalyzer()
    return analyzer.analyze_multiple_conditions(emotion_files) 