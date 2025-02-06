import json
import numpy as np
from scipy import stats
import pandas as pd
from typing import List, Dict, Tuple
from pathlib import Path
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

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
        """
        Perform chi-square test of independence between two conditions.
        If counts are too low or contain zeros, returns (0.0, 1.0) to indicate no significant difference.
        """
        # Create contingency table
        contingency = np.array([
            [condition1_counts['cooperate'], condition1_counts['defect']],
            [condition2_counts['cooperate'], condition2_counts['defect']]
        ])
        
        # Check if we have any zeros or very low counts
        if np.any(contingency == 0) or np.any(contingency < 5):
            # Return no-effect values when chi-square test is not appropriate
            return 0.0, 1.0
            
        try:
            chi2, p_value = stats.chi2_contingency(contingency)[:2]
            return chi2, p_value
        except ValueError:
            # Fallback in case of any other statistical issues
            return 0.0, 1.0
    
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
        
        contingency_array = np.array(contingency_matrix)
        
        # Check if we have any zeros or very low counts
        if np.any(contingency_array == 0) or np.any(contingency_array < 5):
            overall_chi2, overall_p_value = 0.0, 1.0
        else:
            try:
                overall_chi2, overall_p_value = stats.chi2_contingency(contingency_array)[:2]
            except ValueError:
                overall_chi2, overall_p_value = 0.0, 1.0
        
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

    def plot_behavior_comparison(self, results: Dict, output_path: str = None) -> None:
        """
        Generate visualization plots for behavior comparison across emotions.
        
        Args:
            results: Dictionary containing the statistical analysis results
            output_path: Optional path to save the plots (if None, plots will be displayed)
        """
        plt.style.use('seaborn-v0_8')
        
        # Create a figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)
        
        # 1. Bar plot for cooperation rates
        ax1 = fig.add_subplot(gs[0, 0])
        emotions = []
        coop_rates = []
        defect_rates = []
        
        for emotion, data in results['individual_conditions'].items():
            emotions.append(emotion)
            coop_rates.append(data['cooperative_ratio'])
            defect_rates.append(data['defective_ratio'])
        
        x = np.arange(len(emotions))
        width = 0.35
        
        ax1.bar(x - width/2, coop_rates, width, label='Cooperate', color='green', alpha=0.6)
        ax1.bar(x + width/2, defect_rates, width, label='Defect', color='red', alpha=0.6)
        ax1.set_ylabel('Ratio')
        ax1.set_title('Cooperation vs Defection Rates by Emotion')
        ax1.set_xticks(x)
        ax1.set_xticklabels(emotions)
        ax1.legend()
        
        # 2. Heatmap for pairwise p-values
        ax2 = fig.add_subplot(gs[0, 1])
        n_emotions = len(emotions)
        p_values_matrix = np.ones((n_emotions, n_emotions))
        
        for pair, stats in results['pairwise_comparisons'].items():
            em1, em2 = pair.split('_vs_')
            i, j = emotions.index(em1), emotions.index(em2)
            p_values_matrix[i, j] = stats['p_value']
            p_values_matrix[j, i] = stats['p_value']
        
        sns.heatmap(p_values_matrix, annot=True, cmap='RdYlBu_r', 
                    xticklabels=emotions, yticklabels=emotions, ax=ax2)
        ax2.set_title('P-values Heatmap for Pairwise Comparisons')
        
        # 3. Raw counts bar plot
        ax3 = fig.add_subplot(gs[1, :])
        data_counts = []
        for emotion in emotions:
            counts = results['individual_conditions'][emotion]['counts']
            data_counts.append([counts['cooperate'], counts['defect']])
        
        data_counts = np.array(data_counts)
        x = np.arange(len(emotions))
        width = 0.35
        
        ax3.bar(x - width/2, data_counts[:, 0], width, label='Cooperate', color='green', alpha=0.6)
        ax3.bar(x + width/2, data_counts[:, 1], width, label='Defect', color='red', alpha=0.6)
        ax3.set_ylabel('Count')
        ax3.set_title('Raw Behavior Counts by Emotion')
        ax3.set_xticks(x)
        ax3.set_xticklabels(emotions)
        ax3.legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()

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
        
    Returns:    engine = ExperimentEngine("/home/jjl7137/game_theory/config/priDeli_experiment_config.yaml")
        Dictionary containing comprehensive statistical analysis results
    """
    analyzer = BehaviorAnalyzer()
    results = analyzer.analyze_multiple_conditions(emotion_files) 
    analyzer.plot_behavior_comparison(results, 'behavior_analysis.png')
    return results

if __name__ == "__main__":
    # Example usage
    emotion_files = {
        'anger': '/home/jjl7137/game_theory/results/Stag_Hunt/Stag_Hunt_angry_results.json',
        'happy': '/home/jjl7137/game_theory/results/Stag_Hunt/Stag_Hunt_happiness_results.json',
        'sad': '/home/jjl7137/game_theory/results/Stag_Hunt/Stag_Hunt_sadness_results.json'
    }
    

    results = compare_multiple_emotions(emotion_files)
    from pprint import pprint
    pprint(results)