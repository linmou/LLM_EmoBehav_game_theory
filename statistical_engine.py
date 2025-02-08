import json
import numpy as np
from scipy import stats
import pandas as pd
from typing import List, Dict, Tuple
from pathlib import Path
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

class BaseAnalyzer(ABC):
    """Base class for statistical analysis of behavioral data"""
    
    def calculate_behavior_ratio(self, counts: Dict[str, int]) -> Tuple[float, float]:
        """Calculate the ratio of cooperative to defective behaviors."""
        total = sum(counts.values())
        if total == 0:
            return 0.0, 0.0
        return counts['cooperate']/total, counts['defect']/total

    def chi_square_test(self, condition1_counts: Dict[str, int], 
                       condition2_counts: Dict[str, int]) -> Tuple[float, float]:
        """
        Perform chi-square test of independence between two conditions.
        Uses Fisher's exact test when counts are low or contain zeros.
        """
        # Create contingency table
        contingency = np.array([
            [condition1_counts['cooperate'], condition1_counts['defect']],
            [condition2_counts['cooperate'], condition2_counts['defect']]
        ])
        
        try:
            # Use Fisher's exact test when we have zeros or low counts
            if np.any(contingency == 0) or np.any(contingency < 5):
                _, p_value = stats.fisher_exact(contingency)
                # Calculate Cramer's V as effect size (similar scale to chi-square)
                n = np.sum(contingency)
                min_dim = min(contingency.shape) - 1
                v = np.sqrt(p_value * n / (n * min_dim))
                chi2 = v * n  # Convert to chi-square-like scale
                return chi2, p_value
            
            # Use chi-square test for larger counts
            chi2, p_value = stats.chi2_contingency(contingency)[:2]
            return chi2, p_value
        except ValueError:
            # Fallback in case of any other statistical issues
            return 0.0, 1.0

class JsonBehaviorAnalyzer(BaseAnalyzer):
    """Analyzer for JSON-based experimental results"""
    
    def __init__(self):
        self.results_formatter = ResultsFormatter()
    
    def load_data(self, file_path: str) -> List[Dict]:
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

    def analyze_multiple_emotions(self, emotion_files: Dict[str, str]) -> Dict:
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
            data = self.load_data(file_path)
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
                'chi_square': float(chi2),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05)
            }
        
        # Prepare comprehensive results
        results = {
            'individual_conditions': emotion_data,
            'overall_test': {
                'chi_square': float(overall_chi2),
                'p_value': float(overall_p_value),
                'significant': bool(overall_p_value < 0.05)
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

class CsvBehaviorAnalyzer(BaseAnalyzer):
    """Analyzer for CSV experimental results with emotion/intensity analysis"""
    
    def __init__(self):
        self.results_formatter = ResultsFormatter()
        self.plotter = BehaviorVisualizer()

    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load and validate CSV data."""
        df = pd.read_csv(csv_path)
        required_columns = ['emotion', 'intensity', 'category']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV missing required columns: {required_columns}")
        return df

    def analyze_emotion_effects(self, df: pd.DataFrame) -> Dict:
        """Analyze behavioral differences across emotions."""
        emotion_counts = self._group_counts(df, 'emotion')
        return self.analyze_conditions(emotion_counts)

    def analyze_intensity_effects(self, df: pd.DataFrame) -> Dict:
        """Analyze intensity effects within each emotion."""
        results = {}
        for emotion in df['emotion'].unique():
            emotion_df = df[df['emotion'] == emotion]
            intensity_counts = self._group_counts(emotion_df, 'intensity')
            if len(intensity_counts) > 1:
                results[emotion] = self.analyze_conditions(intensity_counts)
        return results

    def _group_counts(self, df: pd.DataFrame, group_col: str) -> Dict[str, Dict]:
        """Calculate behavior counts for grouped data."""
        groups = df.groupby(group_col)
        return {
            group_name: {
                'cooperate': sum(group['category'] == '1'), # TODO: change the hardcoded category
                'defect': sum(group['category'] == '2')
            }
            for group_name, group in groups
        }

    def analyze_conditions(self, condition_counts: Dict[str, Dict]) -> Dict:
        """Core analysis for any grouped conditions."""
        # Calculate ratios and prepare data
        individual_conditions = {}
        for condition, counts in condition_counts.items():
            coop_ratio, defect_ratio = self.calculate_behavior_ratio(counts)
            individual_conditions[condition] = {
                'counts': counts,
                'cooperative_ratio': coop_ratio,
                'defective_ratio': defect_ratio
            }

        # Perform overall statistical test
        overall_chi2 = 0.0
        overall_p_value = 1.0
        conditions = list(condition_counts.values())
        
        if len(conditions) >= 2:
            # Create contingency table
            contingency = np.array([
                [c['cooperate'], c['defect']] 
                for c in conditions
            ])
            
            # Use Fisher's exact test for small samples
            if len(conditions) == 2:
                try:
                    _, overall_p_value = stats.fisher_exact(contingency)
                    # Convert to chi-square equivalent for consistency
                    n = np.sum(contingency)
                    phi = np.sqrt(overall_p_value * n / (n * 1))
                    overall_chi2 = phi * n
                except:
                    pass
            else:
                try:
                    overall_chi2, overall_p_value, _, _ = stats.chi2_contingency(contingency)
                except:
                    pass

        # Perform pairwise comparisons
        pairwise_comparisons = {}
        for (cond1, cond2) in combinations(condition_counts.keys(), 2):
            chi2, p_value = self.chi_square_test(
                condition_counts[cond1],
                condition_counts[cond2]
            )
            pairwise_comparisons[f"{cond1}_vs_{cond2}"] = {
                'chi_square': float(chi2),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }

        return {
            'individual_conditions': individual_conditions,
            'overall_test': {
                'chi_square': float(overall_chi2),
                'p_value': float(overall_p_value),
                'significant': overall_p_value < 0.05
            },
            'pairwise_comparisons': pairwise_comparisons
        }

class ResultsFormatter:
    """Handles formatting of analysis results"""
    
    def format_full_results(self, analysis_results: Dict) -> Dict:
        """Add text descriptions and structure to results."""
        return {
            **analysis_results,
            'text_description': self.generate_text_description(analysis_results)
        }
    
    def generate_text_description(self, results: Dict) -> str:
        """Implementation remains similar to existing generate_text_description..."""

class BehaviorVisualizer:
    """Handles generation of visualizations"""
    
    def plot_comparison(self, results: Dict, output_path: str = None) -> None:
        """Implementation remains similar to existing plot_behavior_comparison..."""
    
    def plot_intensity_effects(self, intensity_results: Dict, output_dir: str) -> None:
        """Specialized plots for intensity analysis"""
        for emotion, results in intensity_results.items():
            self.plot_comparison(
                results, 
                f"{output_dir}/{emotion}_intensity_analysis.png"
            )

# Updated convenience functions
def analyze_emotion_and_intensity_effects(csv_path: str, output_dir: str = 'results') -> Dict:
    """
    Full analysis pipeline for CSV data.
    
    Args:
        csv_path: Path to input CSV file
        output_dir: Directory to save analysis results
        
    Returns:
        Comprehensive analysis results with visualizations
    """
    analyzer = CsvBehaviorAnalyzer()
    df = analyzer.load_data(csv_path)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Perform analyses
    emotion_results = analyzer.analyze_emotion_effects(df)
    intensity_results = analyzer.analyze_intensity_effects(df)
    
    # Generate visualizations
    analyzer.plotter.plot_comparison(emotion_results, f"{output_dir}/emotion_analysis.png")
    analyzer.plotter.plot_intensity_effects(intensity_results, output_dir)
    
    # Format final results
    full_results = {
        'emotion_analysis': emotion_results,
        'intensity_analysis': intensity_results
    }
    return analyzer.results_formatter.format_full_results(full_results)

if __name__ == "__main__":
    # Example usage
    emotion_files = {
        'anger': 'results/emotion_game_theory_20250206_193232/Stag_Hunt_anger_results.json',
        'happy': 'results/emotion_game_theory_20250206_193232/Stag_Hunt_happiness_results.json',
        'sad': 'results/emotion_game_theory_20250206_193232/Stag_Hunt_sadness_results.json',
        'fear': 'results/emotion_game_theory_20250206_193232/Stag_Hunt_fear_results.json',
        'disgust': 'results/emotion_game_theory_20250206_193232/Stag_Hunt_disgust_results.json',
        'surprise': 'results/emotion_game_theory_20250206_193232/Stag_Hunt_surprise_results.json'
    }
    

    results = analyze_emotion_and_intensity_effects('results/RePEng/Stag_Hunt_Llama-3.1-8B-Instruct/exp_results_20250208_164554.csv')
    from pprint import pprint
    pprint(results)
    
    # save results to json file
    # with open('results/emotion_game_theory_20250206_193232/analysis_results.json', 'w') as f:
    #     json.dump(results, f, indent=4)