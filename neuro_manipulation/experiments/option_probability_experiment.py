import json
import torch
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from scipy.stats import chi2_contingency, fisher_exact
import itertools

from torch.utils.data import DataLoader
from vllm import LLM, SamplingParams

from neuro_manipulation.datasets.context_manipulation_dataset import (
    ContextManipulationDataset, 
    collate_context_manipulation_scenarios
)
from neuro_manipulation.repe.sequence_prob_vllm_hook import CombinedVLLMHook
from neuro_manipulation.model_utils import setup_model_and_tokenizer
from neuro_manipulation.prompt_wrapper import GameReactPromptWrapper
from neuro_manipulation.repe.pipelines import get_pipeline


@dataclass
class ExperimentCondition:
    """Represents a single experimental condition in the 2x2 factorial design."""
    emotion: str  # e.g., 'angry', 'neutral'
    context: str  # e.g., 'with_description', 'without_description'
    emotion_intensity: float = 0.0  # For emotion intervention
    include_description: bool = True


class OptionProbabilityExperiment:
    """
    Experiment class for measuring choice probabilities across all options in a 2x2 factorial design.
    
    This experiment tests:
    - Emotion Factor: angry vs neutral
    - Context Factor: with_description vs without_description
    
    Uses SequenceProbVLLMHook to measure the probability of each behavioral choice option,
    enabling analysis of how emotion and context interact to influence decision probabilities.
    """
    
    def __init__(
        self,
        repe_eng_config: Dict[str, Any],
        exp_config: Dict[str, Any],
        game_config: Dict[str, Any],
        batch_size: int = 4,
        sample_num: Optional[int] = None,
    ):
        # Setup logging
        self.logger = logging.getLogger(__name__)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/option_probability_experiment_{timestamp}.log"
        
        if not self.logger.handlers:
            Path("logs").mkdir(parents=True, exist_ok=True)
            self.logger.setLevel(logging.INFO)
            # Add file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(file_handler)
            # Add console handler  
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(console_handler)
            self.logger.propagate = False
            
        self.logger.info(f"Initializing OptionProbabilityExperiment with model: {repe_eng_config['model_name_or_path']}")
        self.logger.info(f"Log file created at: {log_file}")
        
        self.repe_eng_config = repe_eng_config
        self.exp_config = exp_config
        self.game_config = game_config
        self.batch_size = batch_size
        self.sample_num = sample_num
        
        # Setup model and tokenizer
        self.model, self.tokenizer, self.prompt_format = setup_model_and_tokenizer(
            repe_eng_config, from_vllm=True
        )
        self.is_vllm = isinstance(self.model, LLM)
        
        if not self.is_vllm:
            raise ValueError("OptionProbabilityExperiment requires vLLM model for sequence probability measurement")
            
        # Initialize sequence probability hook
        self.sequence_prob_hook = SequenceProbVLLMHook(
            model=self.model,
            tokenizer=self.tokenizer,
            tensor_parallel_size=self.repe_eng_config.get('tensor_parallel_size', 1)
        )
        
        # Setup prompt wrapper
        self.prompt_wrapper = GameReactPromptWrapper(
            self.prompt_format, 
            response_format=self.game_config["decision_class"]
        )
        
        # Get emotion configurations
        self.emotions = self.exp_config["experiment"].get("emotions", ["neutral", "angry"])
        self.emotion_intensities = self.exp_config["experiment"].get(
            "emotion_intensities", [0.0, 1.0]
        )
        
        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = (
            f"{self.exp_config['experiment']['output']['base_dir']}/"
            f"option_probability_{self.exp_config['experiment']['name']}_"
            f"{self.game_config['game_name']}_{self.repe_eng_config['model_name_or_path'].split('/')[-1]}_{timestamp}"
        )
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.results = []
        
    def run_experiment(self) -> str:
        """
        Run the complete 2x2 factorial experiment.
        
        Returns:
            Path to the saved results file
        """
        self.logger.info("Starting 2x2 factorial option probability experiment")
        
        # Generate all experimental conditions
        conditions = self._generate_experimental_conditions()
        self.logger.info(f"Generated {len(conditions)} experimental conditions")
        
        # Run experiment for each condition
        for condition in conditions:
            self.logger.info(f"Running condition: Emotion={condition.emotion}, Context={condition.context}")
            condition_results = self._run_single_condition(condition)
            self.results.extend(condition_results)
            
        # Save results
        results_file = self._save_results()
        
        # Run statistical analysis
        self._run_statistical_analysis(results_file)
        
        self.logger.info(f"Experiment completed. Results saved to: {results_file}")
        return results_file
        
    def _generate_experimental_conditions(self) -> List[ExperimentCondition]:
        """Generate all conditions for the 2x2 factorial design."""
        conditions = []
        
        # Define context conditions
        context_conditions = [
            ("with_description", True),
            ("without_description", False)
        ]
        
        # Generate all combinations
        for emotion in self.emotion:
            emotion_intensity = 0.0 if emotion == "neutral" else self.emotion_intensities[1]
            
            for context_name, include_desc in context_conditions:
                condition = ExperimentCondition(
                    emotion=emotion,
                    context=context_name,
                    emotion_intensity=emotion_intensity,
                    include_description=include_desc
                )
                conditions.append(condition)
                
        return conditions
        
    def _run_single_condition(self, condition: ExperimentCondition) -> List[Dict[str, Any]]:
        """Run experiment for a single condition."""
        # Create dataset for this condition
        dataset = self._create_dataset_for_condition(condition)
        
        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_context_manipulation_scenarios
        )
        
        condition_results = []
        
        for batch_idx, batch in enumerate(data_loader):
            self.logger.debug(f"Processing batch {batch_idx} for condition {condition.emotion}-{condition.context}")
            
            # Measure probabilities for all options in this batch
            batch_results = self._measure_option_probabilities(batch, condition)
            condition_results.extend(batch_results)
            
        return condition_results
        
    def _create_dataset_for_condition(self, condition: ExperimentCondition) -> ContextManipulationDataset:
        """Create dataset with appropriate emotion and context settings for a condition."""
        # Setup emotion-specific prompt wrapper
        if condition.emotion == "neutral":
            user_message = "You are participating in this scenario."
        else:
            user_message = f"You are participating in this scenario. You are feeling {condition.emotion}."
            
        from functools import partial
        emotion_prompt_wrapper = partial(
            self.prompt_wrapper.__call__,
            user_messages=user_message
        )
        
        # Create dataset with context manipulation
        dataset = ContextManipulationDataset(
            game_config=self.game_config,
            prompt_wrapper=emotion_prompt_wrapper,
            include_description=condition.include_description,
            sample_num=self.sample_num
        )
        
        return dataset
        
    def _measure_option_probabilities(
        self, 
        batch: Dict[str, List], 
        condition: ExperimentCondition
    ) -> List[Dict[str, Any]]:
        """Measure probabilities for all options in a batch."""
        prompts = batch['prompt']
        options_list = batch['options']
        
        batch_results = []
        
        for i, (prompt, options) in enumerate(zip(prompts, options_list)):
            # Extract option texts for probability measurement
            option_texts = [opt.split('. ', 1)[1] for opt in options]  # Remove "Option 1. " prefix
            
            try:
                # Use SequenceProbVLLMHook to get probabilities
                prob_results = self.sequence_prob_hook.get_log_prob(
                    text_inputs=[prompt],
                    target_sequences=option_texts
                )
                
                if prob_results and len(prob_results) > 0:
                    log_probs = prob_results[0]  # First (and only) input
                    
                    # Filter out failed calculations and convert tensors to floats
                    valid_log_probs = {
                        seq: val.item() for seq, val in log_probs.items() if isinstance(val, torch.Tensor)
                    }

                    if not valid_log_probs:
                        self.logger.warning(f"Could not calculate any valid probabilities for prompt {i}")
                        continue

                    # Convert log probabilities to probabilities
                    probs = {seq: np.exp(log_prob) for seq, log_prob in valid_log_probs.items()}
                    
                    # Normalize probabilities to sum to 1
                    total_prob = sum(probs.values())
                    if total_prob > 0:
                        probs = {seq: prob / total_prob for seq, prob in probs.items()}
                    
                    # Store result
                    result = {
                        'condition_emotion': condition.emotion,
                        'condition_context': condition.context,
                        'emotion_intensity': condition.emotion_intensity,
                        'include_description': condition.include_description,
                        'prompt': prompt,
                        'scenario': batch['scenario'][i],
                        'behavior_choices': batch['behavior_choices'][i],
                        'option_probabilities': probs,
                        'log_probabilities': valid_log_probs,
                        'options': options,
                        'batch_idx': i
                    }
                    batch_results.append(result)
                    
                else:
                    self.logger.warning(f"No probability results for prompt {i} in condition {condition.emotion}-{condition.context}")
                    
            except Exception as e:
                self.logger.error(f"Error measuring probabilities for prompt {i}: {e}")
                continue
                
        return batch_results
        
    def _save_results(self) -> str:
        """Save experiment results to JSON and CSV files."""
        # Save raw results as JSON
        json_file = f"{self.output_dir}/option_probability_results.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        # Convert to structured DataFrame for analysis
        structured_data = []
        
        for result in self.results:
            base_row = {
                'condition_emotion': result['condition_emotion'],
                'condition_context': result['condition_context'],
                'emotion_intensity': result['emotion_intensity'],
                'include_description': result['include_description'],
                'scenario': result['scenario'],
                'behavior_choices': result['behavior_choices']
            }
            
            # Add probability for each option
            option_probs = result['option_probabilities']
            for i, (option_text, prob) in enumerate(option_probs.items()):
                row = base_row.copy()
                row.update({
                    'option_id': i + 1,
                    'option_text': option_text,
                    'probability': prob,
                    'log_probability': result['log_probabilities'].get(option_text, float('-inf'))
                })
                structured_data.append(row)
                
        # Save as CSV
        df = pd.DataFrame(structured_data)
        csv_file = f"{self.output_dir}/option_probability_results.csv"
        df.to_csv(csv_file, index=False)
        
        self.logger.info(f"Results saved to {json_file} and {csv_file}")
        return csv_file
        
    def _convert_keys_to_str(self, obj):
        """Recursively convert all keys in a dictionary to strings."""
        if isinstance(obj, dict):
            return {str(k): self._convert_keys_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_keys_to_str(i) for i in obj]
        else:
            return obj

    def _run_statistical_analysis(self, csv_file: str):
        """Run statistical analysis on the probability data."""
        self.logger.info("Running statistical analysis for emotion-context interaction effects")

        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                self.logger.warning(f"CSV file is empty, skipping analysis: {csv_file}")
                return
        except pd.errors.EmptyDataError:
            self.logger.warning(f"CSV file is empty, skipping analysis: {csv_file}")
            return
        
        # Analysis results storage
        analysis_results = {
            'summary_statistics': {},
            'interaction_effects': {},
            'pairwise_comparisons': {}
        }
        
        # Summary statistics by condition
        summary_stats = df.groupby(['condition_emotion', 'condition_context', 'option_id']).agg({
            'probability': ['mean', 'std', 'count']
        }).round(4)
        
        if summary_stats.empty:
            self.logger.warning("No data available for summary statistics. Skipping.")
        else:
            analysis_results['summary_statistics'] = summary_stats.reset_index().to_dict(orient='records')
        
        # Test for interaction effects using probability data
        for option_id in df['option_id'].unique():
            option_data = df[df['option_id'] == option_id]
            
            if option_data.empty:
                continue

            # Create contingency table for this option
            # We'll test if high probability choices (>median) interact with conditions
            median_prob = option_data['probability'].median()
            option_data['high_prob'] = option_data['probability'] > median_prob
            
            contingency = pd.crosstab(
                index=[option_data['condition_emotion'], option_data['condition_context']], 
                columns=option_data['high_prob']
            )
            
            if contingency.shape == (4, 2):  # 2x2x2 design
                try:
                    # Chi-square test for independence
                    chi2, p_value, dof, expected = chi2_contingency(contingency)
                    
                    analysis_results['interaction_effects'][f'option_{option_id}'] = {
                        'chi_square': float(chi2),
                        'p_value': float(p_value),
                        'degrees_of_freedom': int(dof),
                        # Convert contingency table to JSON-compatible format, ensuring all keys are strings
                        'contingency_table': [
                            {str(k): v for k, v in row.items()} 
                            for row in contingency.reset_index().to_dict('records')
                        ]
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Could not run chi-square test for option {option_id}: {e}")
        
        # Pairwise comparisons between conditions
        for option_id in df['option_id'].unique():
            option_data = df[df['option_id'] == option_id]

            if option_data.empty:
                continue
            
            pairwise_results = {}
            conditions = option_data.groupby(['condition_emotion', 'condition_context'])
            
            if len(conditions) < 2:
                continue

            for (cond1_name, cond1_data), (cond2_name, cond2_data) in itertools.combinations(conditions, 2):
                try:
                    # Mann-Whitney U test for probability distributions
                    from scipy.stats import mannwhitneyu
                    statistic, p_value = mannwhitneyu(
                        cond1_data['probability'], 
                        cond2_data['probability'], 
                        alternative='two-sided'
                    )
                    
                    # Convert tuple key to string for JSON compatibility
                    comparison_key = f"{'_'.join(cond1_name)}_vs_{'_'.join(cond2_name)}"
                    pairwise_results[comparison_key] = {
                        'mann_whitney_u': float(statistic),
                        'p_value': float(p_value),
                        'cond1_mean': float(cond1_data['probability'].mean()),
                        'cond2_mean': float(cond2_data['probability'].mean())
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Could not run pairwise comparison for option {option_id}: {e}")
                    
            analysis_results['pairwise_comparisons'][f'option_{option_id}'] = pairwise_results

        # Save analysis results
        analysis_file = Path(self.output_dir) / "statistical_analysis.json"
        
        # Convert all keys to strings recursively to ensure JSON compatibility
        safe_results = self._convert_keys_to_str(analysis_results)
        with open(analysis_file, 'w') as f:
            json.dump(safe_results, f, indent=2)
            
        self.logger.info(f"Statistical analysis saved to {analysis_file}")
        
        # Generate summary report
        self._generate_summary_report(analysis_results)

    def _generate_summary_report(self, analysis_results: Dict[str, Any]):
        """Generate a human-readable summary report."""
        report_file = f"{self.output_dir}/experiment_summary_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("OPTION PROBABILITY EXPERIMENT SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Experiment: {self.exp_config['experiment']['name']}\n")
            f.write(f"Game: {self.game_config['game_name']}\n")
            f.write(f"Model: {self.repe_eng_config['model_name_or_path']}\n")
            f.write(f"Total Results: {len(self.results)}\n\n")
            
            f.write("EXPERIMENTAL CONDITIONS:\n")
            f.write("- Emotions: " + ", ".join(self.emotions) + "\n")
            f.write("- Context: with_description, without_description\n")
            f.write("- Design: 2x2 factorial\n\n")
            
            f.write("KEY FINDINGS:\n")
            
            # Report significant interaction effects
            f.write("Interaction Effects (p < 0.05):\n")
            significant_effects = []
            for option, results in analysis_results['interaction_effects'].items():
                if results['p_value'] < 0.05:
                    significant_effects.append(f"  - {option}: χ² = {results['chi_square']:.3f}, p = {results['p_value']:.3f}")
                    
            if significant_effects:
                f.write("\n".join(significant_effects) + "\n\n")
            else:
                f.write("  No significant interaction effects found.\n\n")
                
            f.write("For detailed statistics, see statistical_analysis.json\n")
            
        self.logger.info(f"Summary report saved to {report_file}")


if __name__ == "__main__":
    # Example usage and testing
    from games.game_configs import get_game_config
    from constants import GameNames
    
    # Test configuration
    game_config = get_game_config(GameNames.PRISONERS_DILEMMA)
    
    repe_eng_config = {
        'model_name_or_path': 'meta-llama/Llama-3.1-8B-Instruct',
        'tensor_parallel_size': 1,
        'coeffs': [0.0, 1.0],
        'block_name': 'model.layers.{}.self_attn',
        'control_method': 'reading_vec'
    }
    
    exp_config = {
        'experiment': {
            'name': 'emotion_context_probability_test',
            'emotions': ['neutral', 'angry'],
            'emotion_intensities': [0.0, 1.0],
            'output': {
                'base_dir': 'experiments/option_probability'
            }
        }
    }
    
    # Run experiment
    experiment = OptionProbabilityExperiment(
        repe_eng_config=repe_eng_config,
        exp_config=exp_config,
        game_config=game_config,
        batch_size=2,
        sample_num=10
    )
    
    results_file = experiment.run_experiment()
    print(f"Experiment completed. Results: {results_file}") 