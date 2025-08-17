"""
Main emotion memory experiment class.
Follows the pattern from emotion_game_experiment.py but adapted for memory benchmarks.
"""
import json
import torch
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread, current_thread
import time

from neuro_manipulation.model_layer_detector import ModelLayerDetector
from neuro_manipulation.model_utils import load_emotion_readers, setup_model_and_tokenizer
from neuro_manipulation.repe.pipelines import get_pipeline
from vllm import LLM

from .data_models import ExperimentConfig, ResultRecord, DEFAULT_GENERATION_CONFIG
from .benchmark_adapters import get_adapter


class EmotionMemoryExperiment:
    """
    Main experiment class for testing emotion effects on memory benchmarks.
    Closely follows EmotionGameExperiment pattern but adapted for memory tasks.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.generation_config = config.generation_config or DEFAULT_GENERATION_CONFIG
        
        # Setup logging (same pattern as emotion_game_experiment)
        self.logger = logging.getLogger(__name__)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/emotion_memory_experiment_{timestamp}.log"
        
        if not self.logger.handlers:
            Path("logs").mkdir(parents=True, exist_ok=True)
            self.logger.setLevel(logging.INFO)
            
            # File handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(file_handler)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(console_handler)
            self.logger.propagate = False
        
        self.logger.info(f"Initializing emotion memory experiment with model: {config.model_path}")
        self.logger.info(f"Log file created at: {log_file}")
        
        # Setup model and emotion readers (same pattern as emotion_game_experiment)
        # Import the proper config function
        try:
            from neuro_manipulation.configs.experiment_config import get_repe_eng_config
            repe_config = get_repe_eng_config(config.model_path)
            # Override with experiment-specific values
            repe_config.update({
                "emotions": config.emotions,
                "coeffs": config.intensities,
            })
        except ImportError:
            # Fallback config if the import fails
            repe_config = {
                "model_name_or_path": config.model_path,
                "emotions": config.emotions,
                "coeffs": config.intensities,
                "block_name": "transformer.h.{}.mlp",  # Default for Qwen
                "control_method": "linear_comb",
                "data_dir": "/home/jjl7137/representation-engineering/data/emotions",
                "rep_token": -1,
                "n_difference": 1,
                "direction_method": "pca",
                "rebuild": False
            }
        
        # First load from HF for emotion readers
        self.model, self.tokenizer, self.prompt_format = setup_model_and_tokenizer(
            repe_config, from_vllm=False
        )
        num_hidden_layers = ModelLayerDetector.num_layers(self.model)
        self.hidden_layers = list(range(-1, -num_hidden_layers - 1, -1))
        self.logger.info(f"Using hidden layers: {self.hidden_layers}")
        
        self.emotion_rep_readers = load_emotion_readers(
            repe_config, self.model, self.tokenizer, self.hidden_layers
        )
        del self.model  # Save memory
        
        # Load vLLM model for inference
        self.model, self.tokenizer, self.prompt_format = setup_model_and_tokenizer(
            repe_config, from_vllm=True
        )
        self.logger.info(f"Model loaded: {type(self.model)}")
        self.is_vllm = isinstance(self.model, LLM)
        
        # Setup RepE control pipeline
        self.rep_control_pipeline = get_pipeline(
            "rep-control-vllm" if self.is_vllm else "rep-control",
            model=self.model,
            tokenizer=self.tokenizer,
            layers=self.hidden_layers[len(self.hidden_layers) // 3 : 2 * len(self.hidden_layers) // 3],
            block_name=repe_config["block_name"],
            control_method=repe_config["control_method"],
        )
        
        # Load benchmark adapter
        self.benchmark_adapter = get_adapter(config.benchmark)
        self.logger.info(f"Loaded benchmark: {config.benchmark.name}")
        
        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config.output_dir) / f"emotion_memory_{config.benchmark.name}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Current state tracking
        self.cur_emotion = None
        self.cur_intensity = None
    
    def run_experiment(self) -> pd.DataFrame:
        """Run the complete emotion memory experiment"""
        self.logger.info("Starting emotion memory experiment")
        
        # Load benchmark data
        benchmark_data = self.benchmark_adapter.get_data()
        self.logger.info(f"Loaded {len(benchmark_data)} benchmark items")
        
        all_results = []
        
        # Test each emotion with each intensity
        for emotion in self.config.emotions:
            self.logger.info(f"Processing emotion: {emotion}")
            rep_reader = self.emotion_rep_readers[emotion]
            self.cur_emotion = emotion
            
            for intensity in self.config.intensities:
                self.logger.info(f"Processing intensity: {intensity}")
                self.cur_intensity = intensity
                
                results = self._process_emotion_condition(
                    benchmark_data, rep_reader, emotion, intensity
                )
                all_results.extend(results)
        
        # Add neutral baseline
        self.cur_emotion = "neutral"
        self.cur_intensity = 0.0
        self.logger.info("Processing neutral baseline")
        
        neutral_results = self._process_emotion_condition(
            benchmark_data, None, "neutral", 0.0
        )
        all_results.extend(neutral_results)
        
        return self._save_results(all_results)
    
    def _process_emotion_condition(self, benchmark_data, rep_reader, emotion: str, intensity: float) -> List[ResultRecord]:
        """Process one emotion/intensity condition"""
        self.logger.info(f"Processing {emotion} with intensity {intensity}")
        
        # Setup activations
        if emotion == "neutral":
            activations = {}
        else:
            device = torch.device("cpu")  # For vLLM
            activations = {
                layer: torch.tensor(
                    intensity * rep_reader.directions[layer] * rep_reader.direction_signs[layer]
                ).to(device).half()
                for layer in self.hidden_layers
            }
        
        # Create prompts
        prompts = [self.benchmark_adapter.create_prompt(item) for item in benchmark_data]
        
        # Generate responses
        start_time = time.time()
        try:
            outputs = self.rep_control_pipeline(
                prompts,
                activations=activations,
                batch_size=self.config.batch_size,
                temperature=self.generation_config["temperature"],
                max_new_tokens=self.generation_config["max_new_tokens"],
                do_sample=self.generation_config["do_sample"],
                top_p=self.generation_config["top_p"],
            )
        except Exception as e:
            self.logger.error(f"Error during generation: {e}")
            return []
        
        end_time = time.time()
        self.logger.info(f"Generation completed in {end_time - start_time:.2f}s")
        
        # Process results
        results = []
        for i, (item, output) in enumerate(zip(benchmark_data, outputs)):
            if self.is_vllm:
                response = output.outputs[0].text.replace(prompts[i], "").strip()
            else:
                response = output[0]["generated_text"].replace(prompts[i], "").strip()
            
            # Evaluate response
            try:
                score = self.benchmark_adapter.evaluate_response(
                    response, item.ground_truth, self.config.benchmark.task_type
                )
            except Exception as e:
                self.logger.warning(f"Evaluation error for item {item.id}: {e}")
                score = 0.0
            
            result = ResultRecord(
                emotion=emotion,
                intensity=intensity,
                item_id=item.id,
                task_name=self.config.benchmark.task_type,
                prompt=prompts[i],
                response=response,
                ground_truth=item.ground_truth,
                score=score,
                metadata={
                    "benchmark": self.config.benchmark.name,
                    "item_metadata": item.metadata
                }
            )
            results.append(result)
        
        self.logger.info(f"Processed {len(results)} items for {emotion} intensity {intensity}")
        return results
    
    def _save_results(self, results: List[ResultRecord]) -> pd.DataFrame:
        """Save experiment results and compute summary statistics"""
        self.logger.info("Saving experiment results")
        
        # Convert to DataFrame
        results_data = []
        for result in results:
            results_data.append({
                'emotion': result.emotion,
                'intensity': result.intensity,
                'item_id': result.item_id,
                'task_name': result.task_name,
                'response': result.response,
                'ground_truth': str(result.ground_truth),
                'score': result.score,
                'benchmark': result.metadata.get('benchmark', ''),
            })
        
        df = pd.DataFrame(results_data)
        
        # Save detailed results
        csv_filename = self.output_dir / "detailed_results.csv"
        df.to_csv(csv_filename, index=False)
        self.logger.info(f"Detailed results saved to {csv_filename}")
        
        # Save raw results with full metadata
        json_filename = self.output_dir / "raw_results.json"
        with open(json_filename, "w") as f:
            json.dump([result.__dict__ for result in results], f, indent=2, default=str)
        
        # Compute summary statistics
        summary = df.groupby(['emotion', 'intensity']).agg({
            'score': ['mean', 'std', 'count', 'min', 'max']
        }).round(4)
        
        summary_filename = self.output_dir / "summary_results.csv"
        summary.to_csv(summary_filename)
        self.logger.info(f"Summary results saved to {summary_filename}")
        
        # Print summary
        self.logger.info("\n=== EXPERIMENT RESULTS SUMMARY ===")
        self.logger.info(f"\n{summary}")
        
        return df
    
    def run_sanity_check(self, sample_limit: int = 5) -> pd.DataFrame:
        """Run a quick sanity check with limited samples"""
        self.logger.info(f"Running sanity check with {sample_limit} samples")
        
        # Temporarily modify config
        original_sample_limit = self.config.benchmark.sample_limit
        self.config.benchmark.sample_limit = sample_limit
        
        # Reset adapter to pick up new sample limit
        self.benchmark_adapter = get_adapter(self.config.benchmark)
        
        try:
            return self.run_experiment()
        finally:
            # Restore original config
            self.config.benchmark.sample_limit = original_sample_limit