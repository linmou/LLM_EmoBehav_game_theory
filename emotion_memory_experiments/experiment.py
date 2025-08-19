"""
Main emotion memory experiment class.
Follows the pattern from emotion_game_experiment.py but adapted for memory benchmarks.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path
from queue import Queue
from threading import Thread, current_thread
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader
from vllm import LLM

from neuro_manipulation.configs.experiment_config import get_repe_eng_config
from neuro_manipulation.model_layer_detector import ModelLayerDetector
from neuro_manipulation.model_utils import (
    load_emotion_readers,
    setup_model_and_tokenizer,
)
from neuro_manipulation.repe.pipelines import get_pipeline

from .benchmark_adapters import get_adapter
from .data_models import DEFAULT_GENERATION_CONFIG, ExperimentConfig, ResultRecord
from .memory_prompt_wrapper import get_memory_prompt_wrapper


class EmotionMemoryExperiment:
    """
    Main experiment class for testing emotion effects on memory benchmarks.
    Closely follows EmotionGameExperiment pattern but adapted for memory tasks.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.generation_config = config.generation_config or DEFAULT_GENERATION_CONFIG

        # Pipeline settings (always enabled with DataLoader)
        self.max_evaluation_workers = config.max_evaluation_workers
        self.pipeline_queue_size = config.pipeline_queue_size

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
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            self.logger.addHandler(file_handler)

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            self.logger.addHandler(console_handler)
            self.logger.propagate = False

        self.logger.info(
            f"Initializing emotion memory experiment with model: {config.model_path}"
        )
        self.logger.info(f"Log file created at: {log_file}")

        # Extract enable_thinking from generation config (ISSUE 2 FIX)
        self.enable_thinking = self.generation_config.get("enable_thinking", False)

        # Setup model and emotion readers (same pattern as emotion_game_experiment)
        repe_config = get_repe_eng_config(
            config.model_path, yaml_config=config.repe_eng_config
        )

        # First load from HF for emotion readers
        self.model, self.tokenizer, self.prompt_format, processor = (
            setup_model_and_tokenizer(repe_config, from_vllm=False)
        )
        num_hidden_layers = ModelLayerDetector.num_layers(self.model)
        self.hidden_layers = list(range(-1, -num_hidden_layers - 1, -1))
        self.logger.info(f"Using hidden layers: {self.hidden_layers}")

        self.emotion_rep_readers = load_emotion_readers(
            repe_config,
            self.model,
            self.tokenizer,
            self.hidden_layers,
            processor,
            self.enable_thinking,
        )
        del self.model  # Save memory

        # Load vLLM model for inference
        self.model, self.tokenizer, self.prompt_format, _ = setup_model_and_tokenizer(
            repe_config, from_vllm=True
        )
        self.logger.info(f"Model loaded: {type(self.model)}")
        self.is_vllm = isinstance(self.model, LLM)

        # Setup RepE control pipeline
        self.rep_control_pipeline = get_pipeline(
            "rep-control-vllm" if self.is_vllm else "rep-control",
            model=self.model,
            tokenizer=self.tokenizer,
            layers=self.hidden_layers[
                len(self.hidden_layers) // 3 : 2 * len(self.hidden_layers) // 3
            ],
            block_name=repe_config["block_name"],
            control_method=repe_config["control_method"],
        )

        # Load benchmark adapter
        self.benchmark_adapter = get_adapter(config.benchmark)
        self.logger.info(f"Loaded benchmark: {config.benchmark.name}")

        # Validate adapter dataset (without prompt wrapper for info only)
        test_dataset = self.benchmark_adapter.get_dataset()
        dataset_size = len(test_dataset)  # type: ignore[arg-type]
        self.logger.info(f"Benchmark contains {dataset_size} items")

        # Create partial function for dataset integration
        memory_prompt_wrapper = get_memory_prompt_wrapper(
            config.benchmark.task_type, self.prompt_format
        )

        self.memory_prompt_wrapper_partial = partial(
            memory_prompt_wrapper.__call__,
            user_messages="Please provide your answer.",
            enable_thinking=self.enable_thinking,
        )

        self.logger.info(
            f"Created memory prompt wrapper for task: {config.benchmark.task_type}"
        )

        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = (
            Path(config.output_dir)
            / f"emotion_memory_{config.benchmark.name}_{timestamp}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Current state tracking
        self.cur_emotion: Optional[str] = None
        self.cur_intensity: Optional[float] = None

        # Sample limit for testing
        self.sample_num: Optional[int] = config.benchmark.sample_limit

        # Set sample limit on adapter if specified
        if self.sample_num is not None:
            # Note: This would require adapter modification to support sample_num
            self.logger.info(f"Sample limit set to {self.sample_num}")
            # TODO: Implement sample_num support in adapters if needed

        # Batch size for DataLoader
        self.batch_size = config.batch_size

    def build_dataloader(self) -> DataLoader:
        """Build DataLoader using adapter (PROPER ARCHITECTURE)"""

        self.logger.info(
            f"Creating benchmark dataset via adapter with sample_num={self.sample_num if self.sample_num is not None else 'all'}"
        )

        data_loader = self.benchmark_adapter.get_dataloader(
            batch_size=self.batch_size,
            shuffle=False,
            prompt_wrapper=self.memory_prompt_wrapper_partial,  # ✅ Adapter handles prompt wrapper integration
            collate_fn=self._collate_memory_benchmarks,  # Custom collate for the adapter dataset format
        )

        return data_loader

    def _collate_memory_benchmarks(self, batch):
        """Custom collate function for adapter datasets"""
        return {
            "prompt": [item["prompt"] for item in batch],
            "items": [item["item"] for item in batch],
            "contexts": [item["context"] for item in batch],
            "questions": [item["question"] for item in batch],
            "ground_truths": [item["ground_truth"] for item in batch],
            "metadata": [item.get("metadata", {}) for item in batch],
        }

    def run_experiment(self) -> pd.DataFrame:
        """Run the complete emotion memory experiment"""
        self.logger.info("Starting emotion memory experiment")
        all_results = []

        # Test each emotion with each intensity
        for emotion in self.config.emotions:
            self.logger.info(f"Processing emotion: {emotion}")
            rep_reader = self.emotion_rep_readers[emotion]
            self.cur_emotion = emotion

            # Build DataLoader for this emotion (fresh dataset each time)
            data_loader = self.build_dataloader()

            for intensity in self.config.intensities:
                self.logger.info(f"Processing intensity: {intensity}")
                self.cur_intensity = intensity

                results = self._infer_with_activation(rep_reader, data_loader)
                all_results.extend(results)

        # Add neutral baseline
        self.cur_emotion = "neutral"
        self.cur_intensity = 0.0
        self.logger.info("Processing neutral baseline")

        # Use the same rep_reader for neutral (with 0 intensity)
        data_loader = self.build_dataloader()
        neutral_results = self._infer_with_activation(rep_reader, data_loader)
        all_results.extend(neutral_results)

        return self._save_results(all_results)

    def _infer_with_activation(self, rep_reader, data_loader) -> List[ResultRecord]:
        """Process with activations using DataLoader"""
        self.logger.info(
            f"Setting up activations for emotion {self.cur_emotion} with intensity {self.cur_intensity}"
        )

        # Setup activations
        if self.cur_emotion == "neutral" or self.cur_intensity == 0.0:
            activations = {}
        else:
            # For vLLM models, use cpu device
            device = torch.device("cpu") if self.is_vllm else self.model.device
            activations = {
                layer: torch.tensor(
                    self.cur_intensity
                    * rep_reader.directions[layer]
                    * rep_reader.direction_signs[layer]
                )
                .to(device)
                .half()
                for layer in self.hidden_layers
            }

        # Process batches using DataLoader (matches EmotionGameExperiment._forward_dataloader pattern)
        return self._forward_dataloader(data_loader, activations)

    def _forward_dataloader(self, data_loader, activations: Dict) -> List[ResultRecord]:
        """Forward pass using DataLoader"""
        batch_results = []
        pipeline_queue: "Queue[Any]" = Queue(maxsize=2)  # Control memory usage
        processed_futures = []  # Keep track of futures

        def pipeline_worker():
            for i, batch in enumerate(data_loader):
                # Process batch prompts
                start_time = time.time()
                control_outputs = self.rep_control_pipeline(
                    batch["prompt"],  # Use formatted prompts from dataset
                    activations=activations,
                    batch_size=self.batch_size,
                    temperature=self.generation_config["temperature"],
                    max_new_tokens=self.generation_config["max_new_tokens"],
                    do_sample=self.generation_config["do_sample"],
                    top_p=self.generation_config["top_p"],
                )
                end_time = time.time()
                pipeline_queue.put((i, batch, control_outputs))

            pipeline_queue.put(None)  # Sentinel value

        # Start pipeline worker thread
        worker = Thread(target=pipeline_worker, name="PipelineWorker")
        worker.start()

        # Process results while next batch is being generated
        with ThreadPoolExecutor(
            max_workers=self.batch_size // 2, thread_name_prefix="PostProc"
        ) as post_proc_executor:
            active_post_proc_tasks = 0
            while True:
                item = pipeline_queue.get()

                if item is None:
                    break  # Worker finished

                batch_idx, batch, control_outputs = item

                # Submit post-processing to executor
                active_post_proc_tasks += 1
                future = post_proc_executor.submit(
                    self._post_process_memory_batch, batch, control_outputs, batch_idx
                )
                processed_futures.append((batch_idx, future))

            # Wait for all submitted tasks to complete and collect results
            results_dict = {}
            for batch_idx, future in processed_futures:
                try:
                    result = future.result()  # Blocks here
                    active_post_proc_tasks -= 1
                    results_dict[batch_idx] = result
                except Exception as e:
                    self.logger.error(
                        f"Post-processing failed for batch {batch_idx}: {e}"
                    )
                    results_dict[batch_idx] = []  # Store empty list on error

            # Combine results in order
            for i in sorted(results_dict.keys()):
                batch_results.extend(results_dict[i])

        worker.join()
        return batch_results

    def _post_process_memory_batch(
        self, batch: Dict[str, Any], control_outputs: List, batch_idx: int
    ) -> List[ResultRecord]:
        """Post-process memory batch using adapter dataset format (PROPER ARCHITECTURE)"""
        start_time = time.time()
        log_prefix = f"{time.time():.2f} [{current_thread().name}]"

        results = []
        batch_prompts = batch["prompt"]
        batch_items = batch["items"]  # BenchmarkItem objects from adapter
        batch_ground_truths = batch["ground_truths"]

        assert (
            len(batch_prompts) == len(batch_items) == len(control_outputs)
        ), f"Batch size mismatch: prompts={len(batch_prompts)}, items={len(batch_items)}, outputs={len(control_outputs)}"

        for i, (prompt, item, ground_truth, output) in enumerate(
            zip(batch_prompts, batch_items, batch_ground_truths, control_outputs)
        ):
            # Extract response text
            if output is None:
                self.logger.warning(
                    f"No output for item {item.id} due to processing error"
                )
                response = ""
                score = 0.0
            else:
                if self.is_vllm:
                    response = output.outputs[0].text.replace(prompt, "").strip()
                else:
                    response = output[0]["generated_text"].replace(prompt, "").strip()

                try:
                    score = self.benchmark_adapter.evaluate_response(
                        response, ground_truth, self.config.benchmark.task_type
                    )
                except Exception as e:
                    self.logger.warning(f"Evaluation error for item {item.id}: {e}")
                    score = 0.0

            # Create result record (ensure types are not None)
            current_emotion = self.cur_emotion or "unknown"
            current_intensity = self.cur_intensity or 0.0

            result = ResultRecord(
                emotion=current_emotion,
                intensity=current_intensity,
                item_id=item.id,
                task_name=self.config.benchmark.task_type,
                prompt=prompt,
                response=response,
                ground_truth=ground_truth,
                score=score,
                metadata={
                    "benchmark": self.config.benchmark.name,
                    "item_metadata": item.metadata or {},
                },
            )
            results.append(result)

        end_time = time.time()
        self.logger.info(
            f"{log_prefix} PostProc: Finished for batch {batch_idx} ({end_time - start_time:.2f}s). Returning {len(results)} results."
        )
        return results

    def _save_results(self, results: List[ResultRecord]) -> pd.DataFrame:
        """Save experiment results and compute summary statistics"""
        self.logger.info("Saving experiment results")

        # Convert to DataFrame
        results_data = []
        for result in results:
            results_data.append(
                {
                    "emotion": result.emotion,
                    "intensity": result.intensity,
                    "item_id": result.item_id,
                    "task_name": result.task_name,
                    "response": result.response,
                    "ground_truth": str(result.ground_truth),
                    "score": result.score,
                    "benchmark": (result.metadata or {}).get("benchmark", ""),
                }
            )

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
        summary = (
            df.groupby(["emotion", "intensity"])
            .agg({"score": ["mean", "std", "count", "min", "max"]})
            .round(4)
        )

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
