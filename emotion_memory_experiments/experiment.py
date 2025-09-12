"""
Main emotion experiment class.
Follows the pattern from emotion_game_experiment.py but adapted for otherbenchmarks.
"""

import json
import re
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path
from queue import Queue
from threading import Thread, current_thread
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pandas as pd
try:
    import torch  # type: ignore
    from torch.utils.data import DataLoader  # type: ignore
except Exception:
    torch = None  # type: ignore
    class DataLoader:  # type: ignore
        pass
try:
    # Optional for dry-run; real import only needed for execution
    from vllm import LLM  # type: ignore
except Exception:
    class LLM:  # Fallback dummy type to allow isinstance checks
        pass

from neuro_manipulation.configs.experiment_config import get_repe_eng_config
from neuro_manipulation.model_layer_detector import ModelLayerDetector
from neuro_manipulation.model_utils import (
    load_emotion_readers,
    setup_model_and_tokenizer,
)
from neuro_manipulation.repe.pipelines import get_pipeline

from .data_models import DEFAULT_GENERATION_CONFIG, ExperimentConfig, ResultRecord

# NEW: Import the registry-based component assembly
from .benchmark_component_registry import create_benchmark_components
from .truncation_utils import calculate_max_context_length


class EmotionExperiment:
    """
    Main experiment class for testing emotion effects on benchmarks.
    Closely follows EmotionGameExperiment pattern but adapted for tasks.
    """

    def __init__(self, config: ExperimentConfig, dry_run: bool = False, repeat_runs: int = 1, repeat_seed_base: int | None = None):
        # GPU-independent components first
        self._setup_basic_components(config)
        # Repeat configuration (number of independent runs per condition)
        self.repeat_runs = int(repeat_runs) if repeat_runs and repeat_runs > 0 else 1
        self.cur_repeat: int = 0
        self.repeat_seed_base = repeat_seed_base

        if dry_run:
            # Build real datasets for each emotion
            self.emotion_datasets = self._build_emotion_datasets()
            # Set GPU-dependent components to None
            self._set_gpu_components_to_none()
        else:
            # Set emotion_datasets to None for regular mode
            self.emotion_datasets = None
            # Setup GPU-dependent components
            self._setup_gpu_components(config)

    def _setup_basic_components(self, config: ExperimentConfig):
        """Setup GPU-independent components: logging, tokenizer, prompt_format, truncation"""
        self.config = config
        self.generation_config = config.generation_config or DEFAULT_GENERATION_CONFIG
        self.loading_config = config.loading_config  # May be None for defaults

        # Pipeline settings (always enabled with DataLoader)
        self.max_evaluation_workers = config.max_evaluation_workers
        self.pipeline_queue_size = config.pipeline_queue_size

        # Setup logging (same pattern as emotion_game_experiment)
        self.logger = logging.getLogger(__name__)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/emotion_experiment_{timestamp}.log"

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
            f"Initializing emotion experiment with model: {config.model_path}"
        )
        self.logger.info(f"Log file created at: {log_file}")

        self.enable_thinking = bool(
            self.generation_config.get("enable_thinking", False)
        )

        # Simple dataset validation using registry-based approach
        _, _, test_dataset = create_benchmark_components(
            benchmark_name=config.benchmark.name,
            task_type=config.benchmark.task_type,
            config=config.benchmark,
            prompt_format=None,  # Will be set later when we have proper tokenizer
        )
        dataset_size = len(test_dataset)
        self.logger.info(f"Benchmark contains {dataset_size} items")

        # Load tokenizer using proper utility function (CPU-based, no GPU needed)
        from neuro_manipulation.utils import load_tokenizer_only
        self.tokenizer, _ = load_tokenizer_only(
            model_name_or_path=config.model_path,
            expand_vocab=False,
            auto_load_multimodal=True,
        )

        # Create prompt format (only needs tokenizer)
        from neuro_manipulation.prompt_formats import PromptFormat

        self.prompt_format = PromptFormat(self.tokenizer)

        # Setup truncation parameters
        self.max_context_length = None
        self.truncation_strategy = "right"
        if config.benchmark.enable_auto_truncation:
            if self.loading_config is None:
                raise ValueError(
                    "Truncation enabled but no loading_config provided for max_model_len"
                )
            self.max_context_length = calculate_max_context_length(
                self.loading_config.max_model_len,
                config.benchmark.preserve_ratio,
                prompt_overhead=200,  # Reserve for prompt template
            )
            self.truncation_strategy = config.benchmark.truncation_strategy
            self.logger.info(
                f"Context truncation enabled: max_length={self.max_context_length}, "
                f"strategy='{self.truncation_strategy}'"
            )

        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = (
            Path(config.output_dir)
            / f"{config.model_path.split('/')[-1]}_{config.benchmark.name}_{config.benchmark.task_type}_{timestamp}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Current state tracking
        self.cur_emotion: Optional[str] = None
        self.cur_intensity: Optional[float] = None

        # Sample limit for testing
        self.sample_num: Optional[int] = config.benchmark.sample_limit

        # Batch size for DataLoader
        self.batch_size = config.batch_size

    def _create_dataset_for_emotion(self, emotion: str):
        """Create dataset for a specific emotion using registry-based component assembly"""
        # Use registry to get all three components in one call
        benchmark_prompt_wrapper_partial, answer_wrapper_partial, dataset = create_benchmark_components(
            benchmark_name=self.config.benchmark.name,
            task_type=self.config.benchmark.task_type,
            config=self.config.benchmark,
            prompt_format=self.prompt_format,
            emotion=emotion,
            enable_thinking=self.enable_thinking,
            augmentation_config=self.config.benchmark.augmentation_config,
            user_messages="Please provide your answer.",
            # Dataset parameters
            max_context_length=self.max_context_length,
            tokenizer=self.tokenizer,
            truncation_strategy=self.truncation_strategy,
        )

        return dataset, benchmark_prompt_wrapper_partial

    def _build_emotion_datasets(self):
        """Build real datasets for each emotion - this is the actual validation test"""
        emotion_datasets = {}

        for emotion in self.config.emotions:
            dataset, _ = self._create_dataset_for_emotion(emotion)
            emotion_datasets[emotion] = dataset
            self.logger.info(f"✓ {emotion}: {len(dataset)} items")

        return emotion_datasets

    def _set_gpu_components_to_none(self):
        """Set GPU-dependent attributes to None for dry-run"""
        self.model = None
        self.emotion_rep_readers = None
        self.rep_control_pipeline = None
        self.is_vllm = False
        self.hidden_layers = []
        self.repe_config = None

    def _setup_gpu_components(self, config: ExperimentConfig):
        """Setup GPU-dependent components: models, emotion readers, pipeline"""
        # Setup model and emotion readers (same pattern as emotion_game_experiment)
        self.repe_config = get_repe_eng_config(
            config.model_path, yaml_config=config.repe_eng_config
        )

        # Ensure loading_config has the model path if it exists
        if self.loading_config and not self.loading_config.model_path:
            self.loading_config.model_path = config.model_path

        # First load from HF for emotion readers
        self.model, tokenizer_temp, prompt_format_temp, processor = (
            setup_model_and_tokenizer(self.loading_config, from_vllm=False)
        )

        # Assert tokenizers are functionally equivalent (not necessarily identical objects)
        self._assert_tokenizers_equivalent(
            self.tokenizer, tokenizer_temp, "basic", "gpu"
        )

        num_hidden_layers = ModelLayerDetector.num_layers(self.model)
        self.hidden_layers = list(range(-1, -num_hidden_layers - 1, -1))
        self.logger.info(f"Using hidden layers: {self.hidden_layers}")

        self.emotion_rep_readers = load_emotion_readers(
            self.repe_config,
            self.model,
            tokenizer_temp,
            self.hidden_layers,
            processor,
            self.enable_thinking,
        )
        del self.model  # Save memory

        # Load vLLM model for inference with loading config
        self.model, tokenizer_temp, prompt_format_temp, _ = setup_model_and_tokenizer(
            self.loading_config, from_vllm=True
        )

        # Assert vLLM tokenizer is functionally equivalent to basic tokenizer
        self._assert_tokenizers_equivalent(
            self.tokenizer, tokenizer_temp, "basic", "vllm"
        )

        self.logger.info(f"Model loaded: {type(self.model)}")
        self.is_vllm = isinstance(self.model, LLM)

        # Setup RepE control pipeline - using basic tokenizer for consistency
        self.rep_control_pipeline = get_pipeline(
            "rep-control-vllm" if self.is_vllm else "rep-control",
            model=self.model,
            tokenizer=self.tokenizer,  # Use basic tokenizer instead of tokenizer_temp
            layers=self.hidden_layers[
                len(self.hidden_layers) // 3 : 2 * len(self.hidden_layers) // 3
            ],
            block_name=self.repe_config["block_name"],
            control_method=self.repe_config["control_method"],
        )

    def _assert_tokenizers_equivalent(
        self, tokenizer1, tokenizer2, name1: str, name2: str
    ):
        """
        Assert that two tokenizers are functionally equivalent by comparing key properties.

        Args:
            tokenizer1: First tokenizer to compare
            tokenizer2: Second tokenizer to compare
            name1: Descriptive name for first tokenizer (for error messages)
            name2: Descriptive name for second tokenizer (for error messages)

        Raises:
            AssertionError: If tokenizers are not functionally equivalent
        """
        try:
            # Check basic type compatibility
            assert type(tokenizer1) == type(
                tokenizer2
            ), f"Tokenizer types differ: {name1}={type(tokenizer1)} vs {name2}={type(tokenizer2)}"

            # Check vocabulary size (most important property)
            assert (
                tokenizer1.vocab_size == tokenizer2.vocab_size
            ), f"Vocab sizes differ: {name1}={tokenizer1.vocab_size} vs {name2}={tokenizer2.vocab_size}"

            # Check padding token configuration
            assert (
                tokenizer1.pad_token_id == tokenizer2.pad_token_id
            ), f"Pad token IDs differ: {name1}={tokenizer1.pad_token_id} vs {name2}={tokenizer2.pad_token_id}"

            # Check special tokens that affect model behavior
            assert (
                tokenizer1.eos_token_id == tokenizer2.eos_token_id
            ), f"EOS token IDs differ: {name1}={tokenizer1.eos_token_id} vs {name2}={tokenizer2.eos_token_id}"

            assert (
                tokenizer1.bos_token_id == tokenizer2.bos_token_id
            ), f"BOS token IDs differ: {name1}={tokenizer1.bos_token_id} vs {name2}={tokenizer2.bos_token_id}"

        except AttributeError as e:
            # Handle cases where tokenizers might not have certain attributes
            self.logger.warning(f"Tokenizer attribute missing during comparison: {e}")
            # Fall back to basic type check
            assert type(tokenizer1) == type(
                tokenizer2
            ), f"Tokenizer types differ: {name1}={type(tokenizer1)} vs {name2}={type(tokenizer2)}"

        # Log successful validation with key properties
        self.logger.info(
            f"✓ Tokenizer consistency validated: {name1} ≡ {name2} (vocab_size={tokenizer1.vocab_size})"
        )

    def build_dataloader(self, emotion: str) -> DataLoader:
        """
        Build DataLoader with proper dataset integration.

        Creates dataset using factory pattern and configures DataLoader
        to use the dataset's specialized collate function.
        """
        self.logger.info(
            f"Creating benchmark dataset with sample_num={self.sample_num if self.sample_num is not None else 'all'}"
        )

        # Use common dataset creation logic
        self.dataset, self.benchmark_prompt_wrapper_partial = (
            self._create_dataset_for_emotion(emotion)
        )

        # Use dataset's specialized collate function - each dataset type
        # has specific collation requirements for different benchmarks
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.dataset.collate_fn,
        )

        return dataloader

    def run_experiment(self) -> pd.DataFrame:
        """Run the complete emotion experiment"""
        self.logger.info("Starting emotion experiment")
        all_results = []

        # Test each emotion with each intensity
        for emotion in self.config.emotions:
            self.logger.info(f"Processing emotion: {emotion}")
            rep_reader = self.emotion_rep_readers[emotion]
            self.cur_emotion = emotion

            # Build DataLoader for this emotion (fresh dataset each time)
            data_loader = self.build_dataloader(self.cur_emotion)

            for intensity in self.config.intensities:
                self.logger.info(f"Processing intensity: {intensity}")
                self.cur_intensity = intensity
                # Repeat independent runs for this condition
                for r in range(self.repeat_runs):
                    self.cur_repeat = r
                    results = self._infer_with_activation(rep_reader, data_loader)
                    # Attach repeat_id to metadata for downstream aggregation
                    for rec in results:
                        if rec.metadata is None:
                            rec.metadata = {"repeat_id": r}
                        else:
                            rec.metadata["repeat_id"] = r
                    all_results.extend(results)

        # Add neutral baseline
        self.cur_emotion = "neutral"
        self.cur_intensity = 0.0  # set to 0.0 to avoid using activations
        self.logger.info("Processing neutral baseline")

        # Use the same rep_reader for neutral (with 0 intensity)
        data_loader = self.build_dataloader(self.cur_emotion)
        for r in range(self.repeat_runs):
            self.cur_repeat = r
            neutral_results = self._infer_with_activation(rep_reader, data_loader)
            for rec in neutral_results:
                if rec.metadata is None:
                    rec.metadata = {"repeat_id": r}
                else:
                    rec.metadata["repeat_id"] = r
            all_results.extend(neutral_results)

        return self._save_results(all_results)

    def _infer_with_activation(self, rep_reader, data_loader) -> List[ResultRecord]:
        """Process with activations using DataLoader"""
        self.logger.info(
            f"Setting up activations for emotion {self.cur_emotion} with intensity {self.cur_intensity}"
        )

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
            try:
                for i, batch in enumerate(data_loader):
                    try:
                        # Process batch prompts
                        start_time = time.time()
                        # Pass all generation parameters from config
                        generation_params = {
                            "temperature": self.generation_config.get(
                                "temperature", 0.1
                            ),
                            "max_new_tokens": self.generation_config.get(
                                "max_new_tokens", 100
                            ),
                            "do_sample": self.generation_config.get("do_sample", False),
                            "top_p": self.generation_config.get("top_p", 0.9),
                            "repetition_penalty": self.generation_config.get(
                                "repetition_penalty", 1.0
                            ),
                        }

                        # Add optional parameters if they exist and are not default
                        if self.generation_config.get("top_k", -1) != -1:
                            generation_params["top_k"] = self.generation_config["top_k"]
                        if self.generation_config.get("min_p", 0.0) != 0.0:
                            generation_params["min_p"] = self.generation_config["min_p"]
                        if self.generation_config.get("presence_penalty", 0.0) != 0.0:
                            generation_params["presence_penalty"] = (
                                self.generation_config["presence_penalty"]
                            )
                        if self.generation_config.get("frequency_penalty", 0.0) != 0.0:
                            generation_params["frequency_penalty"] = (
                                self.generation_config["frequency_penalty"]
                            )

                        # Add per-run RNG seed if requested
                        if getattr(self, "repeat_seed_base", None) is not None:
                            generation_params["random_seed"] = int(self.repeat_seed_base) + int(getattr(self, "cur_repeat", 0))

                        # Validate batch structure before accessing
                        if "prompts" not in batch:
                            raise ValueError(
                                f"Batch missing required 'prompts' key. Available keys: {list(batch.keys())}"
                            )

                        control_outputs = self.rep_control_pipeline(
                            batch["prompts"],  # Use formatted prompts from dataset
                            activations=activations,
                            batch_size=self.batch_size,
                            **generation_params,
                        )
                        end_time = time.time()
                        pipeline_queue.put((i, batch, control_outputs))

                    except Exception as batch_error:
                        # Handle errors for individual batch processing
                        # This catches AssertionError from benchmark_prompt_wrapper augmentation
                        # and other batch-level errors, ensuring the pipeline continues
                        import traceback

                        error_trace = traceback.format_exc()
                        self.logger.error(
                            f"🚨 BATCH ERROR in pipeline worker for batch {i}: {str(batch_error)}\n{error_trace}"
                        )

                        # Create an error batch result to maintain sequence integrity
                        error_batch = {
                            "prompts": [
                                f"ERROR: Batch {i} failed - {str(batch_error)}"
                            ],
                            "items": [MagicMock(id=f"error_{i}")],
                            "ground_truths": ["ERROR"],
                        }
                        error_outputs = [
                            {"generated_text": f"ERROR: {str(batch_error)}"}
                        ]

                        pipeline_queue.put((i, error_batch, error_outputs))

                        # Continue with next batch instead of crashing the worker thread
                        self.logger.info(
                            f"🔄 Continuing pipeline worker with next batch after error in batch {i}"
                        )

            except Exception as worker_error:
                # Handle catastrophic worker thread errors
                import traceback

                error_trace = traceback.format_exc()
                self.logger.error(
                    f"🚨 WORKER THREAD ERROR in pipeline_worker: {str(worker_error)}\n{error_trace}"
                )

                # Put error marker in queue to prevent main thread from waiting forever
                pipeline_queue.put(("WORKER_ERROR", str(worker_error), error_trace))

            finally:
                # Always put sentinel value to signal completion, even after errors
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

                # Handle worker thread error case
                if (
                    isinstance(item, tuple)
                    and len(item) == 3
                    and item[0] == "WORKER_ERROR"
                ):
                    error_type, error_msg, error_trace = item
                    self.logger.error(
                        f"🚨 PIPELINE WORKER FAILED: {error_msg}\n{error_trace}"
                    )
                    self.logger.info(
                        "🔄 Main thread continuing despite worker thread failure"
                    )
                    break  # Exit main processing loop but don't crash experiment

                batch_idx, batch, control_outputs = item

                # Submit post-processing to executor
                active_post_proc_tasks += 1
                future = post_proc_executor.submit(
                    self._post_process_batch, batch, control_outputs, batch_idx
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

    def _post_process_batch(
        self, batch: Dict[str, Any], control_outputs: List, batch_idx: int
    ) -> List[ResultRecord]:
        """Post-process batch using adapter dataset format (PROPER ARCHITECTURE)"""
        start_time = time.time()
        log_prefix = f"{time.time():.2f} [{current_thread().name}]"

        results = []

        # Validate batch structure with helpful error messages
        required_keys = ["prompts", "items", "ground_truths"]
        missing_keys = [key for key in required_keys if key not in batch]
        if missing_keys:
            raise ValueError(
                f"Batch missing required keys: {missing_keys}. "
                f"Available keys: {list(batch.keys())}. "
                f"Expected structure from collate_fn: {required_keys}"
            )

        batch_prompts = batch["prompts"]
        batch_items = batch["items"]  # BenchmarkItem objects from adapter
        batch_ground_truths = batch["ground_truths"]

        assert (
            len(batch_prompts) == len(batch_items) == len(control_outputs)
        ), f"Batch size mismatch: prompts={len(batch_prompts)}, items={len(batch_items)}, outputs={len(control_outputs)}"

        # Extract all responses first
        responses = []
        for i, (prompt, item, ground_truth, output) in enumerate(
            zip(batch_prompts, batch_items, batch_ground_truths, control_outputs)
        ):
            if output is None:
                self.logger.warning(
                    f"No output for item {item.id} due to processing error"
                )
                responses.append("")
            else:
                if self.is_vllm:
                    response = output.outputs[0].text.replace(prompt, "").strip()
                else:
                    response = output[0]["generated_text"].replace(prompt, "").strip()
                responses.append(response)

        # Normalize responses: strip leading empty <think> blocks if present
        empty_think_prefix = re.compile(r"^\s*<think>\s*</think>\s*", re.IGNORECASE)
        cleaned_responses = [empty_think_prefix.sub("", r or "") for r in responses]

        # Batch evaluation using LLM
        try:
            task_names = [self.config.benchmark.task_type] * len(responses)
            scores = self.dataset.evaluate_batch(
                cleaned_responses, batch_ground_truths, task_names, batch_prompts
            )
        except Exception as e:
            self.logger.error(f"Batch evaluation failed: {e}")
            scores = [0.0] * len(responses)

        # Create result records with batch-computed scores
        for i, (response, score, prompt, item, ground_truth) in enumerate(
            zip(cleaned_responses, scores, batch_prompts, batch_items, batch_ground_truths)
        ):
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
                repeat_id=getattr(self, "cur_repeat", 0),
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
        """Save experiment results and compute summary statistics

        Adds support for repeat-run aggregation when `repeat_id` is present
        in ResultRecord.metadata. Produces:
          - detailed_results.csv (with repeat_id column when available)
          - summary_results.csv (legacy: per emotion,intensity over all samples)
          - summary_by_repeat.csv (per emotion,intensity,repeat_id)
          - summary_overall.csv (across-repeat aggregation with between/pooled var)
        """
        self.logger.info("Saving experiment results")

        # Save full experiment configuration
        self._save_experiment_config()

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
                    "repeat_id": getattr(result, "repeat_id", None),
                }
            )

        df = pd.DataFrame(results_data)

        # Fallback: legacy records might have repeat_id only in metadata
        if df["repeat_id"].isna().any():
            try:
                # If metadata column exists in df (unlikely here since we flattened), skip.
                pass
            except Exception:
                pass

        # Save detailed results
        csv_filename = self.output_dir / "detailed_results.csv"
        df.to_csv(csv_filename, index=False)
        self.logger.info(f"Detailed results saved to {csv_filename}")

        # Save raw results with full metadata
        json_filename = self.output_dir / "raw_results.json"
        with open(json_filename, "w") as f:
            json.dump([result.__dict__ for result in results], f, indent=2, default=str)

        # Compute legacy summary statistics (ignores repeat_id)
        summary = (
            df.groupby(["emotion", "intensity"], dropna=False)
            .agg({"score": ["mean", "std", "count", "min", "max"]})
            .round(4)
        )

        summary_filename = self.output_dir / "summary_results.csv"
        summary.to_csv(summary_filename)
        self.logger.info(f"Summary results saved to {summary_filename}")

        # Compute per-repeat summary if repeat_id is present
        if "repeat_id" in df.columns and df["repeat_id"].notna().any():
            by_rep = (
                df.dropna(subset=["repeat_id"]).groupby(
                    ["emotion", "intensity", "repeat_id"], dropna=False
                )["score"].agg(["mean", "std", "count", "min", "max"]).reset_index()
            )
            by_rep = by_rep.round(6)
            by_rep_filename = self.output_dir / "summary_by_repeat.csv"
            by_rep.to_csv(by_rep_filename, index=False)
            self.logger.info(f"Per-repeat summary saved to {by_rep_filename}")

            # Compute across-repeat aggregation per (emotion,intensity)
            overall_rows = []
            for (emotion, intensity), g in by_rep.groupby(["emotion", "intensity"], dropna=False):
                # Extract per-repeat stats
                means = g["mean"].tolist()
                stds = g["std"].fillna(0.0).tolist()
                counts = g["count"].astype(int).tolist()
                R = len(means)

                # Mean of means (unweighted)
                mean_of_means = sum(means) / R if R > 0 else float("nan")

                # Between-run variance of means (sample variance)
                if R > 1:
                    between_var = sum((m - mean_of_means) ** 2 for m in means) / (R - 1)
                else:
                    between_var = 0.0

                # Pooled variance across all samples (unbiased)
                N = sum(counts)
                if N > 1:
                    # Weighted mean across all observations
                    mean_weighted = (
                        sum(n * m for n, m in zip(counts, means)) / N
                        if N > 0
                        else float("nan")
                    )
                    within = sum((n - 1) * (s ** 2 if not pd.isna(s) else 0.0) for n, s in zip(counts, stds))
                    between = sum(n * (m - mean_weighted) ** 2 for n, m in zip(counts, means))
                    pooled_var = (within + between) / (N - 1)
                else:
                    pooled_var = 0.0

                overall_rows.append(
                    {
                        "emotion": emotion,
                        "intensity": intensity,
                        "repeats": R,
                        "total_count": N,
                        "mean_of_means": round(mean_of_means, 6) if not pd.isna(mean_of_means) else mean_of_means,
                        "between_run_var": round(between_var, 6),
                        "pooled_var": round(pooled_var, 6),
                    }
                )

            overall_df = pd.DataFrame(overall_rows)
            overall_filename = self.output_dir / "summary_overall.csv"
            overall_df.to_csv(overall_filename, index=False)
            self.logger.info(f"Across-repeat summary saved to {overall_filename}")

        # Create README explaining output files
        try:
            readme_content = (
                "# Experiment Results Files\n\n"
                "This folder contains outputs from EmotionExperiment. Files:\n\n"
                "- detailed_results.csv: Item-level records including emotion, intensity, repeat_id, response, ground_truth, score.\n"
                "- raw_results.json: Full JSON dump of all records with metadata (benchmark, item metadata).\n"
                "- summary_results.csv: Aggregates per (emotion,intensity) across all repeats (mean, std, count, min, max).\n"
                "- summary_by_repeat.csv: Aggregates per (emotion,intensity,repeat_id).\n"
                "- summary_overall.csv: Across-repeat statistics per (emotion,intensity):\n"
                "  - mean_of_means: Unweighted mean of per-repeat means.\n"
                "  - between_run_var: Sample variance of per-repeat means (repeat-level stability).\n"
                "  - pooled_var: Unbiased pooled variance across all observations (law of total variance).\n"
                "- experiment_config.json: Resolved configuration and runtime info (includes repeat settings).\n\n"
                "Notes:\n"
                "- For meaningful repeat variance, enable stochastic decoding (do_sample=true, nonzero temperature/top_p).\n"
                "- Seeds: Each repeat uses random_seed = repeat_seed_base + repeat_id when supported.\n"
            )
            with open(self.output_dir / "README.md", "w") as f:
                f.write(readme_content)
        except Exception:
            pass

        # Print summary
        self.logger.info("\n=== EXPERIMENT RESULTS SUMMARY ===")
        self.logger.info(f"\n{summary}")

        return df

    def _save_experiment_config(self):
        """Save the complete experiment configuration to the results folder"""
        config_filename = self.output_dir / "experiment_config.json"

        # Convert the experiment config to a serializable dictionary
        config_dict = {
            "model_path": self.config.model_path,
            "emotions": self.config.emotions,
            "intensities": self.config.intensities,
            "benchmark": {
                "name": self.config.benchmark.name,
                "data_path": str(self.config.benchmark.get_data_path()),
                "task_type": self.config.benchmark.task_type,
                "sample_limit": self.config.benchmark.sample_limit,
            },
            "output_dir": self.config.output_dir,
            "batch_size": self.config.batch_size,
            "generation_config": self.generation_config,
            "loading_config": self._serialize_loading_config(),
            "repe_eng_config": self.repe_config,
            "max_evaluation_workers": self.config.max_evaluation_workers,
            "pipeline_queue_size": self.config.pipeline_queue_size,
            # Runtime information
            "runtime_info": {
                "timestamp": datetime.now().isoformat(),
                "sample_num": self.sample_num,
                "enable_thinking": self.enable_thinking,
                "max_context_length": self.max_context_length,
                "truncation_strategy": self.truncation_strategy,
                "hidden_layers": self.hidden_layers,
                "is_vllm": self.is_vllm,
                "repeat_runs": getattr(self, "repeat_runs", 1),
                "repeat_seed_base": getattr(self, "repeat_seed_base", None),
            },
        }

        # Save configuration
        with open(config_filename, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

        self.logger.info(f"Experiment configuration saved to {config_filename}")

    def _serialize_loading_config(self) -> dict | None:
        """Convert LoadingConfig to a serializable dictionary"""
        if self.loading_config is None:
            return None

        return {
            "model_path": self.loading_config.model_path,
            "gpu_memory_utilization": self.loading_config.gpu_memory_utilization,
            "tensor_parallel_size": self.loading_config.tensor_parallel_size,
            "max_model_len": self.loading_config.max_model_len,
            "enforce_eager": self.loading_config.enforce_eager,
            "quantization": self.loading_config.quantization,
            "trust_remote_code": self.loading_config.trust_remote_code,
            "dtype": self.loading_config.dtype,
            "seed": self.loading_config.seed,
            "disable_custom_all_reduce": self.loading_config.disable_custom_all_reduce,
            "enable_auto_truncation": self.config.benchmark.enable_auto_truncation,
            "truncation_strategy": self.config.benchmark.truncation_strategy,
            "preserve_ratio": self.config.benchmark.preserve_ratio,
        }

    def run_sanity_check(self, sample_limit: int = 5) -> pd.DataFrame:
        """Run a quick sanity check with limited samples"""
        self.logger.info(f"Running sanity check with {sample_limit} samples")

        # Temporarily modify config
        original_sample_limit = self.config.benchmark.sample_limit
        self.config.benchmark.sample_limit = sample_limit
        self.output_dir = self.output_dir.parent / "sanity_check" / self.output_dir.name
        os.makedirs(self.output_dir, exist_ok=True)

        try:
            return self.run_experiment()
        finally:
            # Restore original config
            self.config.benchmark.sample_limit = original_sample_limit
