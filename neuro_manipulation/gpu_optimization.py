import torch
import gc
import numpy as np
import logging
import time
from functools import partial
from typing import Callable, Dict, Any, Optional, List, Tuple, Union

class BatchSizeFinder:
    """Finds the optimal batch size that fits in GPU memory.
    
    This utility helps find the maximum batch size that can be used without
    encountering out-of-memory (OOM) errors when running inference with a large model.
    
    Args:
        mode: Search strategy to use ('power' or 'binsearch')
        init_val: Initial batch size to start the search
        max_trials: Maximum number of trials before stopping
        safety_margin: Fraction of max GPU memory to target (0.9 = 90%)
        reduction_factor: Factor to reduce final batch size by for extra safety
    
    Example::
    
        finder = BatchSizeFinder()
        optimal_batch_size = finder.find(
            model=my_model,
            sample_input_fn=lambda bs: {"input_ids": torch.ones(bs, 10).long(), 
                                        "attention_mask": torch.ones(bs, 10)}
        )
    """
    
    SUPPORTED_MODES = ["power", "binsearch"]
    
    def __init__(
        self,
        mode: str = "binsearch",
        init_val: int = 1,
        max_trials: int = 25,
        safety_margin: float = 0.9,
        reduction_factor: float = 0.95,
    ):
        mode = mode.lower()
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Mode '{mode}' not supported. Choose from {self.SUPPORTED_MODES}")
            
        self.mode = mode
        self.init_val = init_val
        self.max_trials = max_trials
        self.safety_margin = safety_margin
        self.reduction_factor = reduction_factor
        self.logger = logging.getLogger(__name__)
        
    def find(
        self,
        model: torch.nn.Module,
        sample_input_fn: Callable[[int], Dict[str, torch.Tensor]],
        generation_kwargs: Optional[Dict[str, Any]] = None,
        forward_only: bool = False,
    ) -> int:
        """Find the optimal batch size.
        
        Args:
            model: The model to test batch sizes with
            sample_input_fn: Function that takes a batch size and returns inputs for the model
            generation_kwargs: Optional kwargs for model.generate() if using text generation
            forward_only: If True, only test model.forward() not model.generate()
            
        Returns:
            The optimal batch size
        """
        # Make sure model is in the right state
        model.eval()
        
        # Get GPU specs for all available GPUs
        num_gpus = torch.cuda.device_count()
        total_gpu_mem = 0
        gpu_info = []
        
        for gpu_id in range(num_gpus):
            gpu_mem = torch.cuda.get_device_properties(gpu_id).total_memory
            total_gpu_mem += gpu_mem
            gpu_info.append({
                'id': gpu_id,
                'total_memory': gpu_mem,
                'name': torch.cuda.get_device_properties(gpu_id).name
            })
            
        # Log info about GPUs
        self.logger.info(f"Found {num_gpus} GPUs")
        for gpu in gpu_info:
            self.logger.info(f"GPU {gpu['id']}: {gpu['name']} with {gpu['total_memory'] / 1024**3:.2f} GB")
        self.logger.info(f"Total GPU memory across all devices: {total_gpu_mem / 1024**3:.2f} GB")
        
        # Target memory (with safety margin)
        target_mem = total_gpu_mem * self.safety_margin
        self.logger.info(f"Target memory usage (with {self.safety_margin*100:.1f}% safety margin): {target_mem / 1024**3:.2f} GB")
        
        # Find optimal batch size based on chosen strategy
        if self.mode == "power":
            optimal_batch = self._power_search(model, sample_input_fn, generation_kwargs, forward_only, target_mem)
        else:  # binsearch
            optimal_batch = self._binary_search(model, sample_input_fn, generation_kwargs, forward_only, target_mem)
            
        # Apply reduction factor for safety
        final_batch_size = max(1, int(optimal_batch * self.reduction_factor))
        self.logger.info(f"Found optimal batch size: {optimal_batch}, using {final_batch_size} for safety")
        
        return final_batch_size
        
    def _power_search(
        self, 
        model: torch.nn.Module,
        sample_input_fn: Callable[[int], Dict[str, torch.Tensor]],
        generation_kwargs: Optional[Dict[str, Any]],
        forward_only: bool,
        target_mem: int
    ) -> int:
        """Search by repeatedly doubling batch size until OOM."""
        batch_size = self.init_val
        optimal_batch = self.init_val
        
        for trial in range(self.max_trials):
            self.logger.info(f"Trial {trial+1}: Testing batch size {batch_size}")
            
            try:
                peak_mem = self._measure_memory(model, sample_input_fn, batch_size, generation_kwargs, forward_only)
                self.logger.info(f"Batch size {batch_size} uses {peak_mem / 1024**3:.2f} GB")
                
                if peak_mem > target_mem:
                    # Too much memory, return previous size
                    break
                
                # This size worked, save it and try a larger one
                optimal_batch = batch_size
                batch_size *= 2
                
            except RuntimeError as e:
                # Out of memory error
                if "CUDA out of memory" in str(e):
                    self.logger.warning(f"OOM at batch size {batch_size}")
                    break
                else:
                    # Re-raise other runtime errors
                    raise e
                    
        return optimal_batch
        
    def _binary_search(
        self, 
        model: torch.nn.Module,
        sample_input_fn: Callable[[int], Dict[str, torch.Tensor]],
        generation_kwargs: Optional[Dict[str, Any]],
        forward_only: bool,
        target_mem: int
    ) -> int:
        """Search using binary search algorithm after initial doubling."""
        # Start with power search to get upper bound
        batch_size = self.init_val
        optimal_batch = self.init_val
        upper_bound = None
        
        # First phase: double batch size until we hit OOM
        for trial in range(self.max_trials // 2):
            self.logger.info(f"Power phase - Trial {trial+1}: Testing batch size {batch_size}")
            
            try:
                peak_mem = self._measure_memory(model, sample_input_fn, batch_size, generation_kwargs, forward_only)
                self.logger.info(f"Batch size {batch_size} uses {peak_mem / 1024**3:.2f} GB")
                
                if peak_mem > target_mem:
                    # Too much memory, use this as upper bound
                    upper_bound = batch_size
                    break
                
                # This size worked, save it and try a larger one
                optimal_batch = batch_size
                batch_size *= 2
                
            except RuntimeError as e:
                # Out of memory error
                if "CUDA out of memory" in str(e):
                    self.logger.warning(f"OOM at batch size {batch_size}")
                    upper_bound = batch_size
                    break
                else:
                    # Re-raise other runtime errors
                    raise e
        
        # If we didn't hit an upper bound, return the last successful batch size
        if upper_bound is None:
            return optimal_batch
            
        # Second phase: binary search between optimal_batch and upper_bound
        lower_bound = optimal_batch
        
        for trial in range(self.max_trials // 2):
            # If bounds are adjacent, we're done
            if upper_bound - lower_bound <= 1:
                break
                
            # Try middle point
            mid_batch = (lower_bound + upper_bound) // 2
            self.logger.info(f"Binary phase - Trial {trial+1}: Testing batch size {mid_batch}")
            
            try:
                peak_mem = self._measure_memory(model, sample_input_fn, mid_batch, generation_kwargs, forward_only)
                self.logger.info(f"Batch size {mid_batch} uses {peak_mem / 1024**3:.2f} GB")
                
                if peak_mem > target_mem:
                    # Too much memory, adjust upper bound
                    upper_bound = mid_batch
                else:
                    # This size worked, adjust lower bound
                    lower_bound = mid_batch
                    optimal_batch = mid_batch
                    
            except RuntimeError as e:
                # Out of memory error
                if "CUDA out of memory" in str(e):
                    self.logger.warning(f"OOM at batch size {mid_batch}")
                    upper_bound = mid_batch
                else:
                    # Re-raise other runtime errors
                    raise e
                    
        return optimal_batch
                
    def _measure_memory(
        self, 
        model: torch.nn.Module,
        sample_input_fn: Callable[[int], Dict[str, torch.Tensor]],
        batch_size: int,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        forward_only: bool = False
    ) -> int:
        """Measure peak memory usage for a given batch size across all GPUs."""
        # Clear memory on all devices
        torch.cuda.empty_cache()
        gc.collect()
        
        # Get number of GPUs
        num_gpus = torch.cuda.device_count()
        
        # Reset peak memory stats on all devices and synchronize
        for device_id in range(num_gpus):
            with torch.cuda.device(device_id):
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
        
        # Create inputs
        inputs = sample_input_fn(batch_size)
        
        # Handle inputs for multi-GPU models
        # When using device_map="auto", inputs should be on the same device as the first layer
        try:
            # Try to get the device of the first parameter
            # This might be imperfect for device_map="auto", but we'll let the model handle distribution
            first_param = next(model.parameters())
            device = first_param.device
            
            # If the model is distributed with device_map="auto", we respect that
            # and rely on the model's forward/generate methods to handle the distribution
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        except StopIteration:
            # If model has no parameters (unlikely), default to cuda:0
            device = torch.device('cuda:0')
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Run model
        with torch.no_grad():
            if forward_only:
                _ = model(**inputs)
            else:
                generation_kwargs = generation_kwargs or {}
                _ = model.generate(**inputs, **generation_kwargs)
        
        # Synchronize all devices
        for device_id in range(num_gpus):
            with torch.cuda.device(device_id):
                torch.cuda.synchronize()
        
        # Measure peak memory across all devices
        total_peak_mem = 0
        for device_id in range(num_gpus):
            with torch.cuda.device(device_id):
                peak_mem = torch.cuda.max_memory_allocated()
                self.logger.debug(f"GPU {device_id} peak memory: {peak_mem / 1024**3:.2f} GB")
                total_peak_mem += peak_mem
        
        # Clean up
        torch.cuda.empty_cache()
        gc.collect()
        
        return total_peak_mem


def find_optimal_batch_size_for_llm(
    model: torch.nn.Module,
    tokenizer,
    sample_text: str = "This is a sample input to test batch size.",
    max_length: int = 128,
    generation_kwargs: Optional[Dict[str, Any]] = None,
    mode: str = "binsearch",
    init_val: int = 1,
    safety_margin: float = 0.9,
) -> int:
    """Convenience function to find optimal batch size for a language model.
    
    Supports both single-GPU and multi-GPU setups, including models loaded with device_map="auto".
    
    Args:
        model: The language model
        tokenizer: The tokenizer for the model
        sample_text: Sample text to use for testing
        max_length: Maximum sequence length to pad to
        generation_kwargs: Parameters for model.generate()
        mode: Search strategy ('power' or 'binsearch')
        init_val: Initial batch size
        safety_margin: Fraction of GPU memory to target
        
    Returns:
        Optimal batch size
    """
    logger = logging.getLogger(__name__)
    
    # Detect if model is using multiple GPUs (with device_map="auto" or similar)
    try:
        devices = set()
        for name, param in model.named_parameters():
            devices.add(param.device)
        
        multi_gpu = len(devices) > 1
        if multi_gpu:
            device_list = sorted(list(devices), key=lambda d: d.index if hasattr(d, 'index') else 0)
            logger.info(f"Model is distributed across multiple devices: {device_list}")
    except:
        # If we can't determine, assume it might be multi-GPU capable
        multi_gpu = torch.cuda.device_count() > 1
        logger.info(f"Could not determine model distribution. Multiple GPUs available: {multi_gpu}")
    
    # Create a sample input function
    def sample_input_fn(batch_size):
        # Tokenize the input text
        encoded = tokenizer(
            [sample_text] * batch_size,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )
        return encoded
    
    # Create finder and run search
    finder = BatchSizeFinder(
        mode=mode,
        init_val=init_val,
        safety_margin=safety_margin
    )
    
    return finder.find(
        model=model,
        sample_input_fn=sample_input_fn,
        generation_kwargs=generation_kwargs
    )


def find_optimal_batch_size_for_experiment(model, tokenizer, prompt_wrapper, game_config, exp_config, safety_margin=0.9):
    """
    Find the maximum batch size that fits in GPU memory for the given experiment setup.
    Uses binary search.
    
    This is a convenience wrapper around BatchSizeFinder for backwards compatibility.
    Supports both single-GPU and multi-GPU setups, including models loaded with device_map="auto".
    
    Args:
        model: The LLM model
        tokenizer: The tokenizer
        prompt_wrapper: Function to create prompts based on scenarios
        game_config: Game configuration dictionary
        exp_config: Experiment configuration dictionary
        safety_margin: Fraction of max memory to target (0.9 = 90%)
        
    Returns:
        The estimated optimal batch size
    """
    from torch.utils.data import DataLoader
    from neuro_manipulation.datasets.game_scenario_dataset import GameScenarioDataset, collate_game_scenarios
    
    logger = logging.getLogger(__name__)
    logger.info("Starting batch size estimation for experiment (using BatchSizeFinder)")
    
    # Detect if model is using multiple GPUs (with device_map="auto" or similar)
    try:
        devices = set()
        for name, param in model.named_parameters():
            devices.add(param.device)
        
        multi_gpu = len(devices) > 1
        if multi_gpu:
            device_list = sorted(list(devices), key=lambda d: d.index if hasattr(d, 'index') else 0)
            logger.info(f"Model is distributed across multiple devices: {device_list}")
    except:
        # If we can't determine, assume it might be multi-GPU capable
        multi_gpu = torch.cuda.device_count() > 1
        logger.info(f"Could not determine model distribution. Multiple GPUs available: {multi_gpu}")
    
    # Create a minimal dataset to get a sample
    temp_dataset = GameScenarioDataset(
        game_config,
        partial(prompt_wrapper.__call__,
                user_messages=exp_config['experiment']['system_message_template']),
        sample_num=1,
    )
    temp_loader = DataLoader(temp_dataset, batch_size=1, shuffle=False, collate_fn=collate_game_scenarios)
    sample_batch = next(iter(temp_loader))
    
    # Create a sample prompt
    sample_prompt = sample_batch['prompt'][0]
    generation_config = exp_config['experiment']['llm']['generation_config']
    
    # Create a sample_input_fn using the tokenizer and sample_prompt
    def sample_input_fn(batch_size):
        batch_prompts = [sample_prompt] * batch_size
        inputs = tokenizer(batch_prompts, padding=True, truncation=True, return_tensors="pt")
        return inputs
    
    # Create finder and run search
    finder = BatchSizeFinder(
        mode="binsearch",
        safety_margin=safety_margin
    )
    
    # Pass generation config to the finder
    generation_kwargs = {
        "max_new_tokens": generation_config.get('max_new_tokens', 512),
        "do_sample": generation_config.get('do_sample', True),
        "temperature": generation_config.get('temperature', 0.7),
        "top_p": generation_config.get('top_p', 0.95)
    }
    
    return finder.find(
        model=model,
        sample_input_fn=sample_input_fn,
        generation_kwargs=generation_kwargs
    )


def measure_throughput(model, tokenizer, sample_text, max_length=128, batch_size=1, num_batches=3):
    """Measure throughput (samples/second) for a given batch size"""
    torch.cuda.empty_cache()
    gc.collect()
    
    # Tokenize the input text once
    encoded = tokenizer(
        [sample_text] * batch_size,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )
    
    # Move encoded inputs to the correct device
    if hasattr(encoded, 'to'):
        # If the tokenizer returns an object with a to() method (like a BatchEncoding)
        encoded = encoded.to(model.device)
    else:
        # If it's a regular dict, move each tensor to the device
        encoded = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in encoded.items()}
    
    # Warmup
    with torch.no_grad():
        _ = model.generate(**encoded, max_new_tokens=20)
    
    # Measure throughput
    start_time = time.time()
    
    for _ in range(num_batches):
        with torch.no_grad():
            _ = model.generate(**encoded, max_new_tokens=20)
    
    end_time = time.time()
    
    total_samples = batch_size * num_batches
    throughput = total_samples / (end_time - start_time)
    
    return throughput 