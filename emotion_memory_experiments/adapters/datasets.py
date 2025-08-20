"""
Dataset classes for different memory benchmark suites.
Each dataset handles loading data and creating prompts for specific benchmarks.
"""

import logging
from typing import List, Optional

from torch.utils.data import Dataset

try:
    from ..data_models import BenchmarkItem
    from .truncation_utils import truncate_contexts_batch
except ImportError:
    from emotion_memory_experiments.data_models import BenchmarkItem
    from emotion_memory_experiments.adapters.truncation_utils import truncate_contexts_batch

logger = logging.getLogger(__name__)


class InfiniteBenchDataset(Dataset):
    """
    Ultra-simple PyTorch Dataset for InfiniteBench following GameScenarioDataset pattern.
    Uses prompt wrapper for proper model-specific formatting.
    Supports automatic batch context truncation.
    """

    def __init__(
        self, 
        items: List[BenchmarkItem], 
        prompt_wrapper=None,
        max_context_length: Optional[int] = None,
        tokenizer=None,
        truncation_strategy: str = "right"
    ):
        self.items = items
        self.prompt_wrapper = prompt_wrapper
        self.max_context_length = max_context_length
        self.tokenizer = tokenizer
        self.truncation_strategy = truncation_strategy
        
        # Apply batch truncation if configured
        if max_context_length and tokenizer:
            logger.info(f"Applying batch context truncation with max_length={max_context_length}, strategy='{truncation_strategy}'")
            self._apply_batch_truncation()
    
    def _apply_batch_truncation(self):
        """Apply batch truncation to all items for improved performance."""
        # Extract all contexts for batch processing
        contexts = [item.context for item in self.items if item.context]
        
        if not contexts:
            return
        
        # Batch truncate all contexts
        truncated_contexts = truncate_contexts_batch(
            contexts, self.max_context_length, self.tokenizer, self.truncation_strategy
        )
        
        # Update items with truncated contexts and add metadata
        context_idx = 0
        for item in self.items:
            if item.context:
                original_length = len(self.tokenizer.encode(item.context, add_special_tokens=False))
                truncated_context = truncated_contexts[context_idx]
                truncated_length = len(self.tokenizer.encode(truncated_context, add_special_tokens=False))
                
                # Update context
                item.context = truncated_context
                
                # Add truncation metadata if truncated
                if original_length > self.max_context_length:
                    if item.metadata is None:
                        item.metadata = {}
                    item.metadata['truncation_info'] = {
                        'original_length': original_length,
                        'truncated_length': truncated_length,
                        'strategy': self.truncation_strategy,
                        'was_truncated': True
                    }
                    logger.info(
                        f"Sample {item.id}: context truncated from {original_length} to "
                        f"{truncated_length} tokens using '{self.truncation_strategy}' strategy"
                    )
                
                context_idx += 1

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        if self.prompt_wrapper:
            # Use prompt wrapper for proper formatting (like GameScenarioDataset)
            prompt = self.prompt_wrapper(context=item.context, question=item.input_text)
            return {
                "prompt": prompt,
                "item": item,
                "context": item.context,
                "question": item.input_text,
                "ground_truth": item.ground_truth,
                "metadata": item.metadata,
            }
        else:
            # Fallback to raw item (for backward compatibility)
            return item


class LoCoMoDataset(Dataset):
    """
    Ultra-simple PyTorch Dataset for LoCoMo following GameScenarioDataset pattern.
    Uses prompt wrapper for proper model-specific formatting.
    Supports automatic batch context truncation.
    """

    def __init__(
        self, 
        items: List[BenchmarkItem], 
        prompt_wrapper=None,
        max_context_length: Optional[int] = None,
        tokenizer=None,
        truncation_strategy: str = "right"
    ):
        self.items = items
        self.prompt_wrapper = prompt_wrapper
        self.max_context_length = max_context_length
        self.tokenizer = tokenizer
        self.truncation_strategy = truncation_strategy
        
        # Apply batch truncation if configured
        if max_context_length and tokenizer:
            logger.info(f"Applying batch context truncation with max_length={max_context_length}, strategy='{truncation_strategy}'")
            self._apply_batch_truncation()
    
    def _apply_batch_truncation(self):
        """Apply batch truncation to all items for improved performance."""
        # Extract all contexts for batch processing
        contexts = [item.context for item in self.items if item.context]
        
        if not contexts:
            return
        
        # Batch truncate all contexts
        truncated_contexts = truncate_contexts_batch(
            contexts, self.max_context_length, self.tokenizer, self.truncation_strategy
        )
        
        # Update items with truncated contexts and add metadata
        context_idx = 0
        for item in self.items:
            if item.context:
                original_length = len(self.tokenizer.encode(item.context, add_special_tokens=False))
                truncated_context = truncated_contexts[context_idx]
                truncated_length = len(self.tokenizer.encode(truncated_context, add_special_tokens=False))
                
                # Update context
                item.context = truncated_context
                
                # Add truncation metadata if truncated
                if original_length > self.max_context_length:
                    if item.metadata is None:
                        item.metadata = {}
                    item.metadata['truncation_info'] = {
                        'original_length': original_length,
                        'truncated_length': truncated_length,
                        'strategy': self.truncation_strategy,
                        'was_truncated': True
                    }
                    logger.info(
                        f"Sample {item.id}: context truncated from {original_length} to "
                        f"{truncated_length} tokens using '{self.truncation_strategy}' strategy"
                    )
                
                context_idx += 1

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        if self.prompt_wrapper:
            # Use prompt wrapper for conversational QA formatting
            prompt = self.prompt_wrapper(context=item.context, question=item.input_text)
            return {
                "prompt": prompt,
                "item": item,
                "context": item.context,
                "question": item.input_text,
                "ground_truth": item.ground_truth,
                "metadata": item.metadata,
            }
        else:
            # Fallback to raw item (for backward compatibility)
            return item


class LongBenchDataset(Dataset):
    """
    Ultra-simple PyTorch Dataset for LongBench following GameScenarioDataset pattern.
    Uses prompt wrapper for proper model-specific formatting.
    Supports automatic batch context truncation.
    """

    def __init__(
        self, 
        items: List[BenchmarkItem], 
        prompt_wrapper=None,
        max_context_length: Optional[int] = None,
        tokenizer=None,
        truncation_strategy: str = "right"
    ):
        self.items = items
        self.prompt_wrapper = prompt_wrapper
        self.max_context_length = max_context_length
        self.tokenizer = tokenizer
        self.truncation_strategy = truncation_strategy
        
        # Apply batch truncation if configured
        if max_context_length and tokenizer:
            logger.info(f"Applying batch context truncation with max_length={max_context_length}, strategy='{truncation_strategy}'")
            self._apply_batch_truncation()
    
    def _apply_batch_truncation(self):
        """Apply batch truncation to all items for improved performance."""
        # Extract all contexts for batch processing
        contexts = [item.context for item in self.items if item.context]
        
        if not contexts:
            return
        
        # Batch truncate all contexts
        truncated_contexts = truncate_contexts_batch(
            contexts, self.max_context_length, self.tokenizer, self.truncation_strategy
        )
        
        # Update items with truncated contexts and add metadata
        context_idx = 0
        for item in self.items:
            if item.context:
                original_length = len(self.tokenizer.encode(item.context, add_special_tokens=False))
                truncated_context = truncated_contexts[context_idx]
                truncated_length = len(self.tokenizer.encode(truncated_context, add_special_tokens=False))
                
                # Update context
                item.context = truncated_context
                
                # Add truncation metadata if truncated
                if original_length > self.max_context_length:
                    if item.metadata is None:
                        item.metadata = {}
                    item.metadata['truncation_info'] = {
                        'original_length': original_length,
                        'truncated_length': truncated_length,
                        'strategy': self.truncation_strategy,
                        'was_truncated': True
                    }
                    logger.info(
                        f"Sample {item.id}: context truncated from {original_length} to "
                        f"{truncated_length} tokens using '{self.truncation_strategy}' strategy"
                    )
                
                context_idx += 1

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        if self.prompt_wrapper:
            # Use prompt wrapper for long context QA formatting
            prompt = self.prompt_wrapper(context=item.context, question=item.input_text)
            return {
                "prompt": prompt,
                "item": item,
                "context": item.context,
                "question": item.input_text,
                "ground_truth": item.ground_truth,
                "metadata": item.metadata,
            }
        else:
            # Fallback to raw item (for backward compatibility)
            return item


def collate_memory_benchmarks(batch):
    """
    Custom collate function for memory benchmark datasets following GameScenarioDataset pattern.

    Args:
        batch: List of items from dataset (either raw BenchmarkItem or formatted dict)

    Returns:
        Collated batch with proper structure for pipeline processing
    """
    # Check if items are formatted (with prompt wrapper) or raw
    if batch and isinstance(batch[0], dict) and "prompt" in batch[0]:
        # Items are formatted with prompt wrapper
        return {
            "prompts": [item["prompt"] for item in batch],
            "items": [item["item"] for item in batch],
            "contexts": [item["context"] for item in batch],
            "questions": [item["question"] for item in batch],
            "ground_truths": [item["ground_truth"] for item in batch],
            "metadata": [item["metadata"] for item in batch],
            "batch_size": len(batch),
        }
    else:
        # Items are raw BenchmarkItem objects (backward compatibility)
        return {"items": batch, "batch_size": len(batch)}