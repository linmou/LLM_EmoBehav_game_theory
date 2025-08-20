"""
Simplified utilities for truncating long contexts using Transformers tokenizer built-in capabilities.
"""

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def calculate_max_context_length(
    max_model_len: int,
    preserve_ratio: float = 0.95,
    prompt_overhead: int = 200
) -> int:
    """
    Calculate maximum context length accounting for model limits and overhead.
    
    Args:
        max_model_len: Model's maximum sequence length
        preserve_ratio: Ratio of max_model_len to use for context
        prompt_overhead: Reserved tokens for prompt template and generation
    
    Returns:
        Maximum allowed context length in tokens
    """
    # Reserve space for prompt template and generation
    effective_max = int(max_model_len * preserve_ratio) - prompt_overhead
    
    # Ensure minimum context length
    return max(effective_max, 1000)


def truncate_context(
    context: str,
    max_length: int,
    tokenizer,
    strategy: str = "right",
    return_token_counts: bool = False
) -> Tuple[str, Optional[Tuple[int, int]]]:
    """
    Truncate context using Transformers tokenizer built-in truncation.
    
    Args:
        context: Text to truncate
        max_length: Maximum token length
        tokenizer: Tokenizer to use for truncation
        strategy: Truncation strategy ("right" or "left")
        return_token_counts: Whether to return original and truncated token counts
    
    Returns:
        Tuple of (truncated_context, (original_length, truncated_length) if requested)
    """
    if not context:
        return context, (0, 0) if return_token_counts else None
    
    # Get original length if needed
    original_length = None
    if return_token_counts:
        original_tokens = tokenizer.encode(context, add_special_tokens=False)
        original_length = len(original_tokens)
        
        # No truncation needed
        if original_length <= max_length:
            return context, (original_length, original_length)
    
    # Set truncation side
    if strategy not in ["right", "left"]:
        raise ValueError(f"Strategy must be 'right' or 'left', got: {strategy}")
    
    tokenizer.truncation_side = strategy
    
    # Use tokenizer's built-in truncation
    result = tokenizer(
        context,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
        return_tensors=None
    )
    
    # Decode back to text
    truncated_context = tokenizer.decode(result["input_ids"], skip_special_tokens=True)
    
    # Calculate lengths for logging
    if return_token_counts:
        truncated_length = len(result["input_ids"])
        
        # Log truncation event if it occurred
        if original_length and original_length > max_length:
            logger.info(
                f"Context truncated: original length={original_length} tokens, "
                f"truncated to {truncated_length} tokens using '{strategy}' strategy"
            )
        
        return truncated_context, (original_length or truncated_length, truncated_length)
    
    return truncated_context, None


def truncate_contexts_batch(
    contexts: List[str],
    max_length: int,
    tokenizer,
    strategy: str = "right"
) -> List[str]:
    """
    Batch truncate multiple contexts for improved performance.
    
    Args:
        contexts: List of texts to truncate
        max_length: Maximum token length
        tokenizer: Tokenizer to use for truncation
        strategy: Truncation strategy ("right" or "left")
    
    Returns:
        List of truncated contexts
    """
    if not contexts:
        return []
    
    # Set truncation side
    if strategy not in ["right", "left"]:
        raise ValueError(f"Strategy must be 'right' or 'left', got: {strategy}")
    
    tokenizer.truncation_side = strategy
    
    # Batch process all contexts
    results = tokenizer(
        contexts,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
        padding=False,
        return_tensors=None
    )
    
    # Decode all contexts
    truncated_contexts = [
        tokenizer.decode(ids, skip_special_tokens=True) 
        for ids in results["input_ids"]
    ]
    
    # Log batch truncation summary
    original_lengths = [len(tokenizer.encode(ctx, add_special_tokens=False)) for ctx in contexts]
    truncated_count = sum(1 for orig_len in original_lengths if orig_len > max_length)
    
    if truncated_count > 0:
        logger.info(
            f"Batch truncation: {truncated_count}/{len(contexts)} contexts truncated "
            f"to max {max_length} tokens using '{strategy}' strategy"
        )
    
    return truncated_contexts


def truncate_item_context(
    item,
    max_context_length: int,
    tokenizer,
    strategy: str = "right"
):
    """
    Truncate context in a BenchmarkItem using tokenizer built-in truncation.
    
    Args:
        item: BenchmarkItem with context to truncate
        max_context_length: Maximum context length in tokens
        tokenizer: Tokenizer for truncation
        strategy: Truncation strategy ("right" or "left")
    
    Returns:
        Modified item with truncated context
    """
    if not item.context or not max_context_length:
        return item
    
    truncated_context, token_counts = truncate_context(
        item.context,
        max_context_length,
        tokenizer,
        strategy,
        return_token_counts=True
    )
    
    if token_counts and token_counts[0] != token_counts[1]:
        # Context was truncated
        original_length, truncated_length = token_counts
        logger.info(
            f"Sample {item.id}: context truncated from {original_length} to "
            f"{truncated_length} tokens using '{strategy}' strategy"
        )
        
        # Update item's context
        item.context = truncated_context
        
        # Store truncation info in metadata
        if item.metadata is None:
            item.metadata = {}
        item.metadata['truncation_info'] = {
            'original_length': original_length,
            'truncated_length': truncated_length,
            'strategy': strategy,
            'was_truncated': True
        }
    
    return item