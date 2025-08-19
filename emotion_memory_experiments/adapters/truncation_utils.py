"""
Utilities for truncating long contexts to fit within model's maximum length.
"""

import logging
from typing import Optional, Tuple

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
    Truncate context to fit within maximum token length.
    
    Args:
        context: Text to truncate
        max_length: Maximum token length
        tokenizer: Tokenizer to count tokens
        strategy: Truncation strategy ("right", "left", or "middle")
        return_token_counts: Whether to return original and truncated token counts
    
    Returns:
        Tuple of (truncated_context, (original_length, truncated_length) if requested)
    """
    if not context:
        return context, (0, 0) if return_token_counts else None
    
    # Tokenize to get exact token count
    tokens = tokenizer.encode(context, add_special_tokens=False)
    original_length = len(tokens)
    
    # No truncation needed
    if original_length <= max_length:
        return context, (original_length, original_length) if return_token_counts else None
    
    # Apply truncation based on strategy
    if strategy == "right":
        # Keep beginning, truncate end
        truncated_tokens = tokens[:max_length]
        truncated_context = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
    elif strategy == "left":
        # Keep end, truncate beginning
        truncated_tokens = tokens[-max_length:]
        truncated_context = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
    elif strategy == "middle":
        # Keep beginning and end, remove middle
        keep_start = max_length // 2
        keep_end = max_length - keep_start
        
        start_tokens = tokens[:keep_start]
        end_tokens = tokens[-keep_end:]
        
        # Add ellipsis marker in the middle
        truncated_tokens = start_tokens + end_tokens
        start_text = tokenizer.decode(start_tokens, skip_special_tokens=True)
        end_text = tokenizer.decode(end_tokens, skip_special_tokens=True)
        truncated_context = f"{start_text}\n[...content truncated...]\n{end_text}"
        
    else:
        raise ValueError(f"Unknown truncation strategy: {strategy}")
    
    truncated_length = len(tokenizer.encode(truncated_context, add_special_tokens=False))
    
    # Log truncation event
    logger.info(
        f"Context truncated: original length={original_length} tokens, "
        f"truncated to {truncated_length} tokens using '{strategy}' strategy"
    )
    
    if return_token_counts:
        return truncated_context, (original_length, truncated_length)
    return truncated_context, None


def truncate_item_context(
    item,
    max_context_length: int,
    tokenizer,
    strategy: str = "right"
):
    """
    Truncate context in a BenchmarkItem.
    
    Args:
        item: BenchmarkItem with context to truncate
        max_context_length: Maximum context length in tokens
        tokenizer: Tokenizer for counting tokens
        strategy: Truncation strategy
    
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