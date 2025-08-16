"""
Async vLLM Wrapper for Graceful Degradation

This module provides an async wrapper around vLLM to enable:
- Non-blocking execution with timeouts
- Graceful handling of stuck requests
- Proper resource cleanup

Date: 2025-08-10
"""

import asyncio
import logging
import time
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

logger = logging.getLogger(__name__)


class AsyncVLLMWrapper:
    """
    Async wrapper for vLLM operations with timeout protection.
    
    This wrapper runs vLLM calls in a thread pool executor to prevent
    blocking the main event loop and enables timeout-based cancellation.
    """
    
    def __init__(
        self,
        vllm_hook: Any,
        default_timeout: int = 60,
        max_workers: int = 3,  # Limited to prevent thread exhaustion
        reset_interval: int = 300,  # Stage 2: Reset interval in seconds
        rejection_start_threshold: float = 0.7  # Stage 2: Start probabilistic rejection
    ):
        """
        Initialize the async vLLM wrapper.
        
        Args:
            vllm_hook: The vLLM hook function to wrap
            default_timeout: Default timeout in seconds for vLLM operations
            max_workers: Maximum number of worker threads
            reset_interval: Interval in seconds to reset abandoned thread counter
            rejection_start_threshold: Capacity threshold to start probabilistic rejection
        """
        self.vllm_hook = vllm_hook
        self.default_timeout = default_timeout
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.timeout_requests = 0
        self.failed_requests = 0
        self.total_processing_time = 0.0
        
        # Track abandoned threads
        self.abandoned_threads = 0
        self.max_abandoned_threads = max_workers * 2  # Warning threshold
        
        # Stage 2 additions: periodic reset and graduated response
        self.last_reset_time = time.time()
        self.reset_interval = reset_interval
        self.rejection_start_threshold = rejection_start_threshold
        
        logger.info(
            f"AsyncVLLMWrapper initialized with timeout={default_timeout}s, "
            f"max_workers={max_workers}, reset_interval={self.reset_interval}s"
        )
    
    async def generate_async(
        self,
        text_inputs: List[str],
        activations: Any,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        timeout: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Generate text asynchronously with timeout protection.
        
        Args:
            text_inputs: List of input prompts
            activations: Emotion activation vectors
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            timeout: Override timeout in seconds
            **kwargs: Additional vLLM parameters
            
        Returns:
            vLLM generation output
            
        Raises:
            HTTPException: On timeout or generation failure
        """
        timeout = timeout or self.default_timeout
        self.total_requests += 1
        start_time = time.time()
        
        # Stage 2: Periodic reset of abandoned threads counter
        current_time = time.time()
        if current_time - self.last_reset_time > self.reset_interval:
            old_count = self.abandoned_threads
            self.abandoned_threads = 0
            self.last_reset_time = current_time
            if old_count > 0:
                logger.info(f"Reset abandoned threads counter from {old_count} to 0 after {self.reset_interval}s")
        
        # Stage 2: Graduated response based on capacity
        capacity_used = self.abandoned_threads / self.max_abandoned_threads
        
        if capacity_used >= 1.0:
            # Still reject all at 100% capacity
            raise HTTPException(
                status_code=503,
                detail=f"Server at full capacity ({self.abandoned_threads}/{self.max_abandoned_threads} abandoned threads)"
            )
        elif capacity_used >= self.rejection_start_threshold:
            # Probabilistic rejection between 70% and 100% capacity
            # At 70%: 0% rejection probability
            # At 100%: 100% rejection probability
            rejection_probability = (capacity_used - self.rejection_start_threshold) / (1.0 - self.rejection_start_threshold)
            
            if random.random() < rejection_probability:
                logger.info(
                    f"Probabilistic rejection at {capacity_used:.0%} capacity "
                    f"(rejection probability: {rejection_probability:.0%})"
                )
                raise HTTPException(
                    status_code=503,
                    detail=f"Server at high capacity ({capacity_used:.0%}). Please retry shortly."
                )
        
        # Get event loop for running in executor
        loop = asyncio.get_event_loop()
        
        try:
            # Log request details
            logger.debug(
                f"Starting async vLLM generation: "
                f"inputs={len(text_inputs)}, max_tokens={max_new_tokens}, "
                f"timeout={timeout}s"
            )
            
            # Create task for the blocking operation
            future = loop.run_in_executor(
                self.executor,
                self._blocking_generate,
                text_inputs,
                activations,
                max_new_tokens,
                temperature,
                top_p,
                kwargs
            )
            
            # Wait with timeout
            result = await asyncio.wait_for(future, timeout=timeout)
            
            # Update metrics
            self.successful_requests += 1
            self.total_processing_time += time.time() - start_time
            
            logger.debug(f"vLLM generation completed in {time.time() - start_time:.2f}s")
            return result
            
        except asyncio.TimeoutError:
            # Handle timeout
            self.timeout_requests += 1
            self.abandoned_threads += 1
            elapsed = time.time() - start_time
            
            capacity_after = self.abandoned_threads / self.max_abandoned_threads
            logger.error(
                f"vLLM generation timed out after {elapsed:.2f}s "
                f"(timeout={timeout}s). Abandoned threads: {self.abandoned_threads}/{self.max_abandoned_threads} "
                f"(capacity: {capacity_after:.0%})"
            )
            
            # Check if too many threads abandoned
            if self.abandoned_threads >= self.max_abandoned_threads:
                logger.critical(
                    f"Too many abandoned threads ({self.abandoned_threads}). "
                    f"Server needs restart."
                )
                raise HTTPException(
                    status_code=503,
                    detail="Server overloaded with abandoned requests. Please retry later."
                )
            
            # Note: We can't truly cancel the thread, but we abandon it
            # The thread will continue running but its result will be ignored
            raise HTTPException(
                status_code=504,
                detail=f"Request timed out after {timeout} seconds. "
                       f"Server is under heavy load."
            )
            
        except Exception as e:
            # Handle other errors
            self.failed_requests += 1
            elapsed = time.time() - start_time
            
            logger.error(
                f"vLLM generation failed after {elapsed:.2f}s: {str(e)}",
                exc_info=True
            )
            
            raise HTTPException(
                status_code=500,
                detail=f"Text generation failed: {str(e)}"
            )
    
    def _blocking_generate(
        self,
        text_inputs: List[str],
        activations: Any,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        additional_kwargs: Dict[str, Any]
    ) -> Any:
        """
        The actual blocking vLLM call.
        
        This runs in a separate thread to avoid blocking the event loop.
        """
        # Merge standard parameters with additional kwargs
        params = {
            "text_inputs": text_inputs,
            "activations": activations,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "operator": "linear_comb",
            "normalize": False,
            "token_pos": None,
        }
        params.update(additional_kwargs)
        
        # Call the actual vLLM hook
        return self.vllm_hook(**params)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get wrapper statistics for monitoring.
        
        Returns:
            Dictionary containing performance metrics
        """
        success_rate = (
            self.successful_requests / self.total_requests * 100
            if self.total_requests > 0 else 0
        )
        
        timeout_rate = (
            self.timeout_requests / self.total_requests * 100
            if self.total_requests > 0 else 0
        )
        
        avg_processing_time = (
            self.total_processing_time / self.successful_requests
            if self.successful_requests > 0 else 0
        )
        
        # Calculate time until next reset
        time_since_reset = time.time() - self.last_reset_time
        time_until_reset = max(0, self.reset_interval - time_since_reset)
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "timeout_requests": self.timeout_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round(success_rate, 2),
            "timeout_rate": round(timeout_rate, 2),
            "average_processing_time": round(avg_processing_time, 2),
            "executor_threads": self.executor._max_workers,
            "abandoned_threads": self.abandoned_threads,
            "thread_capacity_used": round(
                self.abandoned_threads / self.max_abandoned_threads * 100, 2
            ),
            # Stage 2 metrics
            "time_until_reset": round(time_until_reset, 0),
            "rejection_threshold": round(self.rejection_start_threshold * 100, 0),
        }
    
    async def shutdown(self):
        """
        Gracefully shutdown the wrapper.
        
        Waits for pending operations and cleans up resources.
        """
        logger.info("Shutting down AsyncVLLMWrapper...")
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info(
            f"AsyncVLLMWrapper shutdown complete. "
            f"Processed {self.total_requests} total requests."
        )


# Singleton instance management
_wrapper_instance: Optional[AsyncVLLMWrapper] = None


def get_async_vllm_wrapper() -> Optional[AsyncVLLMWrapper]:
    """Get the singleton AsyncVLLMWrapper instance."""
    return _wrapper_instance


def initialize_async_vllm_wrapper(
    vllm_hook: Any,
    default_timeout: int = 60,
    max_workers: int = 4,
    reset_interval: int = 300,
    rejection_start_threshold: float = 0.7
) -> AsyncVLLMWrapper:
    """
    Initialize the singleton AsyncVLLMWrapper instance.
    
    Args:
        vllm_hook: The vLLM hook to wrap
        default_timeout: Default timeout in seconds
        max_workers: Maximum worker threads
        reset_interval: Interval in seconds to reset abandoned thread counter
        rejection_start_threshold: Capacity threshold to start probabilistic rejection
        
    Returns:
        The initialized wrapper instance
    """
    global _wrapper_instance
    
    if _wrapper_instance is not None:
        logger.warning("AsyncVLLMWrapper already initialized, returning existing instance")
        return _wrapper_instance
    
    _wrapper_instance = AsyncVLLMWrapper(
        vllm_hook, default_timeout, max_workers, reset_interval, rejection_start_threshold
    )
    return _wrapper_instance