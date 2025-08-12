"""
Request Queue Manager for Graceful Degradation

This module provides intelligent request queuing to prevent server overload
by limiting concurrent requests and providing backpressure.

Date: 2025-08-10
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from fastapi import HTTPException

logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    """Request priority levels."""
    HIGH = 1      # Health checks, admin requests
    NORMAL = 2    # Regular user requests
    LOW = 3       # Batch or background requests


@dataclass
class QueuedRequest:
    """Represents a queued request with metadata."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request: Any = None
    priority: RequestPriority = RequestPriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    future: asyncio.Future = field(default_factory=asyncio.Future)
    
    def __lt__(self, other):
        """Enable priority queue sorting (lower priority value = higher priority)."""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        # Same priority: FIFO based on timestamp
        return self.timestamp < other.timestamp


class RequestQueueManager:
    """
    Manages request queuing with intelligent load management.
    
    Features:
    - Priority-based queuing
    - Configurable queue size limits
    - Concurrent request limiting
    - Queue depth monitoring
    - Automatic request rejection when overloaded
    """
    
    def __init__(
        self,
        max_queue_size: int = 50,
        max_concurrent_requests: int = 3,
        queue_timeout: float = 300.0,  # 5 minutes
        rejection_threshold: float = 0.8  # Reject when 80% full
    ):
        """
        Initialize the request queue manager.
        
        Args:
            max_queue_size: Maximum number of requests in queue
            max_concurrent_requests: Maximum concurrent processing requests
            queue_timeout: Maximum time a request can wait in queue (seconds)
            rejection_threshold: Queue fullness threshold for rejection (0.0-1.0)
        """
        self.max_queue_size = max_queue_size
        self.max_concurrent_requests = max_concurrent_requests
        self.queue_timeout = queue_timeout
        self.rejection_threshold = rejection_threshold
        
        # Priority queue for requests
        self.queue = asyncio.PriorityQueue(maxsize=max_queue_size)
        
        # Track active requests
        self.active_requests = 0
        self.active_lock = asyncio.Lock()
        
        # Metrics
        self.total_queued = 0
        self.total_processed = 0
        self.total_rejected = 0
        self.total_timeout = 0
        self.total_queue_time = 0.0
        
        # Queue processor task
        self.processor_task = None
        self.running = False
        
        logger.info(
            f"RequestQueueManager initialized: "
            f"max_queue={max_queue_size}, max_concurrent={max_concurrent_requests}, "
            f"timeout={queue_timeout}s, rejection_threshold={rejection_threshold}"
        )
    
    async def start(self):
        """Start the queue processor."""
        if self.running:
            logger.warning("RequestQueueManager already running")
            return
            
        self.running = True
        self.processor_task = asyncio.create_task(self._process_queue())
        logger.info("RequestQueueManager started")
    
    async def stop(self):
        """Stop the queue processor gracefully."""
        if not self.running:
            return
            
        logger.info("Stopping RequestQueueManager...")
        self.running = False
        
        # Cancel all pending requests
        while not self.queue.empty():
            try:
                _, queued_request = self.queue.get_nowait()
                queued_request.future.cancel()
            except asyncio.QueueEmpty:
                break
        
        # Wait for processor to finish
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("RequestQueueManager stopped")
    
    async def submit_request(
        self,
        request: Any,
        priority: RequestPriority = RequestPriority.NORMAL
    ) -> asyncio.Future:
        """
        Submit a request to the queue.
        
        Args:
            request: The request object to queue
            priority: Request priority level
            
        Returns:
            Future that will contain the processing result
            
        Raises:
            HTTPException: If queue is full or at capacity
        """
        # Check if we should reject based on queue depth
        current_size = self.queue.qsize()
        if current_size >= self.max_queue_size * self.rejection_threshold:
            self.total_rejected += 1
            logger.warning(
                f"Rejecting request due to queue depth: {current_size}/{self.max_queue_size}"
            )
            raise HTTPException(
                status_code=503,
                detail=f"Server overloaded. Queue depth: {current_size}. Please retry later."
            )
        
        # Check if we're at absolute capacity
        if self.queue.full():
            self.total_rejected += 1
            raise HTTPException(
                status_code=503,
                detail="Server at maximum capacity. Please retry later."
            )
        
        # Create queued request
        queued_request = QueuedRequest(
            request=request,
            priority=priority
        )
        
        # Add to queue
        try:
            await self.queue.put((priority.value, queued_request))
            self.total_queued += 1
            
            logger.debug(
                f"Request {queued_request.id} queued. "
                f"Queue depth: {self.queue.qsize()}, Priority: {priority.name}"
            )
            
            return queued_request.future
            
        except asyncio.QueueFull:
            self.total_rejected += 1
            raise HTTPException(
                status_code=503,
                detail="Failed to queue request. Server overloaded."
            )
    
    async def _process_queue(self):
        """Process requests from the queue."""
        logger.info("Queue processor started")
        
        while self.running:
            try:
                # Wait for available processing slot
                async with self.active_lock:
                    if self.active_requests >= self.max_concurrent_requests:
                        # Wait a bit before checking again
                        await asyncio.sleep(0.1)
                        continue
                
                # Get next request with timeout
                try:
                    _, queued_request = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=1.0  # Check running state every second
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Check if request has timed out in queue
                queue_time = time.time() - queued_request.timestamp
                if queue_time > self.queue_timeout:
                    self.total_timeout += 1
                    queued_request.future.set_exception(
                        HTTPException(
                            status_code=504,
                            detail=f"Request timed out in queue after {queue_time:.1f}s"
                        )
                    )
                    continue
                
                # Mark request as active
                async with self.active_lock:
                    self.active_requests += 1
                
                # Update metrics
                self.total_queue_time += queue_time
                
                # Notify that request is ready for processing
                # The actual processing happens elsewhere
                queued_request.future.set_result({
                    "request": queued_request.request,
                    "queue_time": queue_time,
                    "request_id": queued_request.id
                })
                
                logger.debug(
                    f"Request {queued_request.id} ready for processing. "
                    f"Queue time: {queue_time:.2f}s"
                )
                
            except Exception as e:
                logger.error(f"Queue processor error: {e}", exc_info=True)
    
    async def complete_request(self):
        """Mark a request as completed."""
        async with self.active_lock:
            self.active_requests = max(0, self.active_requests - 1)
            self.total_processed += 1
        
        logger.debug(f"Request completed. Active: {self.active_requests}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get queue statistics.
        
        Returns:
            Dictionary containing queue metrics
        """
        rejection_rate = (
            self.total_rejected / (self.total_queued + self.total_rejected) * 100
            if (self.total_queued + self.total_rejected) > 0 else 0
        )
        
        avg_queue_time = (
            self.total_queue_time / self.total_processed
            if self.total_processed > 0 else 0
        )
        
        capacity_percent = (
            (self.queue.qsize() + self.active_requests) / 
            (self.max_queue_size + self.max_concurrent_requests) * 100
        )
        
        return {
            "queue_depth": self.queue.qsize(),
            "active_requests": self.active_requests,
            "max_queue_size": self.max_queue_size,
            "max_concurrent": self.max_concurrent_requests,
            "total_queued": self.total_queued,
            "total_processed": self.total_processed,
            "total_rejected": self.total_rejected,
            "total_timeout": self.total_timeout,
            "rejection_rate": round(rejection_rate, 2),
            "average_queue_time": round(avg_queue_time, 2),
            "capacity_percent": round(capacity_percent, 2),
            "status": self._get_status()
        }
    
    def _get_status(self) -> str:
        """Get queue status description."""
        capacity = self.queue.qsize() / self.max_queue_size
        
        if capacity >= self.rejection_threshold:
            return "critical"
        elif capacity >= 0.6:
            return "high"
        elif capacity >= 0.3:
            return "moderate"
        else:
            return "healthy"


# Singleton instance management
_queue_manager: Optional[RequestQueueManager] = None


def get_request_queue_manager() -> Optional[RequestQueueManager]:
    """Get the singleton RequestQueueManager instance."""
    return _queue_manager


def initialize_request_queue_manager(
    max_queue_size: int = 50,
    max_concurrent_requests: int = 3,
    queue_timeout: float = 300.0,
    rejection_threshold: float = 0.8
) -> RequestQueueManager:
    """
    Initialize the singleton RequestQueueManager instance.
    
    Args:
        max_queue_size: Maximum queue size
        max_concurrent_requests: Maximum concurrent requests
        queue_timeout: Queue timeout in seconds
        rejection_threshold: Rejection threshold (0.0-1.0)
        
    Returns:
        The initialized queue manager instance
    """
    global _queue_manager
    
    if _queue_manager is not None:
        logger.warning("RequestQueueManager already initialized")
        return _queue_manager
    
    _queue_manager = RequestQueueManager(
        max_queue_size=max_queue_size,
        max_concurrent_requests=max_concurrent_requests,
        queue_timeout=queue_timeout,
        rejection_threshold=rejection_threshold
    )
    
    return _queue_manager