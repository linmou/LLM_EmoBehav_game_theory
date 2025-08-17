#!/usr/bin/env python3
"""
Adaptive Request Processor for OpenAI Server

Dynamically optimizes request parameters based on server health
to maintain performance under varying load conditions.
"""

import json
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from fastapi import HTTPException

# Import the request/response models from server
try:
    from server import ChatCompletionRequest, ChatMessage
except ImportError:
    # Fallback for testing
    from pydantic import BaseModel
    from typing import List, Optional
    
    class ChatMessage(BaseModel):
        role: str = "user"
        content: str = ""
        tool_calls: Optional[List] = None
        
    class ChatCompletionRequest(BaseModel):
        model: str
        messages: List[ChatMessage]
        max_tokens: Optional[int] = 1000
        temperature: Optional[float] = 1.0
        top_p: Optional[float] = 1.0
        n: Optional[int] = 1
        stream: Optional[bool] = False
        stop: Optional[List[str]] = None


@dataclass
class OptimizationStrategy:
    """Strategy for request optimization based on health level"""
    name: str
    health_threshold: float  # Apply when health is below this
    max_tokens_multiplier: float = 1.0
    temperature_multiplier: float = 1.0
    context_retention_ratio: float = 1.0
    batch_size_multiplier: float = 1.0
    enable_context_compression: bool = False
    enable_response_streaming: bool = False
    priority_boost: float = 0.0
    reject_complex_requests: bool = False


class AdaptiveRequestProcessor:
    """Processes requests with health-aware optimizations"""
    
    def __init__(self):
        self.logger = logging.getLogger("AdaptiveProcessor")
        self.logger.setLevel(logging.INFO)
        
        # Health monitor integration
        self._health_monitor = None
        
        # Queue manager integration
        self._queue_manager = None
        
        # vLLM wrapper integration
        self._vllm_wrapper = None
        
        # Request processing statistics
        self.processed_requests = 0
        self.optimized_requests = 0
        self.rejected_requests = 0
        
        # Define optimization strategies
        self.strategies = {
            "healthy": OptimizationStrategy(
                name="healthy",
                health_threshold=0.8,
                max_tokens_multiplier=1.0,
                temperature_multiplier=1.0,
                context_retention_ratio=1.0,
                batch_size_multiplier=1.0
            ),
            "light_optimization": OptimizationStrategy(
                name="light_optimization", 
                health_threshold=0.6,
                max_tokens_multiplier=0.8,
                temperature_multiplier=0.9,
                context_retention_ratio=0.9,
                batch_size_multiplier=0.9,
                enable_response_streaming=True
            ),
            "moderate_optimization": OptimizationStrategy(
                name="moderate_optimization",
                health_threshold=0.4,
                max_tokens_multiplier=0.6,
                temperature_multiplier=0.7,
                context_retention_ratio=0.7,
                batch_size_multiplier=0.7,
                enable_context_compression=True,
                enable_response_streaming=True,
                priority_boost=0.1
            ),
            "aggressive_optimization": OptimizationStrategy(
                name="aggressive_optimization",
                health_threshold=0.2,
                max_tokens_multiplier=0.4,
                temperature_multiplier=0.5,
                context_retention_ratio=0.5,
                batch_size_multiplier=0.5,
                enable_context_compression=True,
                enable_response_streaming=True,
                priority_boost=0.2,
                reject_complex_requests=True
            ),
            "critical": OptimizationStrategy(
                name="critical",
                health_threshold=0.0,
                max_tokens_multiplier=0.2,
                temperature_multiplier=0.3,
                context_retention_ratio=0.3,
                batch_size_multiplier=0.3,
                enable_context_compression=True,
                enable_response_streaming=True,
                priority_boost=0.3,
                reject_complex_requests=True
            )
        }
    
    def _get_health_monitor(self):
        """Get health monitor instance"""
        if self._health_monitor is None:
            try:
                from openai_server.health_monitor import get_health_monitor
                self._health_monitor = get_health_monitor()
            except ImportError:
                self.logger.warning("Health monitor not available, using fallback")
                return None
        return self._health_monitor
    
    def _get_queue_manager(self):
        """Get queue manager instance"""
        if self._queue_manager is None:
            try:
                from openai_server.request_queue_manager import get_request_queue_manager
                self._queue_manager = get_request_queue_manager()
            except ImportError:
                self.logger.warning("Queue manager not available")
                return None
        return self._queue_manager
    
    def _get_vllm_wrapper(self):
        """Get vLLM wrapper instance"""
        if self._vllm_wrapper is None:
            try:
                from openai_server.async_vllm_wrapper import get_async_vllm_wrapper
                self._vllm_wrapper = get_async_vllm_wrapper()
            except ImportError:
                self.logger.warning("vLLM wrapper not available")
                return None
        return self._vllm_wrapper
    
    def _get_current_health_score(self) -> float:
        """Get current health score"""
        health_monitor = self._get_health_monitor()
        if health_monitor:
            return health_monitor.get_current_health_score()
        return 1.0  # Assume healthy if monitoring unavailable
    
    def _select_optimization_strategy(self, health_score: float) -> OptimizationStrategy:
        """Select appropriate optimization strategy based on health"""
        if health_score >= 0.8:
            return self.strategies["healthy"]
        elif health_score >= 0.6:
            return self.strategies["light_optimization"]
        elif health_score >= 0.4:
            return self.strategies["moderate_optimization"]
        elif health_score >= 0.2:
            return self.strategies["aggressive_optimization"]
        else:
            return self.strategies["critical"]
    
    def _calculate_request_complexity(self, request: ChatCompletionRequest) -> float:
        """Calculate complexity score for a request (0.0 = simple, 1.0 = complex)"""
        complexity_score = 0.0
        
        # Message content length
        total_content_length = sum(len(str(msg.content) or "") for msg in request.messages)
        content_complexity = min(1.0, total_content_length / 50000)  # 50k chars = max complexity
        complexity_score += content_complexity * 0.4
        
        # Number of messages
        message_complexity = min(1.0, len(request.messages) / 50)  # 50 messages = max complexity
        complexity_score += message_complexity * 0.2
        
        # Max tokens requested
        max_tokens = request.max_tokens or 1000
        token_complexity = min(1.0, max_tokens / 4000)  # 4k tokens = max complexity
        complexity_score += token_complexity * 0.2
        
        # Tool usage
        has_tools = any(hasattr(msg, 'tool_calls') and msg.tool_calls for msg in request.messages)
        if has_tools or (hasattr(request, 'tools') and request.tools):
            complexity_score += 0.2
        
        return min(1.0, complexity_score)
    
    def _should_reject_request(self, request: ChatCompletionRequest, 
                             strategy: OptimizationStrategy) -> Tuple[bool, str]:
        """Determine if request should be rejected based on multiple factors"""
        
        # Check queue state first (most important)
        queue_manager = self._get_queue_manager()
        if queue_manager:
            queue_stats = queue_manager.get_statistics()
            
            # Critical: Reject if queue is in critical state
            if queue_stats["status"] == "critical":
                return True, f"Server queue critical (capacity: {queue_stats['capacity_percent']:.1f}%)"
            
            # High load: Reject complex requests
            if queue_stats["status"] == "high" and self._calculate_request_complexity(request) > 0.5:
                return True, f"Queue overloaded ({queue_stats['queue_depth']} requests waiting)"
        
        # Check vLLM wrapper state
        vllm_wrapper = self._get_vllm_wrapper()
        if vllm_wrapper:
            wrapper_stats = vllm_wrapper.get_statistics()
            
            # Reject if too many abandoned threads
            if wrapper_stats.get("thread_capacity_used", 0) > 80:
                return True, f"Server at thread capacity ({wrapper_stats['abandoned_threads']} abandoned)"
            
            # Reject if high timeout rate
            if wrapper_stats.get("timeout_rate", 0) > 50:
                return True, f"High timeout rate ({wrapper_stats['timeout_rate']:.1f}%)"
        
        # Original strategy-based rejection
        if not strategy.reject_complex_requests:
            return False, ""
        
        complexity = self._calculate_request_complexity(request)
        
        # Reject high complexity requests under stress
        if complexity > 0.7:
            health_score = self._get_current_health_score()
            return True, f"Request too complex for current server health ({health_score:.2f})"
        
        # Check for specific rejection criteria
        total_content = sum(len(str(msg.content) or "") for msg in request.messages)
        if total_content > 100000:  # 100k character limit under stress
            return True, "Request content too long for current server load"
        
        if (request.max_tokens or 0) > 2000:  # 2k token limit under stress
            return True, "Too many tokens requested for current server load"
        
        return False, ""
    
    def _optimize_max_tokens(self, original_tokens: Optional[int], 
                           strategy: OptimizationStrategy) -> int:
        """Optimize max_tokens parameter"""
        if original_tokens is None:
            original_tokens = 1000  # Default
        
        optimized = int(original_tokens * strategy.max_tokens_multiplier)
        
        # Ensure minimum viable response
        min_tokens = 10
        return max(min_tokens, optimized)
    
    def _optimize_temperature(self, original_temp: Optional[float], 
                            strategy: OptimizationStrategy) -> float:
        """Optimize temperature parameter"""
        if original_temp is None:
            original_temp = 1.0  # Default
        
        optimized = original_temp * strategy.temperature_multiplier
        
        # Keep within valid range
        return max(0.0, min(2.0, optimized))
    
    def _compress_context(self, messages: List[ChatMessage], 
                         strategy: OptimizationStrategy) -> List[ChatMessage]:
        """Compress context by removing or truncating messages"""
        if not strategy.enable_context_compression:
            return messages
        
        if len(messages) <= 2:  # Keep at least system + user message
            return messages
        
        # Calculate target message count
        target_count = max(2, int(len(messages) * strategy.context_retention_ratio))
        
        # Preserve message structure: system, recent user/assistant pairs
        compressed = []
        
        # Always keep system message if present
        if messages and messages[0].role == "system":
            compressed.append(messages[0])
            remaining = messages[1:]
        else:
            remaining = messages
        
        # Keep most recent messages up to target count
        if len(remaining) > (target_count - len(compressed)):
            keep_count = target_count - len(compressed)
            remaining = remaining[-keep_count:]
        
        compressed.extend(remaining)
        
        # If still too long, truncate message content
        if strategy.context_retention_ratio < 0.5:
            compressed = self._truncate_message_content(compressed, 0.7)
        
        self.logger.debug(f"Context compressed: {len(messages)} -> {len(compressed)} messages")
        return compressed
    
    def _truncate_message_content(self, messages: List[ChatMessage], 
                                retention_ratio: float) -> List[ChatMessage]:
        """Truncate content within messages"""
        truncated = []
        
        for msg in messages:
            if msg.role == "system":
                # Never truncate system messages
                truncated.append(msg)
            else:
                # Truncate user/assistant content
                content = str(msg.content or "")
                if len(content) > 1000:  # Only truncate long messages
                    target_length = int(len(content) * retention_ratio)
                    truncated_content = content[:target_length] + "... [truncated]"
                    
                    # Create new message with truncated content
                    new_msg = ChatMessage(
                        role=msg.role,
                        content=truncated_content
                    )
                    # Copy other attributes if they exist
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        new_msg.tool_calls = msg.tool_calls
                    
                    truncated.append(new_msg)
                else:
                    truncated.append(msg)
        
        return truncated
    
    def _calculate_processing_priority(self, request: ChatCompletionRequest, 
                                     strategy: OptimizationStrategy) -> float:
        """Calculate processing priority for request"""
        base_priority = 1.0
        
        # Boost priority based on strategy
        priority = base_priority + strategy.priority_boost
        
        # Lower priority for complex requests under stress
        complexity = self._calculate_request_complexity(request)
        if strategy.reject_complex_requests and complexity > 0.5:
            priority *= (1.0 - complexity * 0.5)
        
        # Higher priority for shorter requests under stress  
        if strategy.enable_context_compression:
            total_length = sum(len(str(msg.content) or "") for msg in request.messages)
            if total_length < 1000:  # Short requests get priority
                priority *= 1.2
        
        return priority
    
    def process_request(self, request: ChatCompletionRequest) -> Tuple[ChatCompletionRequest, Dict[str, Any]]:
        """Process and optimize request based on current health"""
        self.processed_requests += 1
        start_time = time.time()
        
        # Get current health and select strategy
        health_score = self._get_current_health_score()
        strategy = self._select_optimization_strategy(health_score)
        
        self.logger.debug(f"Processing request with strategy: {strategy.name} (health: {health_score:.2f})")
        
        # Check if request should be rejected
        should_reject, reject_reason = self._should_reject_request(request, strategy)
        if should_reject:
            self.rejected_requests += 1
            raise HTTPException(
                status_code=503, 
                detail=f"Request rejected due to server load: {reject_reason}"
            )
        
        # Create optimized request
        optimized_request = ChatCompletionRequest(
            model=request.model,
            messages=self._compress_context(request.messages, strategy),
            max_tokens=self._optimize_max_tokens(request.max_tokens, strategy),
            temperature=self._optimize_temperature(request.temperature, strategy),
            top_p=getattr(request, 'top_p', 1.0),  # Keep top_p unchanged for now
            n=getattr(request, 'n', 1),
            stream=getattr(request, 'stream', False) or strategy.enable_response_streaming,
            stop=getattr(request, 'stop', None)
        )
        
        # Copy additional fields if they exist
        if hasattr(request, 'tools'):
            optimized_request.tools = request.tools
        if hasattr(request, 'tool_choice'):
            optimized_request.tool_choice = request.tool_choice
        
        # Track if optimization was applied
        optimization_applied = (
            len(optimized_request.messages) != len(request.messages) or
            optimized_request.max_tokens != request.max_tokens or
            abs((optimized_request.temperature or 1.0) - (request.temperature or 1.0)) > 0.01 or
            getattr(optimized_request, 'stream', False) != getattr(request, 'stream', False)
        )
        
        if optimization_applied:
            self.optimized_requests += 1
        
        # Create processing metadata
        processing_info = {
            "health_score": health_score,
            "strategy_used": strategy.name,
            "optimization_applied": optimization_applied,
            "request_complexity": self._calculate_request_complexity(request),
            "processing_priority": self._calculate_processing_priority(optimized_request, strategy),
            "original_message_count": len(request.messages),
            "optimized_message_count": len(optimized_request.messages),
            "original_max_tokens": request.max_tokens,
            "optimized_max_tokens": optimized_request.max_tokens,
            "original_temperature": request.temperature,
            "optimized_temperature": optimized_request.temperature,
            "processing_time_ms": (time.time() - start_time) * 1000,
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Request processed: {strategy.name} optimization, "
                        f"complexity={processing_info['request_complexity']:.2f}, "
                        f"priority={processing_info['processing_priority']:.2f}")
        
        return optimized_request, processing_info
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        total = self.processed_requests
        stats = {
            "total_requests": total,
            "optimized_requests": self.optimized_requests,
            "rejected_requests": self.rejected_requests,
            "optimization_rate": (self.optimized_requests / total) if total > 0 else 0,
            "rejection_rate": (self.rejected_requests / total) if total > 0 else 0,
            "current_health_score": self._get_current_health_score(),
            "current_strategy": self._select_optimization_strategy(self._get_current_health_score()).name
        }
        
        # Add queue manager stats if available
        queue_manager = self._get_queue_manager()
        if queue_manager:
            stats["queue_status"] = queue_manager.get_statistics()
        
        # Add vLLM wrapper stats if available
        vllm_wrapper = self._get_vllm_wrapper()
        if vllm_wrapper:
            stats["vllm_status"] = vllm_wrapper.get_statistics()
        
        return stats
    
    def reset_statistics(self):
        """Reset processing statistics"""
        self.processed_requests = 0
        self.optimized_requests = 0
        self.rejected_requests = 0


# Global adaptive processor instance
_adaptive_processor = None

def get_adaptive_processor() -> AdaptiveRequestProcessor:
    """Get global adaptive processor instance"""
    global _adaptive_processor
    if _adaptive_processor is None:
        _adaptive_processor = AdaptiveRequestProcessor()
    return _adaptive_processor


def main():
    """Test the adaptive processor"""
    print("🔧 Testing Adaptive Request Processor...")
    
    processor = AdaptiveRequestProcessor()
    
    # Create test requests with different complexities
    test_requests = [
        # Simple request
        ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello, how are you?")],
            max_tokens=100,
            temperature=0.7
        ),
        # Medium complexity request  
        ChatCompletionRequest(
            model="test-model",
            messages=[
                ChatMessage(role="system", content="You are a helpful assistant."),
                ChatMessage(role="user", content="Explain quantum computing in detail. " * 100),
                ChatMessage(role="assistant", content="Quantum computing is..."),
                ChatMessage(role="user", content="Tell me more about quantum entanglement.")
            ],
            max_tokens=2000,
            temperature=1.0
        ),
        # High complexity request
        ChatCompletionRequest(
            model="test-model", 
            messages=[
                ChatMessage(role="system", content="You are an expert researcher."),
                *[ChatMessage(role="user", content=f"Question {i}: " + "Complex question " * 500) 
                  for i in range(10)]
            ],
            max_tokens=4000,
            temperature=1.5
        )
    ]
    
    # Test with different simulated health scores
    health_scores = [1.0, 0.7, 0.5, 0.3, 0.1]
    
    for health_score in health_scores:
        print(f"\n--- Testing with simulated health score: {health_score:.1f} ---")
        
        # Simulate different health by temporarily modifying the health monitor
        original_get_health = processor._get_current_health_score
        processor._get_current_health_score = lambda: health_score
        
        for i, request in enumerate(test_requests):
            try:
                optimized_request, processing_info = processor.process_request(request)
                
                print(f"Request {i+1} ({processing_info['request_complexity']:.2f} complexity):")
                print(f"  Strategy: {processing_info['strategy_used']}")
                print(f"  Messages: {processing_info['original_message_count']} -> {processing_info['optimized_message_count']}")
                print(f"  Max tokens: {processing_info['original_max_tokens']} -> {processing_info['optimized_max_tokens']}")
                print(f"  Temperature: {processing_info['original_temperature']:.2f} -> {processing_info['optimized_temperature']:.2f}")
                print(f"  Optimized: {processing_info['optimization_applied']}")
                
            except HTTPException as e:
                print(f"Request {i+1}: REJECTED - {e.detail}")
        
        # Restore original method
        processor._get_current_health_score = original_get_health
    
    # Print final statistics
    print(f"\n📊 Processing Statistics:")
    stats = processor.get_statistics()
    print(f"Total requests: {stats['total_requests']}")
    print(f"Optimization rate: {stats['optimization_rate']:.1%}")
    print(f"Rejection rate: {stats['rejection_rate']:.1%}")


if __name__ == "__main__":  
    main()