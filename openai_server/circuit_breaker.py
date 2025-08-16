#!/usr/bin/env python3
"""
Circuit Breaker Pattern for OpenAI Server

Prevents cascading failures by automatically stopping requests to
failing services and allowing them time to recover.
"""

import asyncio
import time
import threading
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from functools import wraps


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation - requests flow through
    OPEN = "open"          # Circuit is open - requests are blocked
    HALF_OPEN = "half_open"  # Testing mode - limited requests allowed


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5          # Number of failures before opening
    recovery_timeout: float = 30.0      # Seconds to wait before trying half-open
    success_threshold: int = 3          # Successes needed in half-open to close
    timeout: float = 60.0              # Request timeout in seconds
    expected_exception_types: tuple = (Exception,)  # Exception types that count as failures
    
    # Health-based dynamic adjustments
    health_score_threshold: float = 0.3  # Below this, be more sensitive
    unhealthy_failure_threshold: int = 2  # Lower threshold when unhealthy
    healthy_recovery_timeout: float = 10.0  # Faster recovery when healthy


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    state_changes: List[Tuple[datetime, CircuitState]] = field(default_factory=list)
    avg_response_time: float = 0.0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None


class CircuitBreaker:
    """Circuit breaker with health-aware adaptive behavior"""
    
    def __init__(self, name: str = "default", config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.last_state_change = datetime.now()
        
        self.metrics = CircuitBreakerMetrics()
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        
        # Health monitor integration
        self._health_monitor = None
        
        self.logger = logging.getLogger(f"CircuitBreaker.{name}")
        self.logger.setLevel(logging.INFO)
    
    def _get_health_score(self) -> float:
        """Get current health score from health monitor"""
        try:
            if self._health_monitor is None:
                from openai_server.health_monitor import get_health_monitor
                self._health_monitor = get_health_monitor()
            return self._health_monitor.get_current_health_score()
        except Exception:
            return 1.0  # Assume healthy if can't get score
    
    def _get_dynamic_thresholds(self) -> Tuple[int, float]:
        """Get dynamic thresholds based on health score"""
        health_score = self._get_health_score()
        
        if health_score < self.config.health_score_threshold:
            # More sensitive when unhealthy
            failure_threshold = self.config.unhealthy_failure_threshold
            recovery_timeout = self.config.recovery_timeout * 1.5  # Longer recovery
        elif health_score > 0.8:
            # Less sensitive when healthy
            failure_threshold = self.config.failure_threshold * 2
            recovery_timeout = self.config.healthy_recovery_timeout
        else:
            # Normal thresholds
            failure_threshold = self.config.failure_threshold
            recovery_timeout = self.config.recovery_timeout
        
        return failure_threshold, recovery_timeout
    
    def _should_allow_request(self) -> Tuple[bool, str]:
        """Determine if request should be allowed through"""
        with self._lock:
            current_time = time.time()
            failure_threshold, recovery_timeout = self._get_dynamic_thresholds()
            
            if self.state == CircuitState.CLOSED:
                return True, "Circuit closed - normal operation"
            
            elif self.state == CircuitState.OPEN:
                # Check if enough time has passed to try half-open
                if current_time - self.last_failure_time >= recovery_timeout:
                    self._change_state(CircuitState.HALF_OPEN)
                    self.success_count = 0  # Reset success counter
                    return True, "Circuit half-open - testing recovery"
                else:
                    remaining = recovery_timeout - (current_time - self.last_failure_time)
                    return False, f"Circuit open - retry in {remaining:.1f}s"
            
            elif self.state == CircuitState.HALF_OPEN:
                # Allow limited requests to test recovery
                return True, "Circuit half-open - testing recovery"
            
            return False, "Unknown circuit state"
    
    def _record_success(self, response_time: float):
        """Record a successful request"""
        with self._lock:
            self.metrics.total_requests += 1
            self.metrics.successful_requests += 1
            self.metrics.last_success_time = datetime.now()
            
            # Update average response time (exponential moving average)
            alpha = 0.1  # Smoothing factor
            self.metrics.avg_response_time = (
                alpha * response_time + 
                (1 - alpha) * self.metrics.avg_response_time
            )
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                _, recovery_timeout = self._get_dynamic_thresholds()
                
                # Check if we have enough successes to close circuit
                if self.success_count >= self.config.success_threshold:
                    self._change_state(CircuitState.CLOSED)
                    self.failure_count = 0
                    self.logger.info(f"Circuit {self.name} closed after {self.success_count} successes")
    
    def _record_failure(self, error: Exception):
        """Record a failed request"""
        with self._lock:
            self.metrics.total_requests += 1
            self.metrics.failed_requests += 1
            self.metrics.last_failure_time = datetime.now()
            
            # Only count expected exception types as failures
            if isinstance(error, self.config.expected_exception_types):
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                failure_threshold, _ = self._get_dynamic_thresholds()
                
                if self.state == CircuitState.CLOSED:
                    if self.failure_count >= failure_threshold:
                        self._change_state(CircuitState.OPEN)
                        self.logger.warning(
                            f"Circuit {self.name} opened after {self.failure_count} failures. "
                            f"Health score: {self._get_health_score():.2f}"
                        )
                
                elif self.state == CircuitState.HALF_OPEN:
                    # Go back to open state on any failure during half-open
                    self._change_state(CircuitState.OPEN)
                    self.logger.warning(f"Circuit {self.name} returned to open state after failure during recovery")
    
    def _record_rejection(self):
        """Record a rejected request"""
        with self._lock:
            self.metrics.total_requests += 1
            self.metrics.rejected_requests += 1
    
    def _change_state(self, new_state: CircuitState):
        """Change circuit breaker state"""
        old_state = self.state
        self.state = new_state
        self.last_state_change = datetime.now()
        self.metrics.state_changes.append((datetime.now(), new_state))
        
        self.logger.info(f"Circuit {self.name} state changed: {old_state.value} -> {new_state.value}")
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection"""
        allowed, reason = self._should_allow_request()
        
        if not allowed:
            self._record_rejection()
            from fastapi import HTTPException
            raise HTTPException(status_code=503, detail=f"Service temporarily unavailable: {reason}")
        
        start_time = time.time()
        
        try:
            # Execute the function with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            response_time = time.time() - start_time
            self._record_success(response_time)
            return result
            
        except Exception as e:
            self._record_failure(e)
            raise e
    
    def call_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Execute sync function with circuit breaker protection"""
        allowed, reason = self._should_allow_request()
        
        if not allowed:
            self._record_rejection()
            raise Exception(f"Circuit breaker open: {reason}")
        
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            response_time = time.time() - start_time
            self._record_success(response_time)
            return result
            
        except Exception as e:
            self._record_failure(e)
            raise e
    
    def get_state(self) -> str:
        """Get current circuit breaker state"""
        return self.state.value
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        with self._lock:
            total = self.metrics.total_requests
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_state_change": self.last_state_change.isoformat(),
                "health_score": self._get_health_score(),
                "metrics": {
                    "total_requests": total,
                    "successful_requests": self.metrics.successful_requests,
                    "failed_requests": self.metrics.failed_requests,
                    "rejected_requests": self.metrics.rejected_requests,
                    "success_rate": (self.metrics.successful_requests / total) if total > 0 else 0,
                    "failure_rate": (self.metrics.failed_requests / total) if total > 0 else 0,
                    "rejection_rate": (self.metrics.rejected_requests / total) if total > 0 else 0,
                    "avg_response_time": self.metrics.avg_response_time,
                    "last_failure": self.metrics.last_failure_time.isoformat() if self.metrics.last_failure_time else None,
                    "last_success": self.metrics.last_success_time.isoformat() if self.metrics.last_success_time else None
                },
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "recovery_timeout": self.config.recovery_timeout,
                    "success_threshold": self.config.success_threshold,
                    "timeout": self.config.timeout
                }
            }
    
    def reset(self):
        """Reset circuit breaker to closed state"""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = 0.0
            self.metrics = CircuitBreakerMetrics()
            self.logger.info(f"Circuit {self.name} manually reset")
    
    def force_open(self):
        """Force circuit breaker to open state"""
        with self._lock:
            self._change_state(CircuitState.OPEN)
            self.last_failure_time = time.time()
            self.logger.warning(f"Circuit {self.name} forced open")
    
    def force_close(self):
        """Force circuit breaker to closed state"""
        with self._lock:
            self._change_state(CircuitState.CLOSED)
            self.failure_count = 0
            self.logger.info(f"Circuit {self.name} forced closed")


# Global circuit breaker registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get or create a circuit breaker by name"""
    with _registry_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(name, config)
        return _circuit_breakers[name]


def circuit_breaker(name: str = "default", config: Optional[CircuitBreakerConfig] = None):
    """Decorator for circuit breaker protection"""
    def decorator(func):
        breaker = get_circuit_breaker(name, config)
        
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await breaker.call_async(func, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return breaker.call_sync(func, *args, **kwargs)
            return sync_wrapper
    
    return decorator


def get_all_circuit_breaker_metrics() -> Dict[str, Dict[str, Any]]:
    """Get metrics for all circuit breakers"""
    with _registry_lock:
        return {name: cb.get_metrics() for name, cb in _circuit_breakers.items()}


def reset_all_circuit_breakers():
    """Reset all circuit breakers"""
    with _registry_lock:
        for cb in _circuit_breakers.values():
            cb.reset()


def main():
    """Test the circuit breaker"""
    import random
    
    print("🔧 Testing Circuit Breaker...")
    
    # Create a test function that fails sometimes
    failure_rate = 0.7  # 70% failure rate initially
    
    @circuit_breaker("test", CircuitBreakerConfig(failure_threshold=3, recovery_timeout=5))
    def test_function(request_id: int) -> str:
        if random.random() < failure_rate:
            raise Exception(f"Simulated failure for request {request_id}")
        return f"Success for request {request_id}"
    
    # Test the circuit breaker behavior
    breaker = get_circuit_breaker("test")
    
    for i in range(20):
        try:
            result = test_function(i)
            print(f"Request {i:2d}: {result}")
        except Exception as e:
            print(f"Request {i:2d}: FAILED - {e}")
        
        # Check circuit breaker state
        state = breaker.get_state()
        metrics = breaker.get_metrics()
        print(f"          State: {state}, Failures: {breaker.failure_count}, "
              f"Success Rate: {metrics['metrics']['success_rate']:.1%}")
        
        # Reduce failure rate after request 10 to test recovery
        if i == 10:
            failure_rate = 0.2  # 20% failure rate
            print("          --- Reducing failure rate for recovery test ---")
        
        time.sleep(1)
    
    # Print final metrics
    print("\n📊 Final Circuit Breaker Metrics:")
    final_metrics = breaker.get_metrics()
    print(f"State: {final_metrics['state']}")
    print(f"Total Requests: {final_metrics['metrics']['total_requests']}")
    print(f"Success Rate: {final_metrics['metrics']['success_rate']:.1%}")
    print(f"Rejection Rate: {final_metrics['metrics']['rejection_rate']:.1%}")


if __name__ == "__main__":
    main()