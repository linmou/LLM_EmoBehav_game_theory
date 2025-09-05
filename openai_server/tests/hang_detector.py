#!/usr/bin/env python3
"""
Hang Detector for Server Stress Testing

Detects when the server becomes unresponsive, stuck, or exhibits
hanging behavior during stress testing.
"""

import time
import threading
import contextlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import logging


class HangType(Enum):
    """Types of hang conditions"""
    REQUEST_TIMEOUT = "request_timeout"
    RESPONSE_INCOMPLETE = "response_incomplete"
    SERVER_UNRESPONSIVE = "server_unresponsive"
    QUEUE_DEADLOCK = "queue_deadlock"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    STREAMING_HANG = "streaming_hang"


@dataclass
class HangEvent:
    """Represents a detected hang event"""
    hang_id: str
    hang_type: HangType
    start_time: datetime
    detection_time: datetime
    timeout_duration: float
    request_details: Dict[str, Any]
    context: Dict[str, Any]
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class RequestTracker:
    """Tracks individual requests for hang detection"""
    
    def __init__(self, request_id: str, timeout: float, context: Dict[str, Any]):
        self.request_id = request_id
        self.timeout = timeout
        self.context = context
        self.start_time = datetime.now()
        self.last_activity = datetime.now()
        self.completed = False
        self.timed_out = False
        
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
    
    def mark_completed(self):
        """Mark request as completed"""
        self.completed = True
        self.update_activity()
    
    def is_timed_out(self) -> bool:
        """Check if request has timed out"""
        if self.completed or self.timed_out:
            return self.timed_out
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if elapsed > self.timeout:
            self.timed_out = True
            return True
        
        return False
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since request start"""
        return (datetime.now() - self.start_time).total_seconds()


class HangDetector:
    """Detects various types of server hang conditions"""
    
    def __init__(self, default_timeout: float = 300.0):
        self.default_timeout = default_timeout
        self._monitoring = False
        self._monitor_thread = None
        self._lock = threading.Lock()
        
        # Tracking structures
        self._active_requests: Dict[str, RequestTracker] = {}
        self._detected_hangs: List[HangEvent] = []
        self._hang_patterns: Dict[str, int] = {}
        
        # Configuration
        self.max_concurrent_requests = 100
        self.queue_deadlock_threshold = 60.0  # seconds
        self.streaming_hang_threshold = 30.0  # seconds
        self.response_incomplete_threshold = 120.0  # seconds
        
        # Setup logging
        self.logger = logging.getLogger("HangDetector")
        self.logger.setLevel(logging.INFO)
    
    def start_monitoring(self):
        """Start hang detection monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        self.logger.info("Hang detection monitoring started")
    
    def stop_monitoring(self):
        """Stop hang detection monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        self.logger.info("Hang detection monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop for hang detection"""
        while self._monitoring:
            try:
                self._check_for_hangs()
                time.sleep(1)  # Check every second
            except Exception as e:
                self.logger.error(f"Hang detection error: {e}")
                time.sleep(1)
    
    def _check_for_hangs(self):
        """Check for various hang conditions"""
        with self._lock:
            current_time = datetime.now()
            
            # Check for request timeouts
            for request_id, tracker in list(self._active_requests.items()):
                if tracker.is_timed_out():
                    self._record_hang(
                        hang_type=HangType.REQUEST_TIMEOUT,
                        request_id=request_id,
                        timeout_duration=tracker.get_elapsed_time(),
                        context=tracker.context
                    )
                    # Remove timed out request
                    del self._active_requests[request_id]
            
            # Check for queue deadlock (too many concurrent requests)
            if len(self._active_requests) > self.max_concurrent_requests:
                oldest_request = min(self._active_requests.values(), key=lambda r: r.start_time)
                if oldest_request.get_elapsed_time() > self.queue_deadlock_threshold:
                    self._record_hang(
                        hang_type=HangType.QUEUE_DEADLOCK,
                        request_id=oldest_request.request_id,
                        timeout_duration=oldest_request.get_elapsed_time(),
                        context={
                            "concurrent_requests": len(self._active_requests),
                            "oldest_request_elapsed": oldest_request.get_elapsed_time()
                        }
                    )
            
            # Check for streaming hangs
            for request_id, tracker in self._active_requests.items():
                if tracker.context.get("streaming", False):
                    time_since_activity = (current_time - tracker.last_activity).total_seconds()
                    if time_since_activity > self.streaming_hang_threshold:
                        self._record_hang(
                            hang_type=HangType.STREAMING_HANG,
                            request_id=request_id,
                            timeout_duration=time_since_activity,
                            context={
                                "time_since_last_chunk": time_since_activity,
                                "total_elapsed": tracker.get_elapsed_time()
                            }
                        )
    
    def _record_hang(self, hang_type: HangType, request_id: str, timeout_duration: float, context: Dict[str, Any]):
        """Record a detected hang event"""
        hang_id = f"{hang_type.value}_{request_id}_{int(time.time())}"
        
        # Check if we already recorded this hang
        existing_hang = any(hang.hang_id.startswith(f"{hang_type.value}_{request_id}") 
                           and not hang.resolved 
                           for hang in self._detected_hangs)
        
        if existing_hang:
            return  # Don't duplicate hang records
        
        tracker = self._active_requests.get(request_id)
        request_details = {
            "request_id": request_id,
            "start_time": tracker.start_time.isoformat() if tracker else None,
            "elapsed_time": timeout_duration,
            "request_context": tracker.context if tracker else {}
        }
        
        hang_event = HangEvent(
            hang_id=hang_id,
            hang_type=hang_type,
            start_time=tracker.start_time if tracker else datetime.now(),
            detection_time=datetime.now(),
            timeout_duration=timeout_duration,
            request_details=request_details,
            context=context
        )
        
        self._detected_hangs.append(hang_event)
        
        # Update hang patterns
        pattern_key = f"{hang_type.value}_{context.get('content_type', 'unknown')}"
        self._hang_patterns[pattern_key] = self._hang_patterns.get(pattern_key, 0) + 1
        
        self.logger.warning(f"Hang detected: {hang_type.value} for request {request_id} "
                           f"(duration: {timeout_duration:.1f}s)")
    
    @contextlib.contextmanager
    def monitor_request(self, request_id: str, timeout: Optional[float] = None, **context):
        """Context manager to monitor a specific request for hangs"""
        if timeout is None:
            timeout = self.default_timeout
        
        tracker = RequestTracker(request_id, timeout, context)
        
        with self._lock:
            self._active_requests[request_id] = tracker
        
        try:
            yield tracker
        except Exception as e:
            # Record exception context
            context["exception"] = str(e)
            context["exception_type"] = type(e).__name__
            
            # Check if this looks like a hang-related exception
            if any(keyword in str(e).lower() for keyword in ["timeout", "connection", "hang", "stuck"]):
                self._record_hang(
                    hang_type=HangType.SERVER_UNRESPONSIVE,
                    request_id=request_id,
                    timeout_duration=tracker.get_elapsed_time(),
                    context=context
                )
            
            raise
        finally:
            with self._lock:
                if request_id in self._active_requests:
                    self._active_requests[request_id].mark_completed()
                    # Keep completed requests for a short time for analysis
                    threading.Timer(30.0, lambda: self._cleanup_request(request_id)).start()
    
    def _cleanup_request(self, request_id: str):
        """Clean up completed request after delay"""
        with self._lock:
            self._active_requests.pop(request_id, None)
    
    def update_request_activity(self, request_id: str):
        """Update activity for a streaming or long-running request"""
        with self._lock:
            if request_id in self._active_requests:
                self._active_requests[request_id].update_activity()
    
    def mark_request_completed(self, request_id: str):
        """Mark a request as completed successfully"""
        with self._lock:
            if request_id in self._active_requests:
                self._active_requests[request_id].mark_completed()
                
                # Check if this resolves any hang
                for hang in self._detected_hangs:
                    if (hang.request_details.get("request_id") == request_id and 
                        not hang.resolved):
                        hang.resolved = True
                        hang.resolution_time = datetime.now()
                        self.logger.info(f"Hang resolved: {hang.hang_id}")
    
    def detect_server_unresponsive(self, health_check_failures: int, 
                                 consecutive_failures: bool = True) -> bool:
        """Detect if server appears completely unresponsive"""
        if health_check_failures >= 3 and consecutive_failures:
            self._record_hang(
                hang_type=HangType.SERVER_UNRESPONSIVE,
                request_id="health_check",
                timeout_duration=0,
                context={
                    "health_check_failures": health_check_failures,
                    "consecutive_failures": consecutive_failures
                }
            )
            return True
        
        return False
    
    def detect_resource_exhaustion_hang(self, gpu_memory_usage: float, 
                                      cpu_usage: float, memory_usage: float) -> bool:
        """Detect hangs due to resource exhaustion"""
        if (gpu_memory_usage > 98.0 or cpu_usage > 95.0 or memory_usage > 95.0):
            # Check if we have long-running requests during high resource usage
            with self._lock:
                long_running_requests = [
                    tracker for tracker in self._active_requests.values()
                    if tracker.get_elapsed_time() > 60.0 and not tracker.completed
                ]
            
            if long_running_requests:
                for tracker in long_running_requests:
                    self._record_hang(
                        hang_type=HangType.RESOURCE_EXHAUSTION,
                        request_id=tracker.request_id,
                        timeout_duration=tracker.get_elapsed_time(),
                        context={
                            "gpu_memory_usage": gpu_memory_usage,
                            "cpu_usage": cpu_usage,
                            "memory_usage": memory_usage,
                            "concurrent_requests": len(self._active_requests)
                        }
                    )
                return True
        
        return False
    
    def get_detected_hangs(self, include_resolved: bool = True) -> List[Dict[str, Any]]:
        """Get all detected hang events"""
        with self._lock:
            hangs = self._detected_hangs.copy()
        
        if not include_resolved:
            hangs = [hang for hang in hangs if not hang.resolved]
        
        return [
            {
                "hang_id": hang.hang_id,
                "hang_type": hang.hang_type.value,
                "detection_time": hang.detection_time.isoformat(),
                "timeout_duration": hang.timeout_duration,
                "request_details": hang.request_details,
                "context": hang.context,
                "resolved": hang.resolved,
                "resolution_time": hang.resolution_time.isoformat() if hang.resolution_time else None
            }
            for hang in hangs
        ]
    
    def get_hang_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected hangs"""
        with self._lock:
            total_hangs = len(self._detected_hangs)
            resolved_hangs = len([hang for hang in self._detected_hangs if hang.resolved])
            active_requests = len(self._active_requests)
            
            # Hang type distribution
            hang_type_counts = {}
            for hang in self._detected_hangs:
                hang_type_counts[hang.hang_type.value] = hang_type_counts.get(hang.hang_type.value, 0) + 1
            
            # Average hang duration by type
            hang_durations = {}
            for hang in self._detected_hangs:
                hang_type = hang.hang_type.value
                if hang_type not in hang_durations:
                    hang_durations[hang_type] = []
                hang_durations[hang_type].append(hang.timeout_duration)
            
            avg_durations = {}
            for hang_type, durations in hang_durations.items():
                avg_durations[hang_type] = sum(durations) / len(durations) if durations else 0
        
        return {
            "total_hangs_detected": total_hangs,
            "resolved_hangs": resolved_hangs,
            "unresolved_hangs": total_hangs - resolved_hangs,
            "active_requests": active_requests,
            "hang_type_distribution": hang_type_counts,
            "average_hang_duration_by_type": avg_durations,
            "hang_patterns": self._hang_patterns.copy()
        }
    
    def get_active_requests(self) -> List[Dict[str, Any]]:
        """Get information about currently active requests"""
        with self._lock:
            return [
                {
                    "request_id": tracker.request_id,
                    "start_time": tracker.start_time.isoformat(),
                    "elapsed_time": tracker.get_elapsed_time(),
                    "timeout": tracker.timeout,
                    "completed": tracker.completed,
                    "timed_out": tracker.timed_out,
                    "context": tracker.context
                }
                for tracker in self._active_requests.values()
            ]
    
    def clear_resolved_hangs(self):
        """Clear resolved hang events to free memory"""
        with self._lock:
            self._detected_hangs = [hang for hang in self._detected_hangs if not hang.resolved]
        
        self.logger.info("Cleared resolved hang events")
    
    def is_server_likely_hung(self) -> Tuple[bool, str]:
        """Determine if server is likely in a hung state"""
        with self._lock:
            # Check for recent unresolved hangs
            recent_cutoff = datetime.now() - timedelta(minutes=5)
            recent_unresolved_hangs = [
                hang for hang in self._detected_hangs
                if hang.detection_time >= recent_cutoff and not hang.resolved
            ]
            
            # Multiple recent hangs indicate serious problems
            if len(recent_unresolved_hangs) >= 3:
                return True, f"Multiple recent unresolved hangs ({len(recent_unresolved_hangs)})"
            
            # Check for critical hang types
            critical_hangs = [
                hang for hang in recent_unresolved_hangs
                if hang.hang_type in [HangType.SERVER_UNRESPONSIVE, HangType.QUEUE_DEADLOCK]
            ]
            
            if critical_hangs:
                return True, f"Critical hang detected: {critical_hangs[0].hang_type.value}"
            
            # Check for too many active requests (potential queue backup)
            if len(self._active_requests) > self.max_concurrent_requests:
                return True, f"Too many active requests ({len(self._active_requests)})"
            
            # Check for very old active requests
            old_requests = [
                tracker for tracker in self._active_requests.values()
                if tracker.get_elapsed_time() > self.default_timeout * 2
            ]
            
            if len(old_requests) >= 2:
                return True, f"Multiple very old requests ({len(old_requests)})"
        
        return False, "Server appears responsive"


def main():
    """Test the hang detector"""
    import argparse
    import random
    
    parser = argparse.ArgumentParser(description="Hang Detector Test")
    parser.add_argument("--timeout", type=float, default=30.0, help="Default timeout")
    parser.add_argument("--test-duration", type=int, default=60, help="Test duration in seconds")
    
    args = parser.parse_args()
    
    detector = HangDetector(args.timeout)
    detector.start_monitoring()
    
    print(f"🕵️ Testing hang detector for {args.test_duration} seconds...")
    
    try:
        # Simulate various request patterns
        for i in range(args.test_duration):
            # Simulate normal requests
            if random.random() < 0.3:  # 30% chance
                request_id = f"normal_request_{i}"
                with detector.monitor_request(request_id, timeout=10.0, content_type="normal"):
                    time.sleep(random.uniform(0.1, 2.0))  # Normal processing time
            
            # Simulate slow requests
            if random.random() < 0.1:  # 10% chance
                request_id = f"slow_request_{i}"
                try:
                    with detector.monitor_request(request_id, timeout=5.0, content_type="slow"):
                        time.sleep(random.uniform(6.0, 8.0))  # Will timeout
                except:
                    pass
            
            # Simulate streaming requests
            if random.random() < 0.05:  # 5% chance
                request_id = f"streaming_request_{i}"
                with detector.monitor_request(request_id, timeout=20.0, content_type="streaming", streaming=True):
                    for chunk in range(5):
                        time.sleep(2.0)
                        detector.update_request_activity(request_id)
                    detector.mark_request_completed(request_id)
            
            time.sleep(1)
            
            # Print status every 10 seconds
            if i % 10 == 0:
                stats = detector.get_hang_statistics()
                active_requests = detector.get_active_requests()
                print(f"  [{i:3d}s] Active: {len(active_requests)} | "
                      f"Hangs: {stats['total_hangs_detected']} | "
                      f"Resolved: {stats['resolved_hangs']}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    
    finally:
        detector.stop_monitoring()
        
        # Print final statistics
        stats = detector.get_hang_statistics()
        hangs = detector.get_detected_hangs()
        
        print("\n🔍 Hang Detection Results:")
        print(f"  Total hangs detected: {stats['total_hangs_detected']}")
        print(f"  Resolved hangs: {stats['resolved_hangs']}")
        print(f"  Unresolved hangs: {stats['unresolved_hangs']}")
        
        if stats['hang_type_distribution']:
            print("  Hang types:")
            for hang_type, count in stats['hang_type_distribution'].items():
                print(f"    {hang_type}: {count}")
        
        # Check if server appears hung
        hung, reason = detector.is_server_likely_hung()
        print(f"  Server likely hung: {hung} ({reason})")


if __name__ == "__main__":
    main()