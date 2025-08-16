#!/usr/bin/env python3
"""
Server Monitor for Stress Testing

Monitors server health, resource usage, and performance metrics
during stress testing operations.
"""

import time
import threading
import subprocess
import requests
import psutil
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import queue
import logging


@dataclass
class ServerState:
    """Represents server state at a point in time"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    gpu_memory_usage: Dict[int, float]  # GPU index -> memory usage %
    gpu_utilization: Dict[int, float]   # GPU index -> utilization %
    network_connections: int
    response_time: Optional[float]
    server_responsive: bool
    error_count: int


class ServerMonitor:
    """Real-time server monitoring"""
    
    def __init__(self, server_url: str, monitor_interval: float = 1.0):
        self.server_url = server_url
        self.monitor_interval = monitor_interval
        self.base_url = server_url.replace("/v1", "")
        
        self._monitoring = False
        self._monitor_thread = None
        self._state_history: List[ServerState] = []
        self._state_queue = queue.Queue()
        self._lock = threading.Lock()
        
        # Performance tracking
        self._response_times: List[float] = []
        self._error_count = 0
        self._peak_gpu_memory = 0.0
        self._peak_cpu_usage = 0.0
        
        # Setup logging
        self.logger = logging.getLogger("ServerMonitor")
        self.logger.setLevel(logging.INFO)
    
    def start_monitoring(self):
        """Start monitoring server state"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        self.logger.info("Server monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring server state"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        self.logger.info("Server monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                state = self._collect_server_state()
                
                with self._lock:
                    self._state_history.append(state)
                    # Keep only last 1000 states to prevent memory bloat
                    if len(self._state_history) > 1000:
                        self._state_history = self._state_history[-1000:]
                
                # Update peak values
                if state.cpu_usage > self._peak_cpu_usage:
                    self._peak_cpu_usage = state.cpu_usage
                
                if state.gpu_memory_usage:
                    max_gpu_memory = max(state.gpu_memory_usage.values())
                    if max_gpu_memory > self._peak_gpu_memory:
                        self._peak_gpu_memory = max_gpu_memory
                
                # Track response times
                if state.response_time is not None:
                    self._response_times.append(state.response_time)
                    # Keep only last 100 response times
                    if len(self._response_times) > 100:
                        self._response_times = self._response_times[-100:]
                
                # Track errors
                if not state.server_responsive:
                    self._error_count += 1
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"Monitor loop error: {e}")
                time.sleep(self.monitor_interval)
    
    def _collect_server_state(self) -> ServerState:
        """Collect current server state"""
        timestamp = datetime.now()
        
        # CPU and memory usage
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_usage = psutil.virtual_memory().percent
        
        # GPU usage
        gpu_memory_usage, gpu_utilization = self._get_gpu_usage()
        
        # Network connections
        network_connections = len([conn for conn in psutil.net_connections() 
                                 if conn.laddr.port == 8000 and conn.status == psutil.CONN_ESTABLISHED])
        
        # Server responsiveness and response time
        server_responsive, response_time = self._test_server_responsiveness()
        
        return ServerState(
            timestamp=timestamp,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_memory_usage=gpu_memory_usage,
            gpu_utilization=gpu_utilization,
            network_connections=network_connections,
            response_time=response_time,
            server_responsive=server_responsive,
            error_count=self._error_count
        )
    
    def _get_gpu_usage(self) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Get GPU memory usage and utilization"""
        try:
            # Use nvidia-smi to get GPU stats
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            memory_usage = {}
            utilization = {}
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) == 4:
                        gpu_index = int(parts[0])
                        memory_used = float(parts[1])
                        memory_total = float(parts[2])
                        gpu_util = float(parts[3])
                        
                        memory_usage[gpu_index] = (memory_used / memory_total) * 100 if memory_total > 0 else 0
                        utilization[gpu_index] = gpu_util
            
            return memory_usage, utilization
            
        except Exception as e:
            self.logger.warning(f"Failed to get GPU usage: {e}")
            return {}, {}
    
    def _test_server_responsiveness(self) -> Tuple[bool, Optional[float]]:
        """Test if server is responsive and measure response time"""
        try:
            start_time = time.time()
            
            # Try health endpoint first
            health_response = requests.get(f"{self.base_url}/health", timeout=5)
            
            if health_response.status_code == 200:
                response_time = time.time() - start_time
                return True, response_time
            
            # If health endpoint fails, try a simple API request
            api_response = requests.post(
                f"{self.server_url}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer dummy"
                },
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "health check"}],
                    "max_tokens": 1
                },
                timeout=10
            )
            
            response_time = time.time() - start_time
            return api_response.status_code == 200, response_time
            
        except Exception as e:
            self.logger.debug(f"Server responsiveness test failed: {e}")
            return False, None
    
    def check_server_health(self) -> Dict[str, Any]:
        """Perform comprehensive server health check"""
        try:
            start_time = time.time()
            
            # Test health endpoint
            health_response = requests.get(f"{self.base_url}/health", timeout=10)
            health_check_time = time.time() - start_time
            
            # Test models endpoint
            start_time = time.time()
            models_response = requests.get(f"{self.server_url}/models", timeout=10)
            models_check_time = time.time() - start_time
            
            # Test simple completion
            start_time = time.time()
            completion_response = requests.post(
                f"{self.server_url}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer dummy"
                },
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 5
                },
                timeout=30
            )
            completion_check_time = time.time() - start_time
            
            return {
                "healthy": all([
                    health_response.status_code == 200,
                    models_response.status_code == 200,
                    completion_response.status_code == 200
                ]),
                "health_endpoint": {
                    "status_code": health_response.status_code,
                    "response_time": health_check_time
                },
                "models_endpoint": {
                    "status_code": models_response.status_code,
                    "response_time": models_check_time
                },
                "completion_endpoint": {
                    "status_code": completion_response.status_code,
                    "response_time": completion_check_time
                },
                "total_check_time": health_check_time + models_check_time + completion_check_time
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "health_endpoint": {"status_code": 0, "response_time": 0},
                "models_endpoint": {"status_code": 0, "response_time": 0},
                "completion_endpoint": {"status_code": 0, "response_time": 0},
                "total_check_time": 0
            }
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current server state"""
        with self._lock:
            if not self._state_history:
                return {"error": "No state history available"}
            
            latest_state = self._state_history[-1]
            
            return {
                "timestamp": latest_state.timestamp.isoformat(),
                "cpu_usage": latest_state.cpu_usage,
                "memory_usage": latest_state.memory_usage,
                "gpu_memory_usage": latest_state.gpu_memory_usage,
                "gpu_utilization": latest_state.gpu_utilization,
                "network_connections": latest_state.network_connections,
                "server_responsive": latest_state.server_responsive,
                "response_time": latest_state.response_time,
                "error_count": latest_state.error_count
            }
    
    def get_state_history(self, last_n_minutes: int = 5) -> List[Dict[str, Any]]:
        """Get state history for the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=last_n_minutes)
        
        with self._lock:
            recent_states = [
                state for state in self._state_history 
                if state.timestamp >= cutoff_time
            ]
            
            return [
                {
                    "timestamp": state.timestamp.isoformat(),
                    "cpu_usage": state.cpu_usage,
                    "memory_usage": state.memory_usage,
                    "gpu_memory_usage": state.gpu_memory_usage,
                    "gpu_utilization": state.gpu_utilization,
                    "network_connections": state.network_connections,
                    "server_responsive": state.server_responsive,
                    "response_time": state.response_time
                }
                for state in recent_states
            ]
    
    def get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage (average across all GPUs)"""
        memory_usage, _ = self._get_gpu_usage()
        if not memory_usage:
            return 0.0
        return sum(memory_usage.values()) / len(memory_usage)
    
    def get_peak_gpu_memory_usage(self) -> float:
        """Get peak GPU memory usage recorded"""
        return self._peak_gpu_memory
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage"""
        return psutil.cpu_percent(interval=0.1)
    
    def get_peak_cpu_usage(self) -> float:
        """Get peak CPU usage recorded"""
        return self._peak_cpu_usage
    
    def get_average_response_time(self, last_n: int = 10) -> float:
        """Get average response time for last N responses"""
        if not self._response_times:
            return 0.0
        
        recent_times = self._response_times[-last_n:] if len(self._response_times) > last_n else self._response_times
        return sum(recent_times) / len(recent_times)
    
    def get_error_rate(self, last_n_minutes: int = 5) -> float:
        """Get error rate for the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=last_n_minutes)
        
        with self._lock:
            recent_states = [
                state for state in self._state_history 
                if state.timestamp >= cutoff_time
            ]
            
            if not recent_states:
                return 0.0
            
            error_states = [state for state in recent_states if not state.server_responsive]
            return len(error_states) / len(recent_states)
    
    def detect_performance_degradation(self, threshold_multiplier: float = 2.0) -> Dict[str, Any]:
        """Detect if server performance has degraded significantly"""
        if len(self._response_times) < 10:
            return {"degradation_detected": False, "reason": "Insufficient data"}
        
        # Compare recent response times to earlier ones
        recent_times = self._response_times[-5:]
        earlier_times = self._response_times[-15:-5]
        
        if not earlier_times:
            return {"degradation_detected": False, "reason": "Insufficient historical data"}
        
        recent_avg = sum(recent_times) / len(recent_times)
        earlier_avg = sum(earlier_times) / len(earlier_times)
        
        if recent_avg > earlier_avg * threshold_multiplier:
            return {
                "degradation_detected": True,
                "recent_avg_response_time": recent_avg,
                "earlier_avg_response_time": earlier_avg,
                "degradation_factor": recent_avg / earlier_avg,
                "threshold": threshold_multiplier
            }
        
        return {"degradation_detected": False, "recent_avg": recent_avg, "earlier_avg": earlier_avg}
    
    def test_health_recovery(self) -> Dict[str, Any]:
        """Test server recovery after potential issues"""
        try:
            # Wait a bit for any pending operations to complete
            time.sleep(5)
            
            # Perform multiple health checks
            health_results = []
            for i in range(3):
                health_check = self.check_server_health()
                health_results.append(health_check["healthy"])
                if not health_check["healthy"]:
                    time.sleep(10)  # Wait longer if unhealthy
                else:
                    time.sleep(2)   # Brief wait if healthy
            
            recovery_success = any(health_results)
            full_recovery = all(health_results)
            
            return {
                "recovery_attempted": True,
                "recovery_success": recovery_success,
                "full_recovery": full_recovery,
                "health_check_results": health_results,
                "final_state": self.get_current_state()
            }
            
        except Exception as e:
            return {
                "recovery_attempted": True,
                "recovery_success": False,
                "error": str(e)
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive monitoring statistics"""
        with self._lock:
            if not self._state_history:
                return {"error": "No monitoring data available"}
            
            # Calculate statistics from state history
            cpu_values = [state.cpu_usage for state in self._state_history]
            memory_values = [state.memory_usage for state in self._state_history]
            connection_values = [state.network_connections for state in self._state_history]
            
            # Response time statistics
            response_time_values = [state.response_time for state in self._state_history if state.response_time is not None]
            
            # Availability statistics
            total_states = len(self._state_history)
            responsive_states = len([state for state in self._state_history if state.server_responsive])
            
            return {
                "monitoring_duration_minutes": (self._state_history[-1].timestamp - self._state_history[0].timestamp).total_seconds() / 60,
                "total_measurements": total_states,
                "cpu_usage": {
                    "min": min(cpu_values),
                    "max": max(cpu_values),
                    "avg": sum(cpu_values) / len(cpu_values),
                    "current": cpu_values[-1]
                },
                "memory_usage": {
                    "min": min(memory_values),
                    "max": max(memory_values),
                    "avg": sum(memory_values) / len(memory_values),
                    "current": memory_values[-1]
                },
                "network_connections": {
                    "min": min(connection_values),
                    "max": max(connection_values),
                    "avg": sum(connection_values) / len(connection_values),
                    "current": connection_values[-1]
                },
                "response_times": {
                    "count": len(response_time_values),
                    "min": min(response_time_values) if response_time_values else 0,
                    "max": max(response_time_values) if response_time_values else 0,
                    "avg": sum(response_time_values) / len(response_time_values) if response_time_values else 0
                },
                "availability": {
                    "uptime_percentage": (responsive_states / total_states) * 100,
                    "total_errors": self._error_count,
                    "error_rate": self.get_error_rate()
                },
                "gpu_stats": {
                    "peak_memory_usage": self._peak_gpu_memory,
                    "current_memory_usage": self.get_gpu_memory_usage()
                }
            }
    
    def get_health_score(self) -> float:
        """Get current server health score (0.0-1.0)"""
        try:
            # Import here to avoid circular imports
            from openai_server.health_monitor import get_health_monitor
            health_monitor = get_health_monitor()
            return health_monitor.get_current_health_score()
        except Exception as e:
            self.logger.warning(f"Failed to get health score: {e}")
            # Fallback health calculation based on basic metrics
            return self._calculate_fallback_health()
    
    def get_health_category(self) -> str:
        """Get current health category"""
        score = self.get_health_score()
        if score >= 0.8:
            return "healthy"
        elif score >= 0.5:
            return "stressed"
        else:
            return "critical"
    
    def get_performance_trend(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get performance trend analysis"""
        try:
            from openai_server.health_monitor import get_health_monitor
            health_monitor = get_health_monitor()
            return health_monitor.get_performance_trend(window_minutes)
        except Exception as e:
            self.logger.warning(f"Failed to get performance trend: {e}")
            return {"trend": "unknown", "error": str(e)}
    
    def _calculate_fallback_health(self) -> float:
        """Fallback health calculation without health monitor"""
        try:
            # Basic health calculation based on current state
            state = self.get_current_state()
            
            # GPU memory score (assume healthy if under 80%)
            gpu_memory = max(state.get('gpu_memory_usage', {}).values()) if state.get('gpu_memory_usage') else 0
            gpu_score = max(0, 1 - (gpu_memory / 100))
            
            # CPU score (assume healthy if under 80%)
            cpu_score = max(0, 1 - (state.get('cpu_usage', 0) / 100))
            
            # Response time score (assume healthy if under 5 seconds)
            response_time = state.get('response_time', 0) or 0
            response_score = max(0, 1 - (response_time / 10))  # 10 second max
            
            # Server responsive score
            responsive_score = 1.0 if state.get('server_responsive', False) else 0.0
            
            # Weighted average
            health_score = (gpu_score * 0.3 + cpu_score * 0.2 + 
                           response_score * 0.3 + responsive_score * 0.2)
            
            return max(0.0, min(1.0, health_score))
            
        except Exception:
            return 0.5  # Neutral score if calculation fails
    
    def should_degrade_service(self) -> Tuple[bool, str]:
        """Check if service should be degraded based on health"""
        health_score = self.get_health_score()
        
        if health_score < 0.3:
            return True, "critical"
        elif health_score < 0.6:
            return True, "stressed"
        else:
            return False, "healthy"
    
    def get_degradation_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for service degradation"""
        health_score = self.get_health_score()
        state = self.get_current_state()
        
        recommendations = {
            "health_score": health_score,
            "should_optimize": health_score < 0.8,
            "optimization_level": "none"
        }
        
        if health_score < 0.8:
            if health_score >= 0.6:
                recommendations["optimization_level"] = "light"
                recommendations.update({
                    "reduce_max_tokens_by": 0.2,
                    "reduce_batch_size_by": 0.1,
                    "increase_timeout_by": 0.1
                })
            elif health_score >= 0.4:
                recommendations["optimization_level"] = "moderate"
                recommendations.update({
                    "reduce_max_tokens_by": 0.4,
                    "reduce_batch_size_by": 0.3,
                    "reduce_temperature_by": 0.2,
                    "truncate_context_by": 0.2
                })
            else:
                recommendations["optimization_level"] = "aggressive"
                recommendations.update({
                    "reduce_max_tokens_by": 0.6,
                    "reduce_batch_size_by": 0.5,
                    "reduce_temperature_by": 0.4,
                    "truncate_context_by": 0.4,
                    "reject_complex_requests": True
                })
        
        return recommendations


def main():
    """Test the server monitor"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Server Monitor Test")
    parser.add_argument("--server-url", default="http://localhost:8000/v1", help="Server URL")
    parser.add_argument("--duration", type=int, default=60, help="Monitoring duration in seconds")
    
    args = parser.parse_args()
    
    monitor = ServerMonitor(args.server_url)
    
    print(f"🔍 Starting server monitoring for {args.duration} seconds...")
    monitor.start_monitoring()
    
    try:
        # Monitor for specified duration
        for i in range(args.duration):
            time.sleep(1)
            if i % 10 == 0:  # Print status every 10 seconds
                state = monitor.get_current_state()
                print(f"  [{i:3d}s] CPU: {state.get('cpu_usage', 0):.1f}% | "
                      f"GPU Mem: {state.get('gpu_memory_usage', {}).get(0, 0):.1f}% | "
                      f"Responsive: {state.get('server_responsive', False)}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Monitoring interrupted by user")
    
    finally:
        monitor.stop_monitoring()
        
        # Print statistics
        stats = monitor.get_statistics()
        print("\n📊 Monitoring Statistics:")
        print(f"  Duration: {stats.get('monitoring_duration_minutes', 0):.1f} minutes")
        print(f"  Availability: {stats.get('availability', {}).get('uptime_percentage', 0):.1f}%")
        print(f"  Avg Response Time: {stats.get('response_times', {}).get('avg', 0):.2f}s")
        print(f"  Peak CPU: {stats.get('cpu_usage', {}).get('max', 0):.1f}%")
        print(f"  Peak GPU Memory: {stats.get('gpu_stats', {}).get('peak_memory_usage', 0):.1f}%")


if __name__ == "__main__":
    main()