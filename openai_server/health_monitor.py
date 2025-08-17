#!/usr/bin/env python3
"""
Health Monitor for OpenAI Server Graceful Degradation

Provides real-time health scoring and performance trend analysis
to enable adaptive request processing without hard limits.
"""

import time
import threading
import subprocess
import psutil
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import logging
import statistics


@dataclass
class HealthMetrics:
    """Health metrics snapshot at a point in time"""
    timestamp: datetime
    gpu_memory_percent: float
    cpu_percent: float
    system_memory_percent: float
    avg_response_time: float
    error_rate: float
    active_connections: int
    health_score: float


class HealthMonitor:
    """Real-time health monitoring with adaptive thresholds"""
    
    def __init__(self, history_window_minutes: int = 10):
        self.history_window = timedelta(minutes=history_window_minutes)
        self.metrics_history: deque = deque()
        self.response_times: deque = deque(maxlen=100)  # Last 100 response times
        self.error_count = 0
        self.total_requests = 0
        
        self._monitoring = False
        self._monitor_thread = None
        self._lock = threading.Lock()
        
        # Adaptive thresholds - these adjust based on observed baseline
        self.baseline_gpu_memory = 20.0  # Will be updated based on observations
        self.baseline_response_time = 2.0  # Will be updated based on observations
        self.baseline_cpu = 15.0  # Will be updated based on observations
        
        # Health score weights
        self.weights = {
            "gpu_memory": 0.4,
            "response_time": 0.3,
            "cpu": 0.2,
            "error_rate": 0.1
        }
        
        self.logger = logging.getLogger("HealthMonitor")
        self.logger.setLevel(logging.INFO)
    
    def start_monitoring(self):
        """Start continuous health monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("Health monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop - runs every 2 seconds"""
        baseline_samples = 0
        baseline_update_interval = 30  # Update baseline every 30 samples
        
        while self._monitoring:
            try:
                # Collect current metrics
                gpu_memory = self._get_gpu_memory_usage()
                cpu_percent = psutil.cpu_percent(interval=0.5)
                memory_percent = psutil.virtual_memory().percent
                avg_response_time = self._get_avg_response_time()
                error_rate = self._get_error_rate()
                active_connections = self._get_active_connections()
                
                # Calculate health score
                health_score = self._calculate_health_score(
                    gpu_memory, cpu_percent, avg_response_time, error_rate
                )
                
                # Create metrics snapshot
                metrics = HealthMetrics(
                    timestamp=datetime.now(),
                    gpu_memory_percent=gpu_memory,
                    cpu_percent=cpu_percent,
                    system_memory_percent=memory_percent,
                    avg_response_time=avg_response_time,
                    error_rate=error_rate,
                    active_connections=active_connections,
                    health_score=health_score
                )
                
                # Store metrics with thread safety
                with self._lock:
                    self.metrics_history.append(metrics)
                    # Remove old metrics outside window
                    cutoff_time = datetime.now() - self.history_window
                    while (self.metrics_history and 
                           self.metrics_history[0].timestamp < cutoff_time):
                        self.metrics_history.popleft()
                
                # Update adaptive baselines periodically
                baseline_samples += 1
                if baseline_samples >= baseline_update_interval:
                    self._update_baselines()
                    baseline_samples = 0
                
                time.sleep(2)  # Monitor every 2 seconds
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                time.sleep(5)  # Longer sleep on error
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage percentage"""
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                max_memory = torch.cuda.max_memory_allocated()
                
                if max_memory > 0:
                    return (allocated / max_memory) * 100
                else:
                    # Use nvidia-smi as fallback
                    result = subprocess.run([
                        'nvidia-smi', '--query-gpu=memory.used,memory.total',
                        '--format=csv,noheader,nounits'
                    ], capture_output=True, text=True, timeout=3)
                    
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        for line in lines:
                            parts = line.split(',')
                            if len(parts) == 2:
                                used = float(parts[0].strip())
                                total = float(parts[1].strip())
                                return (used / total) * 100 if total > 0 else 0
            return 0.0
        except Exception as e:
            self.logger.warning(f"Failed to get GPU memory usage: {e}")
            return 0.0
    
    def _get_active_connections(self) -> int:
        """Get number of active connections to server"""
        try:
            connections = [conn for conn in psutil.net_connections() 
                         if (conn.laddr.port in [8000, 8001, 8002, 8003] and 
                             conn.status == psutil.CONN_ESTABLISHED)]
            return len(connections)
        except Exception:
            return 0
    
    def _get_avg_response_time(self) -> float:
        """Get average response time from recent requests"""
        if not self.response_times:
            return self.baseline_response_time
            
        # Weight recent response times more heavily
        recent_times = list(self.response_times)[-20:]  # Last 20 responses
        return statistics.mean(recent_times) if recent_times else self.baseline_response_time
    
    def _get_error_rate(self) -> float:
        """Get current error rate"""
        if self.total_requests == 0:
            return 0.0
        return self.error_count / self.total_requests
    
    def _calculate_health_score(self, gpu_memory: float, cpu: float, 
                               response_time: float, error_rate: float) -> float:
        """Calculate overall health score (0.0 = critical, 1.0 = excellent)"""
        
        # GPU memory score (lower usage = better health)
        gpu_score = max(0, 1 - (gpu_memory / 100.0))
        if gpu_memory > self.baseline_gpu_memory * 2:
            gpu_score *= 0.5  # Penalize high GPU usage
        
        # CPU score (lower usage = better health)  
        cpu_score = max(0, 1 - (cpu / 100.0))
        if cpu > self.baseline_cpu * 3:
            cpu_score *= 0.3  # Heavily penalize high CPU
        
        # Response time score (faster = better health)
        if response_time <= self.baseline_response_time:
            response_score = 1.0
        else:
            # Exponential penalty for slow responses
            response_score = max(0, 1 - ((response_time - self.baseline_response_time) / 
                                       (self.baseline_response_time * 5)))
        
        # Error rate score (fewer errors = better health)
        error_score = max(0, 1 - (error_rate * 10))  # 10% error rate = 0 score
        
        # Weighted combination
        health_score = (
            gpu_score * self.weights["gpu_memory"] +
            response_score * self.weights["response_time"] +
            cpu_score * self.weights["cpu"] +
            error_score * self.weights["error_rate"]
        )
        
        return max(0.0, min(1.0, health_score))
    
    def _update_baselines(self):
        """Update baseline metrics based on recent observations"""
        with self._lock:
            if len(self.metrics_history) < 10:
                return
                
            recent_metrics = list(self.metrics_history)[-50:]  # Last 50 samples
            
            # Update GPU baseline (25th percentile of recent usage)
            gpu_values = [m.gpu_memory_percent for m in recent_metrics]
            self.baseline_gpu_memory = statistics.quantiles(gpu_values, n=4)[0] if gpu_values else 20.0
            
            # Update CPU baseline (25th percentile of recent usage)
            cpu_values = [m.cpu_percent for m in recent_metrics]
            self.baseline_cpu = statistics.quantiles(cpu_values, n=4)[0] if cpu_values else 15.0
            
            # Update response time baseline (median of recent times)
            response_values = [m.avg_response_time for m in recent_metrics if m.avg_response_time > 0]
            if response_values:
                self.baseline_response_time = statistics.median(response_values)
            
            self.logger.debug(f"Updated baselines - GPU: {self.baseline_gpu_memory:.1f}%, "
                            f"CPU: {self.baseline_cpu:.1f}%, Response: {self.baseline_response_time:.2f}s")
    
    def get_current_health_score(self) -> float:
        """Get current health score"""
        with self._lock:
            if not self.metrics_history:
                return 1.0  # Assume healthy if no data
            return self.metrics_history[-1].health_score
    
    def get_health_category(self) -> str:
        """Get current health category"""
        score = self.get_current_health_score()
        if score >= 0.8:
            return "healthy"
        elif score >= 0.5:
            return "stressed"
        else:
            return "critical"
    
    def get_performance_trend(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get performance trend over specified window"""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        with self._lock:
            recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
            
            if len(recent_metrics) < 2:
                return {"trend": "insufficient_data", "samples": len(recent_metrics)}
            
            # Calculate trends
            timestamps = [(m.timestamp - recent_metrics[0].timestamp).total_seconds() 
                         for m in recent_metrics]
            health_scores = [m.health_score for m in recent_metrics]
            response_times = [m.avg_response_time for m in recent_metrics]
            
            # Simple linear trend calculation
            health_trend = self._calculate_trend(timestamps, health_scores)
            response_trend = self._calculate_trend(timestamps, response_times)
            
            return {
                "trend": "improving" if health_trend > 0.01 else 
                        "degrading" if health_trend < -0.01 else "stable",
                "health_trend_slope": health_trend,
                "response_trend_slope": response_trend,
                "current_score": health_scores[-1],
                "avg_score": statistics.mean(health_scores),
                "samples": len(recent_metrics),
                "window_minutes": window_minutes
            }
    
    def _calculate_trend(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate simple linear trend slope"""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
            
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0.0
            
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def record_request(self, response_time: float, success: bool):
        """Record a request for metrics tracking"""
        self.response_times.append(response_time)
        self.total_requests += 1
        if not success:
            self.error_count += 1
    
    def get_current_metrics(self) -> Optional[HealthMetrics]:
        """Get current metrics snapshot"""
        with self._lock:
            if not self.metrics_history:
                return None
            return self.metrics_history[-1]
    
    def get_metrics_history(self, minutes: int = 10) -> List[HealthMetrics]:
        """Get metrics history for specified time window"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self._lock:
            return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_resource_pressure(self) -> Dict[str, str]:
        """Get current resource pressure levels"""
        current = self.get_current_metrics()
        if not current:
            return {"gpu": "unknown", "cpu": "unknown", "memory": "unknown"}
        
        gpu_pressure = ("high" if current.gpu_memory_percent > self.baseline_gpu_memory * 2 
                       else "medium" if current.gpu_memory_percent > self.baseline_gpu_memory * 1.5 
                       else "low")
        
        cpu_pressure = ("high" if current.cpu_percent > self.baseline_cpu * 3 
                       else "medium" if current.cpu_percent > self.baseline_cpu * 2 
                       else "low")
        
        memory_pressure = ("high" if current.system_memory_percent > 85 
                          else "medium" if current.system_memory_percent > 70 
                          else "low")
        
        return {
            "gpu": gpu_pressure,
            "cpu": cpu_pressure,
            "memory": memory_pressure,
            "overall": max(gpu_pressure, cpu_pressure, memory_pressure, key=lambda x: {"low": 0, "medium": 1, "high": 2}[x])
        }
    
    def should_reject_request(self) -> Tuple[bool, str]:
        """Determine if new requests should be rejected"""
        score = self.get_current_health_score()
        pressure = self.get_resource_pressure()
        
        # Only reject in truly critical situations
        if score < 0.2 and pressure["overall"] == "high":
            return True, "Server overloaded - please try again later"
        
        if score < 0.1:
            return True, "Server in critical state - rejecting new requests"
        
        return False, ""
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for request optimization based on current health"""
        score = self.get_current_health_score()
        pressure = self.get_resource_pressure()
        current = self.get_current_metrics()
        
        if not current:
            return {"optimize": False}
        
        recommendations = {"optimize": score < 0.8}
        
        if score < 0.8:  # Stressed or critical
            recommendations.update({
                "reduce_max_tokens": score < 0.6,
                "reduce_temperature": score < 0.5,
                "truncate_context": score < 0.4,
                "priority_processing": score < 0.6,
                "max_tokens_multiplier": max(0.3, score),  # Reduce tokens based on health
                "temperature_multiplier": max(0.5, score),  # Reduce temperature based on health
                "context_retention_ratio": max(0.5, score * 1.2),  # Keep more context when healthier
                "batch_size_multiplier": max(0.3, score * 1.5)  # Reduce batch size when unhealthy
            })
        
        return recommendations


# Global health monitor instance
_health_monitor = None

def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance"""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor


def main():
    """Test the health monitor"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Health Monitor Test")
    parser.add_argument("--duration", type=int, default=60, help="Monitoring duration in seconds")
    args = parser.parse_args()
    
    print("🔍 Starting health monitoring test...")
    
    monitor = HealthMonitor()
    monitor.start_monitoring()
    
    try:
        for i in range(args.duration):
            time.sleep(1)
            if i % 10 == 0:  # Print status every 10 seconds
                score = monitor.get_current_health_score()
                category = monitor.get_health_category()
                metrics = monitor.get_current_metrics()
                trend = monitor.get_performance_trend()
                
                print(f"[{i:3d}s] Health: {score:.2f} ({category}) | "
                      f"GPU: {metrics.gpu_memory_percent if metrics else 0:.1f}% | "
                      f"Response: {metrics.avg_response_time if metrics else 0:.2f}s | "
                      f"Trend: {trend['trend']}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Monitoring interrupted")
    
    finally:
        monitor.stop_monitoring()
        print("Health monitoring test completed!")


if __name__ == "__main__":
    main()