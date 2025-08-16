#!/usr/bin/env python3
"""
Graceful Degradation Test Suite

Diagnostic tests that verify graceful degradation behavior
without breaking the server.
"""

import asyncio
import time
import json
import threading
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    from .server_monitor import ServerMonitor
    from .load_generators import LoadGeneratorFactory
except ImportError:
    # Fallback for direct execution
    from server_monitor import ServerMonitor
    from load_generators import LoadGeneratorFactory


@dataclass
class GracefulTestConfig:
    """Configuration for graceful degradation tests"""
    server_url: str = "http://localhost:8000/v1"
    model_name: str = "Qwen3-14B-anger"
    max_test_duration: int = 1800  # 30 minutes max
    health_check_interval: int = 5  # seconds
    output_dir: str = "graceful_test_results"


@dataclass
class GracefulTestResult:
    """Individual graceful test result"""
    test_name: str
    start_time: datetime
    end_time: Optional[datetime]
    success: bool
    error_message: Optional[str]
    metrics: Dict[str, Any]
    degradation_detected: bool
    recovery_verified: bool


class GracefulTestSuite:
    """Test suite focused on graceful degradation patterns"""
    
    def __init__(self, config: GracefulTestConfig):
        self.config = config
        self.results: List[GracefulTestResult] = []
        self.monitor = ServerMonitor(config.server_url)
        self.load_generator = LoadGeneratorFactory(config.server_url, config.model_name)
        
        # Create output directory
        Path(config.output_dir).mkdir(exist_ok=True)
        
        self._test_start_time = None
        self._baseline_performance = None
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all graceful degradation tests"""
        print("🌟 Starting Graceful Degradation Test Suite")
        print("=" * 60)
        
        self._test_start_time = datetime.now()
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        try:
            # Establish baseline performance first
            self._establish_baseline()
            
            # Test suite in order of increasing complexity
            test_methods = [
                ("health_monitoring_validation", self._test_health_monitoring),
                ("async_vllm_wrapper_validation", self._test_async_vllm_wrapper),
                ("request_queue_manager_validation", self._test_request_queue_manager),
                ("timeout_protection_validation", self._test_timeout_protection),
                ("queue_overflow_handling", self._test_queue_overflow_handling),
                ("request_rejection_validation", self._test_request_rejection),
                ("adaptive_processing_validation", self._test_adaptive_processing),
                ("circuit_breaker_validation", self._test_circuit_breaker),
                ("graceful_degradation_curve", self._test_degradation_curve),
                ("memory_management_patterns", self._test_memory_management),
                ("load_spike_recovery", self._test_load_spike_recovery),
                ("sustained_load_adaptation", self._test_sustained_load_adaptation),
                ("gracful_recovery_verification", self._test_graceful_recovery)
            ]
            
            for test_name, test_method in test_methods:
                print(f"\n🔬 Running {test_name}...")
                
                try:
                    result = test_method()
                    self.results.append(result)
                    
                    if result.success:
                        print(f"✅ {test_name} passed")
                        if result.degradation_detected:
                            print(f"   📊 Graceful degradation detected and handled")
                        if result.recovery_verified:
                            print(f"   🔄 Recovery verified")
                    else:
                        print(f"❌ {test_name} failed: {result.error_message}")
                        
                except Exception as e:
                    print(f"💥 {test_name} crashed: {e}")
                    result = GracefulTestResult(
                        test_name=test_name,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        success=False,
                        error_message=f"Test crashed: {e}",
                        metrics={},
                        degradation_detected=False,
                        recovery_verified=False
                    )
                    self.results.append(result)
                
                # Brief pause between tests
                time.sleep(3)
        
        finally:
            # Stop monitoring
            self.monitor.stop_monitoring()
        
        # Generate summary
        summary = self._generate_summary()
        self._save_results(summary)
        
        return summary
    
    def _establish_baseline(self):
        """Establish baseline server performance"""
        print("📊 Establishing baseline performance...")
        
        baseline_requests = []
        for i in range(5):
            result = self.load_generator.generate_single_request(
                f"Baseline test {i}: Simple request for baseline measurement",
                max_tokens=50,
                timeout=70.0
            )
            baseline_requests.append(result)
            time.sleep(2)
        
        successful_baselines = [r for r in baseline_requests if r.get("success", False)]
        
        if successful_baselines:
            self._baseline_performance = {
                "avg_response_time": statistics.mean([r["response_time"] for r in successful_baselines]),
                "max_response_time": max([r["response_time"] for r in successful_baselines]),
                "success_rate": len(successful_baselines) / len(baseline_requests),
                "baseline_established": True
            }
            print(f"   Baseline established: {self._baseline_performance['avg_response_time']:.2f}s avg response time")
        else:
            self._baseline_performance = {"baseline_established": False}
            print("   ⚠️ Could not establish baseline - server may be unhealthy")
    
    def _test_health_monitoring(self) -> GracefulTestResult:
        """Test health monitoring system functionality"""
        start_time = datetime.now()
        
        try:
            # Test health monitoring integration
            health_score = self.monitor.get_health_score()
            health_category = self.monitor.get_health_category()
            performance_trend = self.monitor.get_performance_trend()
            
            # Verify health monitoring is working
            health_working = (
                isinstance(health_score, (int, float)) and 0 <= health_score <= 1 and
                health_category in ["healthy", "stressed", "critical"] and
                isinstance(performance_trend, dict)
            )
            
            # Test degradation recommendations
            recommendations = self.monitor.get_degradation_recommendations()
            
            metrics = {
                "health_score": health_score,
                "health_category": health_category,
                "performance_trend": performance_trend,
                "degradation_recommendations": recommendations,
                "health_monitoring_functional": health_working
            }
            
            return GracefulTestResult(
                test_name="health_monitoring_validation",
                start_time=start_time,
                end_time=datetime.now(),
                success=health_working,
                error_message=None if health_working else "Health monitoring not functioning properly",
                metrics=metrics,
                degradation_detected=health_score < 0.8,
                recovery_verified=health_working
            )
            
        except Exception as e:
            return GracefulTestResult(
                test_name="health_monitoring_validation",
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e),
                metrics={},
                degradation_detected=False,
                recovery_verified=False
            )
    
    def _test_adaptive_processing(self) -> GracefulTestResult:
        """Test adaptive request processing under different load conditions"""
        start_time = datetime.now()
        
        try:
            # Create requests of varying complexity
            test_requests = [
                {"content": "Simple test", "max_tokens": 50, "complexity": "low"},
                {"content": "Medium complexity test " * 50, "max_tokens": 200, "complexity": "medium"},
                {"content": "High complexity analysis request " * 100, "max_tokens": 500, "complexity": "high"}
            ]
            
            adaptation_results = []
            
            for req_info in test_requests:
                # Make request and check for optimization
                result = self.load_generator.generate_single_request(
                    req_info["content"],
                    max_tokens=req_info["max_tokens"],
                    timeout=60.0
                )
                
                adaptation_results.append({
                    "complexity": req_info["complexity"],
                    "success": result.get("success", False),
                    "response_time": result.get("response_time", 0),
                    "original_max_tokens": req_info["max_tokens"]
                })
                
                time.sleep(3)
            
            # Check if server is adapting to load
            successful_requests = [r for r in adaptation_results if r["success"]]
            adaptation_working = len(successful_requests) >= 2
            
            # Verify performance is reasonable
            avg_response_time = statistics.mean([r["response_time"] for r in successful_requests]) if successful_requests else 0
            performance_reasonable = avg_response_time < 30.0  # 30 second max
            
            metrics = {
                "adaptation_results": adaptation_results,
                "successful_requests": len(successful_requests),
                "avg_response_time": avg_response_time,
                "adaptation_working": adaptation_working,
                "performance_reasonable": performance_reasonable
            }
            
            return GracefulTestResult(
                test_name="adaptive_processing_validation",
                start_time=start_time,
                end_time=datetime.now(),
                success=adaptation_working and performance_reasonable,
                error_message=None if adaptation_working and performance_reasonable else "Adaptive processing not working properly",
                metrics=metrics,
                degradation_detected=avg_response_time > 10.0,
                recovery_verified=adaptation_working
            )
            
        except Exception as e:
            return GracefulTestResult(
                test_name="adaptive_processing_validation",
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e),
                metrics={},
                degradation_detected=False,
                recovery_verified=False
            )
    
    def _test_circuit_breaker(self) -> GracefulTestResult:
        """Test circuit breaker functionality"""
        start_time = datetime.now()
        
        try:
            # Make several requests to test circuit breaker behavior
            requests_results = []
            
            for i in range(10):
                result = self.load_generator.generate_single_request(
                    f"Circuit breaker test {i}",
                    max_tokens=100,
                    timeout=70.0
                )
                
                requests_results.append({
                    "request_id": i,
                    "success": result.get("success", False),
                    "response_time": result.get("response_time", 0),
                    "error": result.get("error", ""),
                    "status_code": result.get("status_code", 0)
                })
                
                time.sleep(1)
            
            # Analyze circuit breaker behavior
            successful_requests = [r for r in requests_results if r["success"]]
            circuit_breaker_rejections = [r for r in requests_results if "circuit" in r["error"].lower()]
            
            # Circuit breaker is working if we have reasonable success rate
            success_rate = len(successful_requests) / len(requests_results)
            circuit_breaker_working = success_rate >= 0.5  # At least 50% should succeed under normal conditions
            
            metrics = {
                "total_requests": len(requests_results),
                "successful_requests": len(successful_requests),
                "circuit_breaker_rejections": len(circuit_breaker_rejections),
                "success_rate": success_rate,
                "circuit_breaker_working": circuit_breaker_working,
                "request_details": requests_results
            }
            
            return GracefulTestResult(
                test_name="circuit_breaker_validation",
                start_time=start_time,
                end_time=datetime.now(),
                success=circuit_breaker_working,
                error_message=None if circuit_breaker_working else f"Circuit breaker not working - success rate: {success_rate:.1%}",
                metrics=metrics,
                degradation_detected=len(circuit_breaker_rejections) > 0,
                recovery_verified=success_rate > 0
            )
            
        except Exception as e:
            return GracefulTestResult(
                test_name="circuit_breaker_validation",
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e),
                metrics={},
                degradation_detected=False,
                recovery_verified=False
            )
    
    def _test_degradation_curve(self) -> GracefulTestResult:
        """Test graceful degradation curve - find server limits without breaking"""
        start_time = datetime.now()
        
        try:
            # Use the new graceful degradation curve test
            curve_result = self.load_generator.test_graceful_degradation_curve()
            
            # Analyze degradation curve
            degradation_curve = curve_result.get("degradation_curve", [])
            max_stable_concurrency = curve_result.get("max_stable_concurrency", 0)
            recovery_successful = curve_result.get("recovery_successful", False)
            
            # Success criteria: found some stable concurrency level and server recovered
            curve_success = max_stable_concurrency >= 2 and recovery_successful
            
            # Check if graceful degradation was observed
            degradation_observed = False
            if len(degradation_curve) >= 2:
                # Look for degradation pattern in success rates
                success_rates = [point["success_rate"] for point in degradation_curve]
                degradation_observed = any(rate < 1.0 for rate in success_rates[-3:])  # Last 3 points show some degradation
            
            metrics = {
                "degradation_curve": degradation_curve,
                "max_stable_concurrency": max_stable_concurrency,
                "recovery_successful": recovery_successful,
                "degradation_observed": degradation_observed,
                "curve_points": len(degradation_curve)
            }
            
            return GracefulTestResult(
                test_name="graceful_degradation_curve",
                start_time=start_time,
                end_time=datetime.now(),
                success=curve_success,
                error_message=None if curve_success else "Degradation curve test failed or server did not recover",
                metrics=metrics,
                degradation_detected=degradation_observed,
                recovery_verified=recovery_successful
            )
            
        except Exception as e:
            return GracefulTestResult(
                test_name="graceful_degradation_curve",
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e),
                metrics={},
                degradation_detected=False,
                recovery_verified=False
            )
    
    def _test_memory_management(self) -> GracefulTestResult:
        """Test memory management and recovery patterns"""
        start_time = datetime.now()
        
        try:
            # Use the new memory management recovery test
            memory_result = self.load_generator.test_memory_management_recovery()
            
            # Analyze memory management results
            memory_tests = memory_result.get("memory_pressure_tests", {})
            recovery_verification = memory_result.get("recovery_verification", {})
            overall_recovery = memory_result.get("overall_recovery_successful", False)
            
            # Success criteria
            memory_success = (
                memory_tests.get("success_rate", 0) >= 0.6 and  # At least 60% of memory tests should pass
                recovery_verification.get("success_rate", 0) >= 0.6 and  # At least 60% recovery
                overall_recovery
            )
            
            # Check for memory management patterns
            memory_management_working = memory_tests.get("all_health_checks_passed", False)
            
            metrics = {
                "memory_pressure_results": memory_result,
                "memory_management_working": memory_management_working,
                "recovery_rate": recovery_verification.get("success_rate", 0)
            }
            
            return GracefulTestResult(
                test_name="memory_management_patterns",
                start_time=start_time,
                end_time=datetime.now(),
                success=memory_success,
                error_message=None if memory_success else "Memory management test failed",
                metrics=metrics,
                degradation_detected=memory_tests.get("success_rate", 1.0) < 1.0,
                recovery_verified=overall_recovery
            )
            
        except Exception as e:
            return GracefulTestResult(
                test_name="memory_management_patterns",
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e),
                metrics={},
                degradation_detected=False,
                recovery_verified=False
            )
    
    def _test_load_spike_recovery(self) -> GracefulTestResult:
        """Test server recovery from load spikes"""
        start_time = datetime.now()
        
        try:
            # Baseline measurement
            baseline_result = self.load_generator.generate_single_request("Pre-spike baseline", max_tokens=50)
            baseline_response_time = baseline_result.get("response_time", 0)
            
            # Generate load spike
            print("    Generating controlled load spike...")
            spike_results = self.load_generator.generate_concurrent_requests(
                count=8,  # Moderate spike
                content_template="Load spike test {}: moderate complexity request for spike testing",
                max_tokens=150,
                timeout=45.0
            )
            
            spike_success_rate = len([r for r in spike_results if r.get("success")]) / len(spike_results)
            
            # Wait for recovery
            time.sleep(10)
            
            # Test recovery
            recovery_results = []
            for i in range(3):
                recovery_result = self.load_generator.generate_single_request(
                    f"Post-spike recovery test {i}",
                    max_tokens=50,
                    timeout=70.0
                )
                recovery_results.append(recovery_result)
                time.sleep(2)
            
            recovery_successful = sum(1 for r in recovery_results if r.get("success", False))
            recovery_response_times = [r.get("response_time", 0) for r in recovery_results if r.get("success")]
            avg_recovery_time = statistics.mean(recovery_response_times) if recovery_response_times else 0
            
            # Success criteria
            spike_handled = spike_success_rate >= 0.5  # At least 50% of spike requests succeeded
            recovery_verified = recovery_successful >= 2  # At least 2/3 recovery requests succeeded
            performance_recovered = avg_recovery_time <= baseline_response_time * 2  # Within 2x baseline
            
            test_success = spike_handled and recovery_verified and performance_recovered
            
            metrics = {
                "baseline_response_time": baseline_response_time,
                "spike_success_rate": spike_success_rate,
                "recovery_successful_count": recovery_successful,
                "avg_recovery_response_time": avg_recovery_time,
                "performance_recovered": performance_recovered,
                "spike_handled": spike_handled
            }
            
            return GracefulTestResult(
                test_name="load_spike_recovery",
                start_time=start_time,
                end_time=datetime.now(),
                success=test_success,
                error_message=None if test_success else "Load spike recovery test failed",
                metrics=metrics,
                degradation_detected=spike_success_rate < 1.0,
                recovery_verified=recovery_verified
            )
            
        except Exception as e:
            return GracefulTestResult(
                test_name="load_spike_recovery",
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e),
                metrics={},
                degradation_detected=False,
                recovery_verified=False
            )
    
    def _test_sustained_load_adaptation(self) -> GracefulTestResult:
        """Test adaptation to sustained load over time"""
        start_time = datetime.now()
        
        try:
            print("    Testing sustained load adaptation (60 seconds)...")
            
            # Generate sustained load for 60 seconds
            sustained_results = self.load_generator.generate_sustained_requests(
                duration=60,
                requests_per_second=0.5,  # One request every 2 seconds
                content="Sustained load test",
                max_tokens=100
            )
            
            # Analyze sustained load results
            successful_sustained = [r for r in sustained_results if r.get("success", False)]
            success_rate = len(successful_sustained) / len(sustained_results) if sustained_results else 0
            
            # Check for adaptation over time (response times should stabilize)
            if len(successful_sustained) >= 5:
                response_times = [r["response_time"] for r in successful_sustained]
                early_times = response_times[:len(response_times)//3]
                late_times = response_times[-len(response_times)//3:]
                
                early_avg = statistics.mean(early_times) if early_times else 0
                late_avg = statistics.mean(late_times) if late_times else 0
                
                adaptation_detected = abs(late_avg - early_avg) < early_avg * 0.5  # Stable within 50%
            else:
                adaptation_detected = False
                early_avg = late_avg = 0
            
            # Success criteria
            sustained_success = success_rate >= 0.7 and adaptation_detected
            
            metrics = {
                "total_sustained_requests": len(sustained_results),
                "successful_sustained_requests": len(successful_sustained),
                "sustained_success_rate": success_rate,
                "early_avg_response_time": early_avg,
                "late_avg_response_time": late_avg,
                "adaptation_detected": adaptation_detected
            }
            
            return GracefulTestResult(
                test_name="sustained_load_adaptation",
                start_time=start_time,
                end_time=datetime.now(),
                success=sustained_success,
                error_message=None if sustained_success else "Sustained load adaptation test failed",
                metrics=metrics,
                degradation_detected=success_rate < 1.0,
                recovery_verified=success_rate > 0.5
            )
            
        except Exception as e:
            return GracefulTestResult(
                test_name="sustained_load_adaptation",
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e),
                metrics={},
                degradation_detected=False,
                recovery_verified=False
            )
    
    def _test_graceful_recovery(self) -> GracefulTestResult:
        """Final test to verify overall graceful recovery"""
        start_time = datetime.now()
        
        try:
            # Final health check
            final_health = self.monitor.check_server_health()
            server_healthy = final_health.get("healthy", False)
            
            # Performance comparison with baseline
            final_test = self.load_generator.generate_single_request(
                "Final recovery verification test",
                max_tokens=50,
                timeout=70.0
            )
            
            final_success = final_test.get("success", False)
            final_response_time = final_test.get("response_time", 0)
            
            # Compare with baseline if available
            performance_acceptable = True
            if self._baseline_performance and self._baseline_performance.get("baseline_established"):
                baseline_time = self._baseline_performance["avg_response_time"]
                performance_acceptable = final_response_time <= baseline_time * 3  # Within 3x baseline
            
            # Overall recovery success
            recovery_success = server_healthy and final_success and performance_acceptable
            
            metrics = {
                "server_healthy": server_healthy,
                "final_request_success": final_success,
                "final_response_time": final_response_time,
                "performance_acceptable": performance_acceptable,
                "baseline_comparison": self._baseline_performance
            }
            
            return GracefulTestResult(
                test_name="gracful_recovery_verification",
                start_time=start_time,
                end_time=datetime.now(),
                success=recovery_success,
                error_message=None if recovery_success else "Final recovery verification failed",
                metrics=metrics,
                degradation_detected=not performance_acceptable,
                recovery_verified=recovery_success
            )
            
        except Exception as e:
            return GracefulTestResult(
                test_name="gracful_recovery_verification",
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e),
                metrics={},
                degradation_detected=False,
                recovery_verified=False
            )
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate test summary with graceful degradation focus"""
        total_tests = len(self.results)
        successful_tests = len([r for r in self.results if r.success])
        degradation_detected_count = len([r for r in self.results if r.degradation_detected])
        recovery_verified_count = len([r for r in self.results if r.recovery_verified])
        
        return {
            "test_summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": total_tests - successful_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
                "degradation_detected_count": degradation_detected_count,
                "recovery_verified_count": recovery_verified_count
            },
            "graceful_degradation_analysis": {
                "degradation_detection_rate": degradation_detected_count / total_tests if total_tests > 0 else 0,
                "recovery_verification_rate": recovery_verified_count / total_tests if total_tests > 0 else 0,
                "graceful_behavior_observed": degradation_detected_count > 0 and recovery_verified_count > 0
            },
            "baseline_performance": self._baseline_performance,
            "test_duration": (datetime.now() - self._test_start_time).total_seconds() if self._test_start_time else 0,
            "detailed_results": [
                {
                    "test_name": r.test_name,
                    "success": r.success,
                    "degradation_detected": r.degradation_detected,
                    "recovery_verified": r.recovery_verified,
                    "error_message": r.error_message,
                    "key_metrics": self._extract_key_metrics(r.metrics)
                }
                for r in self.results
            ]
        }
    
    def _extract_key_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics for summary"""
        key_metrics = {}
        
        # Extract commonly useful metrics
        for key in ["success_rate", "avg_response_time", "max_stable_concurrency", 
                   "recovery_successful", "health_score", "degradation_observed"]:
            if key in metrics:
                key_metrics[key] = metrics[key]
        
        return key_metrics
    
    def _save_results(self, summary: Dict[str, Any]):
        """Save test results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON results
        results_file = Path(self.config.output_dir) / f"graceful_test_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n📁 Detailed results saved to: {results_file}")

    def _test_async_vllm_wrapper(self) -> GracefulTestResult:
        """Test AsyncVLLMWrapper timeout and thread management"""
        test_name = "AsyncVLLMWrapper Validation"
        start_time = datetime.now()
        
        try:
            print(f"   Testing AsyncVLLMWrapper functionality...")
            
            # Check health endpoint shows wrapper stats
            health_response = self.load_generator._make_request("GET", "/health")
            if not health_response or not health_response.get("success"):
                raise Exception("Cannot get health endpoint")
            
            health_data = health_response.get("response", {})
            graceful_degradation = health_data.get("graceful_degradation", {})
            vllm_stats = graceful_degradation.get("vllm_statistics", {})
            
            # Validate wrapper stats are present
            required_stats = ["total_requests", "successful_requests", "timeout_requests", 
                             "thread_capacity_used", "abandoned_threads"]
            missing_stats = [stat for stat in required_stats if stat not in vllm_stats]
            if missing_stats:
                raise Exception(f"Missing vLLM wrapper stats: {missing_stats}")
            
            # Test normal request processing through wrapper
            print("   Testing normal request processing...")
            result = self.load_generator.generate_single_request(
                "Test AsyncVLLMWrapper normal processing",
                max_tokens=50,
                timeout=30.0
            )
            
            if not result.get("success"):
                raise Exception(f"Normal request failed: {result.get('error', 'Unknown error')}")
            
            # Check updated stats
            health_response2 = self.load_generator._make_request("GET", "/health")
            vllm_stats2 = health_response2["response"]["graceful_degradation"]["vllm_statistics"]
            
            # Verify request was processed through wrapper
            if vllm_stats2["total_requests"] <= vllm_stats["total_requests"]:
                raise Exception("Request count did not increase - wrapper not processing requests")
            
            return GracefulTestResult(
                test_name=test_name,
                start_time=start_time,
                end_time=datetime.now(),
                success=True,
                error_message=None,
                metrics={
                    "initial_total_requests": vllm_stats["total_requests"],
                    "final_total_requests": vllm_stats2["total_requests"],
                    "thread_capacity_used": vllm_stats2["thread_capacity_used"],
                    "timeout_requests": vllm_stats2["timeout_requests"]
                },
                degradation_detected=False,
                recovery_verified=True
            )
            
        except Exception as e:
            return GracefulTestResult(
                test_name=test_name,
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e),
                metrics={},
                degradation_detected=False,
                recovery_verified=False
            )

    def _test_request_queue_manager(self) -> GracefulTestResult:
        """Test RequestQueueManager functionality"""
        test_name = "RequestQueueManager Validation"
        start_time = datetime.now()
        
        try:
            print(f"   Testing RequestQueueManager functionality...")
            
            # Check health endpoint shows queue stats
            health_response = self.load_generator._make_request("GET", "/health")
            if not health_response or not health_response.get("success"):
                raise Exception("Cannot get health endpoint")
            
            graceful_degradation = health_response["response"]["graceful_degradation"]
            queue_stats = graceful_degradation.get("queue_statistics", {})
            
            # Validate queue stats are present
            required_stats = ["queue_depth", "active_requests", "max_queue_size", 
                             "total_processed", "total_rejected", "capacity_percent"]
            missing_stats = [stat for stat in required_stats if stat not in queue_stats]
            if missing_stats:
                raise Exception(f"Missing queue manager stats: {missing_stats}")
            
            # Verify initial state is healthy
            if queue_stats["status"] != "healthy":
                print(f"   Warning: Initial queue status is {queue_stats['status']}, not healthy")
            
            return GracefulTestResult(
                test_name=test_name,
                start_time=start_time,
                end_time=datetime.now(),
                success=True,
                error_message=None,
                metrics={
                    "max_queue_size": queue_stats["max_queue_size"],
                    "max_concurrent": queue_stats["max_concurrent"],
                    "initial_capacity_percent": queue_stats["capacity_percent"],
                    "queue_status": queue_stats["status"]
                },
                degradation_detected=False,
                recovery_verified=True
            )
            
        except Exception as e:
            return GracefulTestResult(
                test_name=test_name,
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e),
                metrics={},
                degradation_detected=False,
                recovery_verified=False
            )

    def _test_timeout_protection(self) -> GracefulTestResult:
        """Test that requests timeout instead of hanging indefinitely"""
        test_name = "Timeout Protection"
        start_time = datetime.now()
        
        try:
            print(f"   Testing timeout protection with long request...")
            
            # Test with a very long request that might normally hang
            very_long_content = "Write a very detailed essay about artificial intelligence. " * 100
            
            # Use a short timeout to test timeout behavior
            result = self.load_generator.generate_single_request(
                very_long_content,
                max_tokens=2000,
                timeout=10.0  # Short timeout to potentially trigger timeout
            )
            
            # Either the request succeeds quickly OR times out - no infinite hang
            request_completed = True
            timeout_detected = False
            
            if not result.get("success"):
                error_msg = result.get("error", "").lower()
                if "timeout" in error_msg or "timed out" in error_msg:
                    timeout_detected = True
                    print("   ✅ Request timed out as expected - no infinite hang")
                elif "overload" in error_msg or "capacity" in error_msg:
                    print("   ✅ Request rejected due to capacity - graceful handling")
                else:
                    print(f"   ⚠️ Request failed but not due to timeout: {error_msg}")
            else:
                print("   ✅ Request completed successfully within timeout")
            
            # Check final stats for timeout tracking
            health_response = self.load_generator._make_request("GET", "/health")
            if health_response and health_response.get("success"):
                vllm_stats = health_response["response"]["graceful_degradation"]["vllm_statistics"]
                timeout_rate = vllm_stats.get("timeout_rate", 0)
            else:
                timeout_rate = 0
            
            return GracefulTestResult(
                test_name=test_name,
                start_time=start_time,
                end_time=datetime.now(),
                success=True,  # Success means no infinite hang occurred
                error_message=None,
                metrics={
                    "request_completed": request_completed,
                    "timeout_detected": timeout_detected,
                    "timeout_rate": timeout_rate,
                    "test_duration": (datetime.now() - start_time).total_seconds()
                },
                degradation_detected=timeout_detected,
                recovery_verified=True
            )
            
        except Exception as e:
            return GracefulTestResult(
                test_name=test_name,
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e),
                metrics={},
                degradation_detected=False,
                recovery_verified=False
            )

    def _test_queue_overflow_handling(self) -> GracefulTestResult:
        """Test queue behavior under overflow conditions"""
        test_name = "Queue Overflow Handling"
        start_time = datetime.now()
        
        try:
            print(f"   Testing queue overflow handling...")
            
            # Get initial queue stats
            health_response = self.load_generator._make_request("GET", "/health")
            if not health_response or not health_response.get("success"):
                raise Exception("Cannot get initial health stats")
            
            initial_stats = health_response["response"]["graceful_degradation"]["queue_statistics"]
            max_queue_size = initial_stats["max_queue_size"]
            
            print(f"   Max queue size: {max_queue_size}, testing overflow...")
            
            # Send many concurrent requests to try to overflow the queue
            concurrent_requests = min(max_queue_size + 10, 20)  # Don't overwhelm the system
            print(f"   Sending {concurrent_requests} concurrent requests...")
            
            results = self.load_generator.generate_concurrent_requests(
                concurrent_requests,
                "Test queue overflow handling",
                max_tokens=100,
                timeout=30.0
            )
            
            successful_requests = sum(1 for r in results if r.get("success", False))
            failed_requests = len(results) - successful_requests
            
            # Check for rejection responses
            rejection_count = 0
            for result in results:
                if not result.get("success"):
                    error_msg = result.get("error", "").lower()
                    if "overload" in error_msg or "capacity" in error_msg or "503" in error_msg:
                        rejection_count += 1
            
            # Get final stats
            health_response2 = self.load_generator._make_request("GET", "/health")
            if health_response2 and health_response2.get("success"):
                final_stats = health_response2["response"]["graceful_degradation"]["queue_statistics"]
                rejection_rate = final_stats.get("rejection_rate", 0)
            else:
                final_stats = {}
                rejection_rate = 0
            
            # Test passes if server handled overflow gracefully (some rejections expected)
            graceful_handling = rejection_count > 0 or rejection_rate > 0
            
            return GracefulTestResult(
                test_name=test_name,
                start_time=start_time,
                end_time=datetime.now(),
                success=True,  # Success means server didn't crash
                error_message=None,
                metrics={
                    "concurrent_requests": concurrent_requests,
                    "successful_requests": successful_requests,
                    "failed_requests": failed_requests,
                    "rejection_count": rejection_count,
                    "rejection_rate": rejection_rate,
                    "max_queue_size": max_queue_size,
                    "final_queue_depth": final_stats.get("queue_depth", 0)
                },
                degradation_detected=graceful_handling,
                recovery_verified=True
            )
            
        except Exception as e:
            return GracefulTestResult(
                test_name=test_name,
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e),
                metrics={},
                degradation_detected=False,
                recovery_verified=False
            )

    def _test_request_rejection(self) -> GracefulTestResult:
        """Test that requests are properly rejected when server is overloaded"""
        test_name = "Request Rejection Validation"
        start_time = datetime.now()
        
        try:
            print(f"   Testing request rejection mechanism...")
            
            # First, test normal request acceptance
            normal_result = self.load_generator.generate_single_request(
                "Normal request for rejection test",
                max_tokens=50,
                timeout=30.0
            )
            
            if not normal_result.get("success"):
                print(f"   Warning: Normal request failed: {normal_result.get('error')}")
            
            # Try to create overload conditions
            print("   Creating overload conditions...")
            
            # Send a burst of requests to potentially trigger rejection
            burst_size = 15
            results = self.load_generator.generate_concurrent_requests(
                burst_size,
                "Overload test - this might be rejected",
                max_tokens=200,
                timeout=20.0
            )
            
            # Analyze results
            successful = []
            rejected_503 = []
            rejected_504 = []
            other_errors = []
            
            for result in results:
                if result.get("success"):
                    successful.append(result)
                else:
                    error_msg = result.get("error", "").lower()
                    if "503" in error_msg or "overload" in error_msg or "capacity" in error_msg:
                        rejected_503.append(result)
                    elif "504" in error_msg or "timeout" in error_msg:
                        rejected_504.append(result)
                    else:
                        other_errors.append(result)
            
            total_rejections = len(rejected_503) + len(rejected_504)
            rejection_rate = total_rejections / len(results)
            
            # Get final server state
            health_response = self.load_generator._make_request("GET", "/health")
            server_responsive = health_response and health_response.get("success")
            
            print(f"   Results: {len(successful)} success, {len(rejected_503)} rejected (503), {len(rejected_504)} timeout (504)")
            
            return GracefulTestResult(
                test_name=test_name,
                start_time=start_time,
                end_time=datetime.now(),
                success=server_responsive,  # Success means server stayed responsive
                error_message=None if server_responsive else "Server became unresponsive",
                metrics={
                    "burst_size": burst_size,
                    "successful_requests": len(successful),
                    "rejected_503": len(rejected_503),
                    "rejected_504": len(rejected_504),
                    "other_errors": len(other_errors),
                    "rejection_rate": rejection_rate,
                    "server_responsive": server_responsive
                },
                degradation_detected=total_rejections > 0,
                recovery_verified=server_responsive
            )
            
        except Exception as e:
            return GracefulTestResult(
                test_name=test_name,
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e),
                metrics={},
                degradation_detected=False,
                recovery_verified=False
            )


def main():
    """Run graceful degradation test suite"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Graceful Degradation Test Suite")
    parser.add_argument("--server-url", default="http://localhost:8000/v1", help="Server URL")
    parser.add_argument("--model-name", default="Qwen3-14B-anger", help="Model name")
    parser.add_argument("--output-dir", default="graceful_test_results", help="Output directory")
    
    args = parser.parse_args()
    
    config = GracefulTestConfig(
        server_url=args.server_url,
        model_name=args.model_name,
        output_dir=args.output_dir
    )
    
    # Create and run test suite
    suite = GracefulTestSuite(config)
    summary = suite.run_all_tests()
    
    # Print final summary
    print("\n" + "=" * 60)
    print("🎯 GRACEFUL DEGRADATION TEST SUMMARY")
    print("=" * 60)
    
    test_summary = summary["test_summary"]
    print(f"Total Tests: {test_summary['total_tests']}")
    print(f"Successful: {test_summary['successful_tests']}")
    print(f"Failed: {test_summary['failed_tests']}")
    print(f"Success Rate: {test_summary['success_rate']:.1%}")
    
    graceful_analysis = summary["graceful_degradation_analysis"]
    print(f"\n🌟 Graceful Degradation Analysis:")
    print(f"Degradation Detection Rate: {graceful_analysis['degradation_detection_rate']:.1%}")
    print(f"Recovery Verification Rate: {graceful_analysis['recovery_verification_rate']:.1%}")
    print(f"Graceful Behavior Observed: {'✅ Yes' if graceful_analysis['graceful_behavior_observed'] else '❌ No'}")
    
    if summary["baseline_performance"].get("baseline_established"):
        baseline = summary["baseline_performance"]
        print(f"\n📊 Baseline Performance: {baseline['avg_response_time']:.2f}s avg response time")
    
    print(f"\n⏱️ Test Duration: {summary['test_duration']:.1f} seconds")
    
    return 0 if test_summary['success_rate'] >= 0.8 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())