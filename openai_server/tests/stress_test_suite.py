#!/usr/bin/env python3
"""
Comprehensive Stress Test Suite for OpenAI Server

This suite is designed to identify scenarios where the server gets stuck,
becomes unresponsive, or exhibits performance degradation.
"""

import asyncio
import time
import json
import threading
import queue
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import concurrent.futures
import statistics

try:
    from .server_monitor import ServerMonitor
    from .hang_detector import HangDetector
    from .load_generators import LoadGeneratorFactory
    from .stress_report_generator import StressReportGenerator
except ImportError:
    # Fallback for direct execution
    from server_monitor import ServerMonitor
    from hang_detector import HangDetector
    from load_generators import LoadGeneratorFactory
    from stress_report_generator import StressReportGenerator


@dataclass
class TestConfig:
    """Configuration for stress tests"""
    server_url: str = "http://localhost:8000/v1"
    model_name: str = "Qwen3-14B-anger"
    max_concurrent_requests: int = 50
    max_test_duration: int = 3600  # 1 hour max
    hang_timeout: int = 300  # 5 minutes
    safety_timeout: int = 600  # 10 minutes absolute max
    output_dir: str = "stress_test_results"
    enable_gpu_monitoring: bool = True
    enable_recovery_tests: bool = True


@dataclass
class TestResult:
    """Individual test result"""
    test_name: str
    start_time: datetime
    end_time: Optional[datetime]
    success: bool
    error_message: Optional[str]
    metrics: Dict[str, Any]
    server_state: Dict[str, Any]


class StressTestSuite:
    """Main stress test orchestrator"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.results: List[TestResult] = []
        self.monitor = ServerMonitor(config.server_url)
        self.hang_detector = HangDetector(config.hang_timeout)
        self.load_generator = LoadGeneratorFactory(config.server_url, config.model_name)
        self.report_generator = StressReportGenerator(config.output_dir)
        
        self._shutdown_event = threading.Event()
        self._active_tests = set()
        self._test_start_time = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n🛑 Received signal {signum}, initiating graceful shutdown...")
        self._shutdown_event.set()
    
    def _check_safety_limits(self) -> bool:
        """Check if we should continue testing based on safety limits"""
        if self._shutdown_event.is_set():
            return False
        
        if self._test_start_time:
            elapsed = datetime.now() - self._test_start_time
            if elapsed.total_seconds() > self.config.safety_timeout:
                print(f"⚠️ Safety timeout reached ({self.config.safety_timeout}s), stopping tests")
                return False
        
        return True
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all stress tests and return summary"""
        print("🚀 Starting Comprehensive Server Stress Tests")
        print("=" * 60)
        
        self._test_start_time = datetime.now()
        
        # Start monitoring
        self.monitor.start_monitoring()
        self.hang_detector.start_monitoring()
        
        try:
            # Test categories in order of increasing stress
            test_methods = [
                ("server_health_baseline", self._test_server_health_baseline),
                ("basic_concurrent_load", self._test_basic_concurrent_load),
                ("progressive_context_length", self._test_progressive_context_length),
                ("rapid_fire_requests", self._test_rapid_fire_requests),
                ("mixed_load_patterns", self._test_mixed_load_patterns),
                ("resource_exhaustion", self._test_resource_exhaustion),
                ("hang_detection_scenarios", self._test_hang_detection_scenarios),
                ("recovery_scenarios", self._test_recovery_scenarios),
            ]
            
            for test_name, test_method in test_methods:
                if not self._check_safety_limits():
                    break
                
                print(f"\n📋 Running {test_name}...")
                self._active_tests.add(test_name)
                
                try:
                    result = test_method()
                    self.results.append(result)
                    
                    if result.success:
                        print(f"✅ {test_name} completed successfully")
                    else:
                        print(f"❌ {test_name} failed: {result.error_message}")
                        
                        # Stop on critical failures
                        if "server_unresponsive" in str(result.error_message):
                            print("🛑 Server appears unresponsive, stopping tests")
                            break
                            
                except Exception as e:
                    print(f"💥 {test_name} crashed: {e}")
                    result = TestResult(
                        test_name=test_name,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        success=False,
                        error_message=f"Test crashed: {e}",
                        metrics={},
                        server_state=self.monitor.get_current_state()
                    )
                    self.results.append(result)
                
                finally:
                    self._active_tests.discard(test_name)
                    time.sleep(2)  # Brief pause between tests
        
        finally:
            # Stop monitoring
            self.monitor.stop_monitoring()
            self.hang_detector.stop_monitoring()
        
        # Generate comprehensive report
        summary = self._generate_summary()
        self.report_generator.generate_report(self.results, summary)
        
        return summary
    
    def _test_server_health_baseline(self) -> TestResult:
        """Establish baseline server performance"""
        start_time = datetime.now()
        
        try:
            # Simple health check
            health_response = self.monitor.check_server_health()
            
            # Single request baseline
            baseline_response = self.load_generator.generate_single_request(
                content="Baseline test: Please respond with 'OK'",
                max_tokens=10
            )
            
            metrics = {
                "server_healthy": health_response.get("healthy", False),
                "baseline_response_time": baseline_response.get("response_time", 0),
                "baseline_tokens": baseline_response.get("tokens_used", 0),
                "gpu_memory_usage": self.monitor.get_gpu_memory_usage(),
                "cpu_usage": self.monitor.get_cpu_usage(),
            }
            
            success = health_response.get("healthy", False) and baseline_response.get("success", False)
            
            return TestResult(
                test_name="server_health_baseline",
                start_time=start_time,
                end_time=datetime.now(),
                success=success,
                error_message=None if success else "Baseline test failed",
                metrics=metrics,
                server_state=self.monitor.get_current_state()
            )
            
        except Exception as e:
            return TestResult(
                test_name="server_health_baseline",
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e),
                metrics={},
                server_state=self.monitor.get_current_state()
            )
    
    def _test_basic_concurrent_load(self) -> TestResult:
        """Test basic concurrent request handling"""
        start_time = datetime.now()
        
        concurrent_levels = [2, 5, 10, 15, 20]
        results = {}
        
        try:
            for level in concurrent_levels:
                if not self._check_safety_limits():
                    break
                
                print(f"  Testing {level} concurrent requests...")
                
                # Generate concurrent requests
                concurrent_results = self.load_generator.generate_concurrent_requests(
                    count=level,
                    content_template="Concurrent test {}: Please respond briefly.",
                    max_tokens=50
                )
                
                # Analyze results
                successful = [r for r in concurrent_results if r.get("success", False)]
                response_times = [r.get("response_time", 0) for r in successful]
                
                results[f"concurrent_{level}"] = {
                    "total_requests": level,
                    "successful_requests": len(successful),
                    "success_rate": len(successful) / level,
                    "avg_response_time": statistics.mean(response_times) if response_times else 0,
                    "max_response_time": max(response_times) if response_times else 0,
                    "min_response_time": min(response_times) if response_times else 0,
                }
                
                # Check for signs of server stress
                if len(successful) < level * 0.8:  # Less than 80% success
                    print(f"  ⚠️ Success rate dropped to {len(successful)/level:.1%}")
                    break
                    
                time.sleep(1)  # Brief pause between levels
            
            # Overall assessment
            success_rates = [results[k]["success_rate"] for k in results.keys()]
            overall_success = all(rate >= 0.8 for rate in success_rates)
            
            return TestResult(
                test_name="basic_concurrent_load",
                start_time=start_time,
                end_time=datetime.now(),
                success=overall_success,
                error_message=None if overall_success else "Concurrent load test showed degradation",
                metrics=results,
                server_state=self.monitor.get_current_state()
            )
            
        except Exception as e:
            return TestResult(
                test_name="basic_concurrent_load",
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e),
                metrics=results,
                server_state=self.monitor.get_current_state()
            )
    
    def _test_progressive_context_length(self) -> TestResult:
        """Test with progressively longer context lengths"""
        start_time = datetime.now()
        
        context_lengths = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
        results = {}
        
        try:
            for length in context_lengths:
                if not self._check_safety_limits():
                    break
                
                print(f"  Testing {length} character context...")
                
                # Generate long context
                long_content = self.load_generator.generate_long_content(length)
                
                # Test with timeout monitoring
                with self.hang_detector.monitor_request(f"context_{length}"):
                    result = self.load_generator.generate_single_request(
                        content=f"Analyze this text ({length} chars): {long_content}",
                        max_tokens=100,
                        timeout=300  # 5 minute timeout
                    )
                
                if result.get("success"):
                    results[f"context_{length}"] = {
                        "input_length": length,
                        "response_time": result.get("response_time", 0),
                        "tokens_used": result.get("tokens_used", 0),
                        "success": True
                    }
                else:
                    results[f"context_{length}"] = {
                        "input_length": length,
                        "error": result.get("error", "Unknown error"),
                        "success": False
                    }
                    
                    # Stop if we hit the limit
                    if "timeout" in str(result.get("error", "")).lower():
                        print(f"  📏 Context limit reached at {length} characters")
                        break
                
                time.sleep(2)  # Pause between length tests
            
            successful_tests = [k for k, v in results.items() if v.get("success", False)]
            max_successful_length = max([results[k]["input_length"] for k in successful_tests]) if successful_tests else 0
            
            return TestResult(
                test_name="progressive_context_length",
                start_time=start_time,
                end_time=datetime.now(),
                success=len(successful_tests) > 0,
                error_message=None if successful_tests else "No context length tests succeeded",
                metrics={
                    "results": results,
                    "max_successful_length": max_successful_length,
                    "successful_tests": len(successful_tests),
                    "total_tests": len(results)
                },
                server_state=self.monitor.get_current_state()
            )
            
        except Exception as e:
            return TestResult(
                test_name="progressive_context_length",
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e),
                metrics=results,
                server_state=self.monitor.get_current_state()
            )
    
    def _test_rapid_fire_requests(self) -> TestResult:
        """Test rapid succession of requests"""
        start_time = datetime.now()
        
        try:
            print("  Testing rapid fire request patterns...")
            
            # Burst pattern: 20 requests in quick succession
            burst_results = self.load_generator.generate_burst_requests(
                count=20,
                content="Quick test",
                max_tokens=20,
                delay_between_requests=0.1
            )
            
            # Sustained pattern: 1 request per second for 60 seconds
            sustained_results = self.load_generator.generate_sustained_requests(
                duration=60,
                requests_per_second=1,
                content="Sustained test",
                max_tokens=30
            )
            
            # Analyze patterns
            burst_successful = [r for r in burst_results if r.get("success", False)]
            sustained_successful = [r for r in sustained_results if r.get("success", False)]
            
            metrics = {
                "burst_pattern": {
                    "total_requests": len(burst_results),
                    "successful_requests": len(burst_successful),
                    "success_rate": len(burst_successful) / len(burst_results) if burst_results else 0,
                    "avg_response_time": statistics.mean([r.get("response_time", 0) for r in burst_successful]) if burst_successful else 0
                },
                "sustained_pattern": {
                    "total_requests": len(sustained_results),
                    "successful_requests": len(sustained_successful),
                    "success_rate": len(sustained_successful) / len(sustained_results) if sustained_results else 0,
                    "avg_response_time": statistics.mean([r.get("response_time", 0) for r in sustained_successful]) if sustained_successful else 0
                }
            }
            
            # Success if both patterns work reasonably well
            success = (metrics["burst_pattern"]["success_rate"] >= 0.7 and 
                      metrics["sustained_pattern"]["success_rate"] >= 0.8)
            
            return TestResult(
                test_name="rapid_fire_requests",
                start_time=start_time,
                end_time=datetime.now(),
                success=success,
                error_message=None if success else "Rapid fire patterns showed poor performance",
                metrics=metrics,
                server_state=self.monitor.get_current_state()
            )
            
        except Exception as e:
            return TestResult(
                test_name="rapid_fire_requests",
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e),
                metrics={},
                server_state=self.monitor.get_current_state()
            )
    
    def _test_mixed_load_patterns(self) -> TestResult:
        """Test mixed load patterns that might cause conflicts"""
        start_time = datetime.now()
        
        try:
            print("  Testing mixed load patterns...")
            
            # Mixed pattern: concurrent long and short requests
            mixed_results = self.load_generator.generate_mixed_load(
                short_requests=10,
                long_requests=3,
                concurrent_execution=True
            )
            
            # Analyze mixed results
            short_results = [r for r in mixed_results if r.get("type") == "short"]
            long_results = [r for r in mixed_results if r.get("type") == "long"]
            
            short_successful = [r for r in short_results if r.get("success", False)]
            long_successful = [r for r in long_results if r.get("success", False)]
            
            metrics = {
                "short_requests": {
                    "total": len(short_results),
                    "successful": len(short_successful),
                    "success_rate": len(short_successful) / len(short_results) if short_results else 0,
                    "avg_response_time": statistics.mean([r.get("response_time", 0) for r in short_successful]) if short_successful else 0
                },
                "long_requests": {
                    "total": len(long_results),
                    "successful": len(long_successful),
                    "success_rate": len(long_successful) / len(long_results) if long_results else 0,
                    "avg_response_time": statistics.mean([r.get("response_time", 0) for r in long_successful]) if long_successful else 0
                },
                "overall_success_rate": len([r for r in mixed_results if r.get("success", False)]) / len(mixed_results) if mixed_results else 0
            }
            
            success = metrics["overall_success_rate"] >= 0.7
            
            return TestResult(
                test_name="mixed_load_patterns",
                start_time=start_time,
                end_time=datetime.now(),
                success=success,
                error_message=None if success else "Mixed load patterns showed conflicts",
                metrics=metrics,
                server_state=self.monitor.get_current_state()
            )
            
        except Exception as e:
            return TestResult(
                test_name="mixed_load_patterns",
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e),
                metrics={},
                server_state=self.monitor.get_current_state()
            )
    
    def _test_resource_exhaustion(self) -> TestResult:
        """Test resource exhaustion scenarios"""
        start_time = datetime.now()
        
        try:
            print("  Testing resource exhaustion scenarios...")
            
            # Monitor initial state
            initial_gpu_memory = self.monitor.get_gpu_memory_usage()
            initial_cpu = self.monitor.get_cpu_usage()
            
            # Test GPU memory pressure
            memory_pressure_results = self.load_generator.generate_memory_pressure_requests(
                count=5,
                context_length=30000  # Large contexts to stress GPU memory
            )
            
            # Test connection limits
            connection_limit_results = self.load_generator.test_connection_limits(
                max_connections=100
            )
            
            # Monitor resource changes
            peak_gpu_memory = self.monitor.get_peak_gpu_memory_usage()
            peak_cpu = self.monitor.get_peak_cpu_usage()
            
            metrics = {
                "initial_gpu_memory": initial_gpu_memory,
                "peak_gpu_memory": peak_gpu_memory,
                "gpu_memory_increase": peak_gpu_memory - initial_gpu_memory,
                "initial_cpu": initial_cpu,
                "peak_cpu": peak_cpu,
                "memory_pressure_tests": {
                    "total_requests": len(memory_pressure_results),
                    "successful_requests": len([r for r in memory_pressure_results if r.get("success", False)]),
                    "success_rate": len([r for r in memory_pressure_results if r.get("success", False)]) / len(memory_pressure_results) if memory_pressure_results else 0
                },
                "connection_limit_test": connection_limit_results
            }
            
            # Success if server handled resource pressure without crashing
            success = (metrics["memory_pressure_tests"]["success_rate"] >= 0.6 and 
                      not connection_limit_results.get("server_crashed", False))
            
            return TestResult(
                test_name="resource_exhaustion",
                start_time=start_time,
                end_time=datetime.now(),
                success=success,
                error_message=None if success else "Resource exhaustion caused server issues",
                metrics=metrics,
                server_state=self.monitor.get_current_state()
            )
            
        except Exception as e:
            return TestResult(
                test_name="resource_exhaustion",
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e),
                metrics={},
                server_state=self.monitor.get_current_state()
            )
    
    def _test_hang_detection_scenarios(self) -> TestResult:
        """Test scenarios that might cause server hangs"""
        start_time = datetime.now()
        
        try:
            print("  Testing hang detection scenarios...")
            
            hang_scenarios = []
            
            # Test extremely long context
            print("    Testing extremely long context...")
            with self.hang_detector.monitor_request("extremely_long_context", timeout=180):
                extremely_long_result = self.load_generator.generate_single_request(
                    content=self.load_generator.generate_long_content(100000),  # 100K characters
                    max_tokens=50,
                    timeout=180
                )
                hang_scenarios.append(("extremely_long_context", extremely_long_result))
            
            # Test malformed requests
            print("    Testing malformed requests...")
            malformed_results = self.load_generator.test_malformed_requests()
            hang_scenarios.append(("malformed_requests", malformed_results))
            
            # Test streaming hangs
            print("    Testing streaming responses...")
            streaming_result = self.load_generator.test_streaming_hangs()
            hang_scenarios.append(("streaming_hangs", streaming_result))
            
            # Test concurrent identical requests (potential deadlock)
            print("    Testing potential deadlock scenario...")
            identical_content = "This is an identical request that might cause issues: " * 100
            deadlock_results = self.load_generator.generate_concurrent_requests(
                count=10,
                content_template=identical_content,
                max_tokens=100
            )
            hang_scenarios.append(("potential_deadlock", {"results": deadlock_results}))
            
            # Analyze hang scenarios
            detected_hangs = self.hang_detector.get_detected_hangs()
            
            metrics = {
                "hang_scenarios_tested": len(hang_scenarios),
                "detected_hangs": len(detected_hangs),
                "hang_details": detected_hangs,
                "scenario_results": {name: result for name, result in hang_scenarios}
            }
            
            # Success if no hangs were detected
            success = len(detected_hangs) == 0
            
            return TestResult(
                test_name="hang_detection_scenarios",
                start_time=start_time,
                end_time=datetime.now(),
                success=success,
                error_message=None if success else f"Detected {len(detected_hangs)} potential hangs",
                metrics=metrics,
                server_state=self.monitor.get_current_state()
            )
            
        except Exception as e:
            return TestResult(
                test_name="hang_detection_scenarios",
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e),
                metrics={},
                server_state=self.monitor.get_current_state()
            )
    
    def _test_recovery_scenarios(self) -> TestResult:
        """Test server recovery from various scenarios"""
        start_time = datetime.now()
        
        if not self.config.enable_recovery_tests:
            return TestResult(
                test_name="recovery_scenarios",
                start_time=start_time,
                end_time=datetime.now(),
                success=True,
                error_message="Recovery tests disabled",
                metrics={"recovery_tests_enabled": False},
                server_state=self.monitor.get_current_state()
            )
        
        try:
            print("  Testing recovery scenarios...")
            
            recovery_tests = []
            
            # Test graceful degradation curve
            print("    Testing graceful degradation curve...")
            degradation_curve = self.load_generator.test_graceful_degradation_curve()
            recovery_tests.append(("degradation_curve", degradation_curve))
            
            # Test memory management recovery
            print("    Testing memory management recovery...")
            memory_recovery = self.load_generator.test_memory_management_recovery()
            recovery_tests.append(("memory_recovery", memory_recovery))
            
            # Test health check recovery
            print("    Testing health check recovery...")
            health_recovery = self.monitor.test_health_recovery()
            recovery_tests.append(("health_recovery", health_recovery))
            
            # Test health monitoring integration
            print("    Testing health monitoring integration...")
            try:
                health_score = self.monitor.get_health_score()
                health_category = self.monitor.get_health_category()
                health_integration = {
                    "health_score": health_score,
                    "health_category": health_category,
                    "health_monitoring_working": isinstance(health_score, (int, float)) and 0 <= health_score <= 1
                }
            except Exception as e:
                health_integration = {"health_monitoring_working": False, "error": str(e)}
            
            recovery_tests.append(("health_integration", health_integration))
            
            # Analyze graceful degradation results
            curve_success = degradation_curve.get("recovery_successful", False)
            memory_success = memory_recovery.get("overall_recovery_successful", False)
            health_working = health_integration.get("health_monitoring_working", False)
            
            # Final server responsiveness check
            final_health = self.monitor.check_server_health()
            server_responsive = final_health.get("healthy", False)
            
            metrics = {
                "recovery_tests": {name: result for name, result in recovery_tests},
                "server_responsive_after_tests": server_responsive,
                "degradation_curve_success": curve_success,
                "memory_recovery_success": memory_success,
                "health_monitoring_working": health_working,
                "max_stable_concurrency": degradation_curve.get("max_stable_concurrency", 0),
                "graceful_behavior_detected": curve_success and memory_success
            }
            
            # Success criteria: server responsive + graceful behavior observed + health monitoring working
            success = server_responsive and (curve_success or memory_success) and health_working
            
            error_message = None
            if not success:
                issues = []
                if not server_responsive:
                    issues.append("server not responsive")
                if not curve_success and not memory_success:
                    issues.append("no graceful behavior observed")
                if not health_working:
                    issues.append("health monitoring not working")
                error_message = f"Graceful degradation test failed: {', '.join(issues)}"
            
            return TestResult(
                test_name="recovery_scenarios",
                start_time=start_time,
                end_time=datetime.now(),
                success=success,
                error_message=error_message,
                metrics=metrics,
                server_state=self.monitor.get_current_state()
            )
            
        except Exception as e:
            return TestResult(
                test_name="recovery_scenarios",
                start_time=start_time,
                end_time=datetime.now(),
                success=False,
                error_message=str(e),
                metrics={},
                server_state=self.monitor.get_current_state()
            )
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate test summary"""
        total_tests = len(self.results)
        successful_tests = len([r for r in self.results if r.success])
        
        # Extract key metrics
        response_times = []
        error_types = {}
        
        for result in self.results:
            if result.success and "response_time" in result.metrics:
                if isinstance(result.metrics["response_time"], (int, float)):
                    response_times.append(result.metrics["response_time"])
            
            if not result.success and result.error_message:
                error_type = result.error_message.split(":")[0]
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        summary = {
            "test_summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": total_tests - successful_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0
            },
            "performance_metrics": {
                "avg_response_time": statistics.mean(response_times) if response_times else 0,
                "max_response_time": max(response_times) if response_times else 0,
                "min_response_time": min(response_times) if response_times else 0,
                "response_time_std": statistics.stdev(response_times) if len(response_times) > 1 else 0
            },
            "error_analysis": error_types,
            "server_health": self.monitor.get_current_state(),
            "detected_issues": self.hang_detector.get_detected_hangs(),
            "test_duration": (datetime.now() - self._test_start_time).total_seconds() if self._test_start_time else 0
        }
        
        return summary


def main():
    """Main function to run stress tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenAI Server Stress Test Suite")
    parser.add_argument("--server-url", default="http://localhost:8000/v1", help="Server URL")
    parser.add_argument("--model-name", default="Qwen3-14B-anger", help="Model name")
    parser.add_argument("--max-concurrent", type=int, default=20, help="Max concurrent requests")
    parser.add_argument("--hang-timeout", type=int, default=300, help="Hang detection timeout (seconds)")
    parser.add_argument("--output-dir", default="stress_test_results", help="Output directory")
    parser.add_argument("--no-gpu-monitoring", action="store_true", help="Disable GPU monitoring")
    parser.add_argument("--no-recovery-tests", action="store_true", help="Disable recovery tests")
    
    args = parser.parse_args()
    
    config = TestConfig(
        server_url=args.server_url,
        model_name=args.model_name,
        max_concurrent_requests=args.max_concurrent,
        hang_timeout=args.hang_timeout,
        output_dir=args.output_dir,
        enable_gpu_monitoring=not args.no_gpu_monitoring,
        enable_recovery_tests=not args.no_recovery_tests
    )
    
    # Create and run stress test suite
    suite = StressTestSuite(config)
    summary = suite.run_all_tests()
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 STRESS TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {summary['test_summary']['total_tests']}")
    print(f"Successful: {summary['test_summary']['successful_tests']}")
    print(f"Failed: {summary['test_summary']['failed_tests']}")
    print(f"Success Rate: {summary['test_summary']['success_rate']:.1%}")
    
    if summary['detected_issues']:
        print(f"\n⚠️ Detected Issues: {len(summary['detected_issues'])}")
        for issue in summary['detected_issues']:
            print(f"  - {issue}")
    
    print(f"\n📊 Average Response Time: {summary['performance_metrics']['avg_response_time']:.2f}s")
    print(f"📈 Max Response Time: {summary['performance_metrics']['max_response_time']:.2f}s")
    
    if summary['error_analysis']:
        print(f"\n🔍 Error Analysis:")
        for error_type, count in summary['error_analysis'].items():
            print(f"  - {error_type}: {count}")
    
    print(f"\n📁 Detailed report saved to: {config.output_dir}/")
    
    return 0 if summary['test_summary']['success_rate'] >= 0.8 else 1


if __name__ == "__main__":
    sys.exit(main())