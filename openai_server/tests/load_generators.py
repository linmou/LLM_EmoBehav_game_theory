#!/usr/bin/env python3
"""
Load Generators for Server Stress Testing

Generates various types of load patterns to test server behavior
under different stress conditions.
"""

import asyncio
import time
import json
import random
import string
import threading
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging


class LoadGeneratorFactory:
    """Factory for generating different types of load patterns"""
    
    def __init__(self, server_url: str, model_name: str):
        self.server_url = server_url
        self.model_name = model_name
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer dummy"
        }
        
        # Setup logging
        self.logger = logging.getLogger("LoadGenerator")
        self.logger.setLevel(logging.INFO)
        
        # Text generation templates
        self.text_templates = {
            "short": "This is a short test message for {purpose}.",
            "medium": "This is a medium-length test message for {purpose}. " * 10,
            "long": "This is a long test message for {purpose}. " * 50,
            "code": '''
def example_function_{num}():
    """
    This is an example function for {purpose}.
    It demonstrates code-like content for testing.
    """
    result = []
    for i in range(100):
        if i % 2 == 0:
            result.append(f"Item {{i}}: even")
        else:
            result.append(f"Item {{i}}: odd")
    return result
''',
            "structured": '''{{
    "test_id": "{num}",
    "purpose": "{purpose}",
    "data": {{
        "items": [
            {{"id": 1, "value": "test_data_1", "timestamp": "2024-01-01T00:00:00Z"}},
            {{"id": 2, "value": "test_data_2", "timestamp": "2024-01-01T00:01:00Z"}},
            {{"id": 3, "value": "test_data_3", "timestamp": "2024-01-01T00:02:00Z"}}
        ],
        "metadata": {{
            "version": "1.0",
            "created_by": "stress_test",
            "description": "Structured test data for {purpose}"
        }}
    }}
}}'''
        }
    
    def _make_request(self, content: str, max_tokens: int = 100, 
                     temperature: float = 0.7, timeout: float = 60.0,
                     stream: bool = False) -> Dict[str, Any]:
        """Make a single API request"""
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.server_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=timeout,
                stream=stream
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                if stream:
                    # Handle streaming response
                    chunks = []
                    for line in response.iter_lines():
                        if line:
                            line_str = line.decode('utf-8')
                            if line_str.startswith("data: "):
                                data_str = line_str[6:]
                                if data_str.strip() != "[DONE]":
                                    try:
                                        chunk_data = json.loads(data_str)
                                        chunks.append(chunk_data)
                                    except json.JSONDecodeError:
                                        continue
                    
                    return {
                        "success": True,
                        "response_time": response_time,
                        "chunks": len(chunks),
                        "streaming": True,
                        "status_code": response.status_code
                    }
                else:
                    # Handle regular response
                    data = response.json()
                    return {
                        "success": True,
                        "response_time": response_time,
                        "response_content": data["choices"][0]["message"]["content"],
                        "tokens_used": data.get("usage", {}).get("total_tokens", 0),
                        "status_code": response.status_code,
                        "streaming": False
                    }
            else:
                return {
                    "success": False,
                    "response_time": response_time,
                    "error": response.text,
                    "status_code": response.status_code
                }
                
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "response_time": timeout,
                "error": "Request timeout",
                "status_code": 0
            }
        except Exception as e:
            return {
                "success": False,
                "response_time": time.time() - start_time,
                "error": str(e),
                "status_code": 0
            }
    
    def generate_long_content(self, length: int, pattern: str = "realistic") -> str:
        """Generate long content of specified length"""
        if pattern == "realistic":
            # Generate realistic text
            sentences = [
                "The rapid advancement of artificial intelligence has transformed many industries.",
                "Machine learning models require extensive training data to achieve optimal performance.",
                "Natural language processing enables computers to understand and generate human language.",
                "Deep learning networks can recognize complex patterns in large datasets.",
                "Computer vision systems are becoming increasingly sophisticated and accurate.",
                "Automated systems can process information much faster than human operators.",
                "The integration of AI into everyday applications continues to expand rapidly.",
                "Data science techniques help organizations make informed business decisions."
            ]
            
            text = []
            current_length = 0
            sentence_idx = 0
            
            while current_length < length:
                sentence = sentences[sentence_idx % len(sentences)]
                if current_length + len(sentence) + 1 <= length:
                    text.append(sentence)
                    current_length += len(sentence) + 1
                else:
                    # Add partial sentence to reach exact length
                    remaining = length - current_length - 1
                    if remaining > 0:
                        text.append(sentence[:remaining])
                    break
                sentence_idx += 1
            
            return " ".join(text)
        
        elif pattern == "repetitive":
            base_text = "This is a repetitive pattern for testing long input handling. "
            repetitions = (length // len(base_text)) + 1
            return (base_text * repetitions)[:length]
        
        elif pattern == "random":
            return ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=length))
        
        else:
            # Default pattern
            return f"Long content for testing ({length} characters): " + "x" * (length - 40)
    
    def generate_single_request(self, content: str, max_tokens: int = 100, 
                              timeout: float = 60.0, **kwargs) -> Dict[str, Any]:
        """Generate a single request"""
        return self._make_request(content, max_tokens, timeout=timeout, **kwargs)
    
    def generate_concurrent_requests(self, count: int, content_template: str,
                                   max_tokens: int = 100, timeout: float = 60.0) -> List[Dict[str, Any]]:
        """Generate multiple concurrent requests"""
        results = []
        
        def make_concurrent_request(index):
            if "{}" in content_template:
                content = content_template.format(index)
            else:
                content = f"{content_template} (request {index})"
            
            return self._make_request(content, max_tokens, timeout=timeout)
        
        # Use ThreadPoolExecutor for true concurrency
        with ThreadPoolExecutor(max_workers=min(count, 20)) as executor:
            future_to_index = {
                executor.submit(make_concurrent_request, i): i 
                for i in range(count)
            }
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    result["request_index"] = index
                    results.append(result)
                except Exception as e:
                    results.append({
                        "success": False,
                        "error": str(e),
                        "request_index": index,
                        "response_time": 0
                    })
        
        # Sort results by request index
        results.sort(key=lambda x: x.get("request_index", 0))
        return results
    
    def generate_burst_requests(self, count: int, content: str, max_tokens: int = 50,
                               delay_between_requests: float = 0.1) -> List[Dict[str, Any]]:
        """Generate burst of requests in quick succession"""
        results = []
        
        for i in range(count):
            result = self._make_request(f"{content} (burst {i})", max_tokens, timeout=70.0)
            result["burst_index"] = i
            results.append(result)
            
            if i < count - 1:  # Don't delay after last request
                time.sleep(delay_between_requests)
        
        return results
    
    def generate_sustained_requests(self, duration: int, requests_per_second: float,
                                  content: str, max_tokens: int = 50) -> List[Dict[str, Any]]:
        """Generate sustained load over time"""
        results = []
        start_time = time.time()
        request_interval = 1.0 / requests_per_second
        request_count = 0
        
        while (time.time() - start_time) < duration:
            request_start = time.time()
            
            result = self._make_request(
                f"{content} (sustained {request_count})", 
                max_tokens, 
                timeout=20.0
            )
            result["sustained_index"] = request_count
            result["elapsed_time"] = time.time() - start_time
            results.append(result)
            
            request_count += 1
            
            # Calculate sleep time to maintain rate
            request_duration = time.time() - request_start
            sleep_time = max(0, request_interval - request_duration)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        return results
    
    def generate_mixed_load(self, short_requests: int, long_requests: int,
                           concurrent_execution: bool = True) -> List[Dict[str, Any]]:
        """Generate mixed load with different request types"""
        all_requests = []
        
        # Prepare request definitions
        short_content = self.text_templates["short"].format(purpose="mixed_load_short")
        long_content = self.generate_long_content(5000, "realistic")
        
        for i in range(short_requests):
            all_requests.append({
                "type": "short",
                "content": short_content + f" (short {i})",
                "max_tokens": 30,
                "timeout": 30.0
            })
        
        for i in range(long_requests):
            all_requests.append({
                "type": "long", 
                "content": f"Analyze this long text (long {i}): {long_content}",
                "max_tokens": 100,
                "timeout": 120.0
            })
        
        # Shuffle requests for mixed execution
        random.shuffle(all_requests)
        
        if concurrent_execution:
            # Execute all requests concurrently
            def execute_request(req_def):
                result = self._make_request(
                    req_def["content"], 
                    req_def["max_tokens"], 
                    timeout=req_def["timeout"]
                )
                result["type"] = req_def["type"]
                return result
            
            results = []
            with ThreadPoolExecutor(max_workers=min(len(all_requests), 15)) as executor:
                future_to_req = {
                    executor.submit(execute_request, req): req 
                    for req in all_requests
                }
                
                for future in as_completed(future_to_req):
                    req = future_to_req[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        results.append({
                            "success": False,
                            "error": str(e),
                            "type": req["type"],
                            "response_time": 0
                        })
        else:
            # Execute requests sequentially
            results = []
            for req_def in all_requests:
                result = self._make_request(
                    req_def["content"], 
                    req_def["max_tokens"], 
                    timeout=req_def["timeout"]
                )
                result["type"] = req_def["type"]
                results.append(result)
                time.sleep(0.5)  # Brief pause between sequential requests
        
        return results
    
    def generate_memory_pressure_requests(self, count: int, context_length: int = 30000) -> List[Dict[str, Any]]:
        """Generate requests designed to create memory pressure"""
        results = []
        
        for i in range(count):
            # Create very long context
            long_content = self.generate_long_content(context_length, "realistic")
            
            result = self._make_request(
                f"Memory pressure test {i}: Please analyze and summarize this extensive text: {long_content}",
                max_tokens=200,
                timeout=180.0
            )
            result["memory_test_index"] = i
            result["input_length"] = len(long_content)
            results.append(result)
            
            # Brief pause to allow garbage collection
            time.sleep(2)
        
        return results
    
    def test_connection_limits(self, max_connections: int = 100) -> Dict[str, Any]:
        """Test server connection limits"""
        successful_connections = 0
        failed_connections = 0
        connection_errors = []
        
        def test_single_connection(conn_id):
            try:
                result = self._make_request(f"Connection test {conn_id}", max_tokens=10, timeout=70.0)
                return {"connection_id": conn_id, "success": result["success"], "error": result.get("error")}
            except Exception as e:
                return {"connection_id": conn_id, "success": False, "error": str(e)}
        
        # Test connections in batches to avoid overwhelming the system
        batch_size = 20
        results = []
        
        for batch_start in range(0, max_connections, batch_size):
            batch_end = min(batch_start + batch_size, max_connections)
            batch_futures = []
            
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                for conn_id in range(batch_start, batch_end):
                    future = executor.submit(test_single_connection, conn_id)
                    batch_futures.append(future)
                
                for future in as_completed(batch_futures):
                    result = future.result()
                    results.append(result)
                    
                    if result["success"]:
                        successful_connections += 1
                    else:
                        failed_connections += 1
                        connection_errors.append(result["error"])
            
            # Brief pause between batches
            time.sleep(1)
            
            # Stop early if we're seeing consistent failures
            if failed_connections > successful_connections and len(results) > 20:
                break
        
        return {
            "total_attempts": len(results),
            "successful_connections": successful_connections,
            "failed_connections": failed_connections,
            "success_rate": successful_connections / len(results) if results else 0,
            "connection_errors": connection_errors,
            "max_concurrent_achieved": successful_connections,
            "server_crashed": False  # Would need additional detection logic
        }
    
    def test_malformed_requests(self) -> Dict[str, Any]:
        """Test server behavior with malformed requests"""
        malformed_tests = [
            # Missing required fields
            {"messages": [{"role": "user", "content": "test"}]},  # No model
            {"model": self.model_name},  # No messages
            
            # Invalid field values
            {"model": "nonexistent_model", "messages": [{"role": "user", "content": "test"}]},
            {"model": self.model_name, "messages": [{"role": "invalid_role", "content": "test"}]},
            {"model": self.model_name, "messages": [{"role": "user", "content": "test"}], "max_tokens": -1},
            {"model": self.model_name, "messages": [{"role": "user", "content": "test"}], "temperature": 5.0},
            
            # Malformed JSON would be handled by requests library, so we test invalid structures
            {"model": self.model_name, "messages": "not_a_list"},
            {"model": self.model_name, "messages": [{"content": "missing_role"}]},
            {"model": self.model_name, "messages": [{"role": "user"}]},  # Missing content
        ]
        
        results = []
        for i, malformed_payload in enumerate(malformed_tests):
            try:
                response = requests.post(
                    f"{self.server_url}/chat/completions",
                    headers=self.headers,
                    json=malformed_payload,
                    timeout=70.0
                )
                
                results.append({
                    "test_index": i,
                    "status_code": response.status_code,
                    "expected_error": True,
                    "got_error": response.status_code != 200,
                    "response_time": 0,  # Quick error responses
                    "payload": malformed_payload
                })
                
            except Exception as e:
                results.append({
                    "test_index": i,
                    "status_code": 0,
                    "expected_error": True,
                    "got_error": True,
                    "error": str(e),
                    "payload": malformed_payload
                })
        
        # Analyze results
        expected_errors = len([r for r in results if r["got_error"]])
        
        return {
            "total_malformed_tests": len(results),
            "expected_errors_received": expected_errors,
            "error_handling_rate": expected_errors / len(results) if results else 0,
            "results": results,
            "server_handled_gracefully": expected_errors == len(results)
        }
    
    def test_streaming_hangs(self) -> Dict[str, Any]:
        """Test for streaming response hangs"""
        try:
            content = self.generate_long_content(2000, "realistic")
            
            start_time = time.time()
            result = self._make_request(
                f"Please provide a detailed streaming response about: {content}",
                max_tokens=200,
                timeout=60.0,
                stream=True
            )
            end_time = time.time()
            
            if result["success"] and result.get("streaming"):
                return {
                    "streaming_test_success": True,
                    "total_time": end_time - start_time,
                    "chunks_received": result.get("chunks", 0),
                    "hang_detected": False
                }
            else:
                return {
                    "streaming_test_success": False,
                    "error": result.get("error", "Unknown streaming error"),
                    "hang_detected": "timeout" in str(result.get("error", "")).lower()
                }
                
        except Exception as e:
            return {
                "streaming_test_success": False,
                "error": str(e),
                "hang_detected": "timeout" in str(e).lower()
            }
    
    def test_graceful_degradation_curve(self) -> Dict[str, Any]:
        """Test server behavior as load increases gradually - find limits without breaking"""
        results = []
        max_stable_concurrency = 1
        
        # Gradually increase concurrent requests to find server capacity
        for concurrency in [1, 2, 3, 5, 7, 10, 12, 15, 18, 20, 25, 30]:
            print(f"    Testing {concurrency} concurrent requests...")
            
            test_result = self.generate_concurrent_requests(
                count=concurrency,
                content_template="Load test {}: moderate length message for testing server capacity under increasing concurrent load.",
                max_tokens=100,
                timeout=60.0  # Reasonable timeout
            )
            
            success_rate = len([r for r in test_result if r.get("success")]) / len(test_result)
            successful_requests = [r for r in test_result if r.get("success")]
            avg_response_time = (sum(r.get("response_time", 0) for r in successful_requests) / 
                               max(1, len(successful_requests)))
            max_response_time = max([r.get("response_time", 0) for r in successful_requests], default=0)
            
            # Check server health after this load
            time.sleep(2)  # Brief pause
            health_check = self._make_request("Health check", max_tokens=5, timeout=70.0)
            server_responsive = health_check.get("success", False)
            
            result = {
                "concurrency": concurrency,
                "success_rate": success_rate,
                "avg_response_time": avg_response_time,
                "max_response_time": max_response_time,
                "server_responsive_after": server_responsive,
                "total_requests": len(test_result),
                "successful_requests": len(successful_requests)
            }
            results.append(result)
            
            # Stop if server starts showing significant degradation (but not failure)
            if success_rate < 0.7 or avg_response_time > 30.0 or not server_responsive:
                print(f"    Server capacity limit reached at {concurrency} concurrent requests")
                print(f"    Success rate: {success_rate:.1%}, Avg response time: {avg_response_time:.1f}s")
                break
            else:
                max_stable_concurrency = concurrency
            
            time.sleep(3)  # Recovery time between tests
        
        # Verify server recovers after load test
        time.sleep(10)
        recovery_result = self._make_request("Recovery verification test", max_tokens=20, timeout=70.0)
        
        return {
            "degradation_curve": results,
            "max_stable_concurrency": max_stable_concurrency,
            "recovery_successful": recovery_result.get("success", False),
            "recovery_response_time": recovery_result.get("response_time", 0),
            "test_type": "graceful_degradation_curve"
        }
    
    def test_memory_management_recovery(self) -> Dict[str, Any]:
        """Test server memory management and recovery patterns"""
        initial_memory_check = self._make_request("Initial memory check", max_tokens=10, timeout=70.0)
        
        memory_pressure_results = []
        
        # Generate controlled memory pressure (not destructive)
        for i in range(3):
            # Use realistic long content (not random garbage)
            long_content = self.generate_long_content(15000, "realistic")
            
            memory_test = self._make_request(
                f"Memory test {i}: Please analyze and summarize this document: {long_content}",
                max_tokens=150,
                timeout=90.0  # Reasonable timeout
            )
            memory_pressure_results.append(memory_test)
            
            # Check server responsiveness after each memory-intensive request
            health_check = self._make_request(f"Health check after memory test {i}", max_tokens=5, timeout=15.0)
            memory_test["health_check_after"] = health_check.get("success", False)
            
            # Allow time for cleanup between requests
            time.sleep(8)
        
        # Test recovery pattern: burst -> pause -> normal
        print("    Testing burst -> pause -> recovery pattern...")
        
        # Small burst of requests
        burst_results = self.generate_burst_requests(
            count=5,
            content="Burst test for memory recovery",
            max_tokens=100,
            delay_between_requests=0.5
        )
        
        # Pause for cleanup
        time.sleep(15)
        
        # Test normal operation recovery
        recovery_tests = []
        for i in range(3):
            recovery_result = self._make_request(
                f"Normal recovery test {i}: Simple operation",
                max_tokens=50,
                timeout=30.0
            )
            recovery_tests.append(recovery_result)
            time.sleep(3)
        
        # Analyze results
        memory_tests_successful = sum(1 for r in memory_pressure_results if r.get("success", False))
        burst_successful = sum(1 for r in burst_results if r.get("success", False))
        recovery_successful = sum(1 for r in recovery_tests if r.get("success", False))
        
        return {
            "initial_server_responsive": initial_memory_check.get("success", False),
            "memory_pressure_tests": {
                "total": len(memory_pressure_results),
                "successful": memory_tests_successful,
                "success_rate": memory_tests_successful / len(memory_pressure_results),
                "all_health_checks_passed": all(r.get("health_check_after", False) for r in memory_pressure_results)
            },
            "burst_pattern": {
                "total": len(burst_results),
                "successful": burst_successful,
                "success_rate": burst_successful / len(burst_results)
            },
            "recovery_verification": {
                "total": len(recovery_tests),
                "successful": recovery_successful,
                "success_rate": recovery_successful / len(recovery_tests)
            },
            "overall_recovery_successful": recovery_successful >= 2,  # At least 2/3 recovery tests should pass
            "test_type": "memory_management_recovery"
        }


def main():
    """Test the load generators"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load Generator Test")
    parser.add_argument("--server-url", default="http://localhost:8000/v1", help="Server URL")
    parser.add_argument("--model-name", default="Qwen3-14B-anger", help="Model name")
    parser.add_argument("--test-type", choices=["single", "concurrent", "burst", "mixed"], 
                       default="single", help="Type of test to run")
    
    args = parser.parse_args()
    
    generator = LoadGeneratorFactory(args.server_url, args.model_name)
    
    print(f"🔧 Testing {args.test_type} load generation...")
    
    if args.test_type == "single":
        result = generator.generate_single_request("Test message for single request")
        print(f"Single request result: {result}")
    
    elif args.test_type == "concurrent":
        results = generator.generate_concurrent_requests(5, "Concurrent test message {}")
        successful = len([r for r in results if r.get("success", False)])
        print(f"Concurrent requests: {successful}/{len(results)} successful")
    
    elif args.test_type == "burst":
        results = generator.generate_burst_requests(10, "Burst test", delay_between_requests=0.1)
        successful = len([r for r in results if r.get("success", False)])
        print(f"Burst requests: {successful}/{len(results)} successful")
    
    elif args.test_type == "mixed":
        results = generator.generate_mixed_load(5, 2, concurrent_execution=True)
        successful = len([r for r in results if r.get("success", False)])
        print(f"Mixed load: {successful}/{len(results)} successful")
    
    print("Load generation test completed!")


if __name__ == "__main__":
    main()