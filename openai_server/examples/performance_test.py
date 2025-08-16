#!/usr/bin/env python3
"""
Performance Testing Script for OpenAI Server Optimizations

This script tests the performance improvements made to the openai_server
by comparing throughput and latency under different configurations.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import requests
from openai import OpenAI


def test_single_request(client: OpenAI, prompt: str = "What is 2+2?") -> float:
    """Test a single request and return response time."""
    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.0,
        )
        end_time = time.time()
        return end_time - start_time
    except Exception as e:
        print(f"Request failed: {e}")
        return -1


def test_concurrent_requests(base_url: str, num_requests: int = 10, num_workers: int = 5) -> dict:
    """Test multiple concurrent requests and measure performance."""
    client = OpenAI(api_key="dummy", base_url=base_url)

    # Test server connectivity first
    try:
        response = requests.get(f"{base_url.rstrip('/v1')}/health", timeout=5)
        if response.status_code != 200:
            return {"error": f"Server health check failed: {response.status_code}"}
    except Exception as e:
        return {"error": f"Server not reachable: {e}"}

    print(f"Testing {num_requests} concurrent requests with {num_workers} workers...")

    prompts = [
        "What is 2+2?",
        "Explain quantum computing in one sentence.",
        "Write a haiku about programming.",
        "What is the capital of France?",
        "Describe machine learning briefly.",
    ] * (num_requests // 5 + 1)

    start_time = time.time()
    response_times = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(test_single_request, client, prompts[i % len(prompts)])
            for i in range(num_requests)
        ]

        for future in futures:
            response_time = future.result()
            if response_time > 0:
                response_times.append(response_time)

    end_time = time.time()
    total_time = end_time - start_time

    if not response_times:
        return {"error": "All requests failed"}

    return {
        "total_requests": num_requests,
        "successful_requests": len(response_times),
        "failed_requests": num_requests - len(response_times),
        "total_time": total_time,
        "throughput": len(response_times) / total_time,  # requests per second
        "avg_latency": sum(response_times) / len(response_times),
        "min_latency": min(response_times),
        "max_latency": max(response_times),
        "p95_latency": sorted(response_times)[int(0.95 * len(response_times))],
        "concurrent_workers": num_workers,
    }


def run_performance_comparison():
    """Run performance comparison tests."""
    print("=" * 60)
    print("OpenAI Server Performance Test")
    print("=" * 60)

    base_url = "http://localhost:8000/v1"

    # Test configurations
    test_configs = [
        {"name": "Batch Processing (Optimized)", "requests": 20, "workers": 10},
        {"name": "Sequential Processing", "requests": 10, "workers": 1},
        {"name": "Low Concurrency", "requests": 15, "workers": 3},
        {"name": "High Concurrency", "requests": 30, "workers": 15},
    ]

    results = []

    for config in test_configs:
        print(f"\nTesting: {config['name']}")
        print("-" * 40)

        result = test_concurrent_requests(
            base_url=base_url, num_requests=config["requests"], num_workers=config["workers"]
        )

        if "error" in result:
            print(f"❌ Test failed: {result['error']}")
            continue

        results.append({**result, "config_name": config["name"]})

        print(f"✅ Successful requests: {result['successful_requests']}/{result['total_requests']}")
        print(f"📊 Throughput: {result['throughput']:.2f} req/s")
        print(f"⏱️  Average latency: {result['avg_latency']:.3f}s")
        print(f"📈 P95 latency: {result['p95_latency']:.3f}s")
        print(f"🔄 Workers: {result['concurrent_workers']}")

    # Summary
    if results:
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)

        best_throughput = max(results, key=lambda x: x["throughput"])
        best_latency = min(results, key=lambda x: x["avg_latency"])

        print(f"🏆 Best Throughput: {best_throughput['config_name']}")
        print(f"   └─ {best_throughput['throughput']:.2f} req/s")
        print(f"⚡ Best Latency: {best_latency['config_name']}")
        print(f"   └─ {best_latency['avg_latency']:.3f}s average")

        print("\n📋 Detailed Results:")
        for result in results:
            print(
                f"   {result['config_name']:25} | "
                f"{result['throughput']:6.2f} req/s | "
                f"{result['avg_latency']:6.3f}s avg | "
                f"{result['p95_latency']:6.3f}s p95"
            )


def test_batching_effectiveness():
    """Test specifically for batching effectiveness."""
    print("\n" + "=" * 60)
    print("BATCHING EFFECTIVENESS TEST")
    print("=" * 60)

    base_url = "http://localhost:8000/v1"
    client = OpenAI(api_key="dummy", base_url=base_url)

    # Test rapid fire requests (should benefit from batching)
    print("Testing rapid-fire requests (should trigger batching)...")

    async def async_request():
        """Make an async request."""
        start = time.time()
        try:
            response = client.chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "Count to 5"}],
                max_tokens=20,
                temperature=0.0,
            )
            return time.time() - start
        except Exception as e:
            print(f"Async request failed: {e}")
            return -1

    # Test burst of simultaneous requests
    async def burst_test():
        tasks = [async_request() for _ in range(8)]  # Same as batch size
        return await asyncio.gather(*tasks)

    print("Sending burst of 8 simultaneous requests...")
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response_times = loop.run_until_complete(burst_test())
        loop.close()

        valid_times = [t for t in response_times if t > 0]
        if valid_times:
            print(f"✅ Burst test completed:")
            print(f"   📊 {len(valid_times)}/8 requests successful")
            print(f"   ⏱️  Average time: {sum(valid_times)/len(valid_times):.3f}s")
            print(f"   📈 Time range: {min(valid_times):.3f}s - {max(valid_times):.3f}s")

            # Check if batching worked (similar response times)
            time_variance = max(valid_times) - min(valid_times)
            if time_variance < 0.1:  # Less than 100ms variance suggests batching
                print("   🎯 Low variance suggests effective batching!")
            else:
                print("   ⚠️  High variance may indicate sequential processing")
        else:
            print("❌ All burst requests failed")

    except Exception as e:
        print(f"❌ Burst test failed: {e}")


if __name__ == "__main__":
    try:
        run_performance_comparison()
        test_batching_effectiveness()

        print("\n" + "=" * 60)
        print("Test completed! To run with different server configurations:")
        print("  1. Start optimized server: make server")
        print("  2. Start fast server: make server-fast")
        print("  3. Start no-batch server: make server-no-batch")
        print("  4. Run this test: python openai_server/examples/performance_test.py")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
