#!/usr/bin/env python3
"""
Simple demonstration of the batching effectiveness in the optimized OpenAI server.

This script compares the behavior of individual requests vs burst requests
to show how batching reduces latency variance and improves throughput.
"""

import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI


def make_single_request(client, prompt="What is 2+2?"):
    """Make a single request and measure timing."""
    start = time.time()
    try:
        response = client.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.0,
        )
        end = time.time()
        return {
            "success": True,
            "latency": end - start,
            "response": response.choices[0].message.content.strip(),
        }
    except Exception as e:
        return {"success": False, "latency": -1, "error": str(e)}


def test_individual_requests(client, num_requests=8):
    """Test requests sent one by one."""
    print("Testing Individual Requests (Sequential)")
    print("-" * 40)

    results = []
    start_time = time.time()

    for i in range(num_requests):
        result = make_single_request(client, f"Count to {i+1}")
        results.append(result)
        print(f"Request {i+1}: {result['latency']:.3f}s")

    end_time = time.time()
    total_time = end_time - start_time

    successful = [r for r in results if r["success"]]
    if successful:
        latencies = [r["latency"] for r in successful]
        print(f"\nResults:")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average latency: {sum(latencies)/len(latencies):.3f}s")
        print(f"  Latency range: {min(latencies):.3f}s - {max(latencies):.3f}s")
        print(f"  Throughput: {len(successful)/total_time:.2f} req/s")

    return results


def test_burst_requests(client, num_requests=8):
    """Test requests sent simultaneously (should trigger batching)."""
    print("\nTesting Burst Requests (Simultaneous)")
    print("-" * 40)

    start_time = time.time()

    # Send all requests simultaneously
    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [
            executor.submit(make_single_request, client, f"Count to {i+1}")
            for i in range(num_requests)
        ]
        results = [future.result() for future in futures]

    end_time = time.time()
    total_time = end_time - start_time

    successful = [r for r in results if r["success"]]
    if successful:
        latencies = [r["latency"] for r in successful]
        print(f"Burst Results:")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average latency: {sum(latencies)/len(latencies):.3f}s")
        print(f"  Latency range: {min(latencies):.3f}s - {max(latencies):.3f}s")
        print(f"  Latency variance: {max(latencies) - min(latencies):.3f}s")
        print(f"  Throughput: {len(successful)/total_time:.2f} req/s")

        # Analyze latency variance (batching indicator)
        variance = max(latencies) - min(latencies)
        if variance < 0.1:  # Less than 100ms variance
            print(f"  🎯 Low variance suggests effective batching!")
        else:
            print(f"  ⚠️  High variance may indicate sequential processing")

    return results


def main():
    """Run the batching demonstration."""
    print("=" * 60)
    print("OpenAI Server Batching Demonstration")
    print("=" * 60)

    # Connect to server
    client = OpenAI(api_key="dummy", base_url="http://localhost:8000/v1")

    # Test server connectivity
    try:
        response = client.chat.completions.create(
            model="test-model", messages=[{"role": "user", "content": "Hello"}], max_tokens=5
        )
        print("✅ Server connected successfully")
    except Exception as e:
        print(f"❌ Server connection failed: {e}")
        print("Make sure the server is running: python -m openai_server ...")
        return

    # Run tests
    print(f"\nTesting with {8} requests to demonstrate batching...")

    # Test 1: Individual requests
    individual_results = test_individual_requests(client, 8)

    # Test 2: Burst requests (should be batched)
    burst_results = test_burst_requests(client, 8)

    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON ANALYSIS")
    print("=" * 60)

    ind_successful = [r for r in individual_results if r["success"]]
    burst_successful = [r for r in burst_results if r["success"]]

    if ind_successful and burst_successful:
        ind_latencies = [r["latency"] for r in ind_successful]
        burst_latencies = [r["latency"] for r in burst_successful]

        ind_avg = sum(ind_latencies) / len(ind_latencies)
        burst_avg = sum(burst_latencies) / len(burst_latencies)

        ind_variance = max(ind_latencies) - min(ind_latencies)
        burst_variance = max(burst_latencies) - min(burst_latencies)

        print(f"Individual Requests:")
        print(f"  Average latency: {ind_avg:.3f}s")
        print(f"  Latency variance: {ind_variance:.3f}s")

        print(f"\nBurst Requests (Batched):")
        print(f"  Average latency: {burst_avg:.3f}s")
        print(f"  Latency variance: {burst_variance:.3f}s")

        print(f"\nBatching Benefits:")
        if burst_avg < ind_avg:
            improvement = ((ind_avg - burst_avg) / ind_avg) * 100
            print(f"  ⚡ {improvement:.1f}% faster average latency")

        if burst_variance < ind_variance:
            consistency = ((ind_variance - burst_variance) / ind_variance) * 100
            print(f"  🎯 {consistency:.1f}% more consistent latency")

        if burst_variance < 0.1:
            print(f"  ✅ Effective batching detected (low variance)")

    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
