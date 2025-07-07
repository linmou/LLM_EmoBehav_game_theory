#!/usr/bin/env python3
"""
Sanity check script for testing neural emotion activation with PromptExperiment.
This script runs a performance test with 100 scenarios and repeat=2 to measure throughput.
"""

import logging
import os
import sys
import time

# Add the parent directory to the path to import prompt_experiment
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from prompt_experiment import PromptExperiment

# Configure logging for sanity check
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_neutral_performance_test():
    """Test neutral server configuration with 100 scenarios."""
    logger.info("=" * 60)
    logger.info("PERFORMANCE TEST: Testing Neutral Server (100 scenarios, repeat=2)")
    logger.info("=" * 60)

    try:
        engine = PromptExperiment(
            "config/priDeli_neural_test_config.yaml", experiment_id="performance_test_neutral"
        )

        # Record start time
        start_time = time.time()
        logger.info(f"🕐 Starting performance test at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Run performance test with 100 scenarios and repeat=2
        engine.run_sanity_check(max_scenarios=100, max_repeat=2)

        # Record end time and calculate duration
        end_time = time.time()
        duration = end_time - start_time
        total_requests = 100 * 2  # scenarios * repeat
        throughput = total_requests / duration

        logger.info(f"🕐 Performance test completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"⏱️  Total duration: {duration:.2f} seconds")
        logger.info(f"📊 Total requests: {total_requests}")
        logger.info(f"🚀 Throughput: {throughput:.2f} requests/second")
        logger.info(f"⚡ Average response time: {duration/total_requests:.3f} seconds/request")
        logger.info("✅ Neutral server performance test COMPLETED")

        return {
            "success": True,
            "duration": duration,
            "total_requests": total_requests,
            "throughput": throughput,
            "avg_response_time": duration / total_requests,
        }

    except ConnectionError as e:
        logger.error(f"❌ Neutral server performance test FAILED:\n{e}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"❌ Neutral server performance test FAILED: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return {"success": False, "error": str(e)}


def run_anger_performance_test():
    """Test anger-activated server configuration with 100 scenarios."""
    logger.info("=" * 60)
    logger.info("PERFORMANCE TEST: Testing Anger-Activated Server (100 scenarios, repeat=2)")
    logger.info("=" * 60)

    try:
        engine = PromptExperiment(
            "config/priDeli_neural_anger_test_config.yaml", experiment_id="performance_test_anger"
        )

        # Record start time
        start_time = time.time()
        logger.info(f"🕐 Starting performance test at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Run performance test with 100 scenarios and repeat=2
        engine.run_sanity_check(max_scenarios=100, max_repeat=2)

        # Record end time and calculate duration
        end_time = time.time()
        duration = end_time - start_time
        total_requests = 100 * 2  # scenarios * repeat
        throughput = total_requests / duration

        logger.info(f"🕐 Performance test completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"⏱️  Total duration: {duration:.2f} seconds")
        logger.info(f"📊 Total requests: {total_requests}")
        logger.info(f"🚀 Throughput: {throughput:.2f} requests/second")
        logger.info(f"⚡ Average response time: {duration/total_requests:.3f} seconds/request")
        logger.info("✅ Anger server performance test COMPLETED")

        return {
            "success": True,
            "duration": duration,
            "total_requests": total_requests,
            "throughput": throughput,
            "avg_response_time": duration / total_requests,
        }

    except ConnectionError as e:
        logger.error(f"❌ Anger server performance test FAILED:\n{e}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"❌ Anger server performance test FAILED: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return {"success": False, "error": str(e)}


def main():
    """Run complete performance test for both configurations."""
    logger.info("Starting Neural Emotion Activation Performance Test")
    logger.info("Testing with 100 scenarios, repeat=2 (200 total requests per server)")
    logger.info("")
    logger.info("Make sure both servers are running:")
    logger.info("  Neutral: python -m openai_server --model <model_path> --port 8000")
    logger.info(
        "  Anger:   python -m openai_server --model <model_path> --emotion anger --port 8001"
    )
    logger.info("")

    # Test both configurations
    neutral_result = run_neutral_performance_test()
    anger_result = run_anger_performance_test()

    # Summary
    logger.info("=" * 80)
    logger.info("PERFORMANCE TEST SUMMARY")
    logger.info("=" * 80)

    if neutral_result["success"]:
        logger.info(f"✅ Neutral Server Performance:")
        logger.info(f"   Duration: {neutral_result['duration']:.2f} seconds")
        logger.info(f"   Throughput: {neutral_result['throughput']:.2f} req/s")
        logger.info(f"   Avg Response Time: {neutral_result['avg_response_time']:.3f} s/req")
    else:
        logger.info(f"❌ Neutral Server: FAILED - {neutral_result.get('error', 'Unknown error')}")

    if anger_result["success"]:
        logger.info(f"✅ Anger Server Performance:")
        logger.info(f"   Duration: {anger_result['duration']:.2f} seconds")
        logger.info(f"   Throughput: {anger_result['throughput']:.2f} req/s")
        logger.info(f"   Avg Response Time: {anger_result['avg_response_time']:.3f} s/req")
    else:
        logger.info(f"❌ Anger Server: FAILED - {anger_result.get('error', 'Unknown error')}")

    # Performance comparison
    if neutral_result["success"] and anger_result["success"]:
        logger.info("")
        logger.info("🔍 Performance Comparison:")
        avg_throughput = (neutral_result["throughput"] + anger_result["throughput"]) / 2
        avg_response_time = (
            neutral_result["avg_response_time"] + anger_result["avg_response_time"]
        ) / 2
        logger.info(f"   Average Throughput: {avg_throughput:.2f} req/s")
        logger.info(f"   Average Response Time: {avg_response_time:.3f} s/req")
        logger.info("")
        logger.info("🎉 Both servers completed performance test successfully!")
        logger.info(
            f"📊 Total requests processed: {neutral_result['total_requests'] + anger_result['total_requests']}"
        )

        # Calculate total time for all requests
        total_time = neutral_result["duration"] + anger_result["duration"]
        total_requests = neutral_result["total_requests"] + anger_result["total_requests"]
        overall_throughput = total_requests / total_time
        logger.info(f"🚀 Overall throughput: {overall_throughput:.2f} req/s")
    else:
        logger.error("❌ Some performance tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
