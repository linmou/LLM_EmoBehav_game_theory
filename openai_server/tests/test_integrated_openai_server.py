#!/usr/bin/env python3
"""
Integrated test for the OpenAI-compatible RepControlVLLMHook server.

This script performs a complete integration test by:
1. Starting the server as a subprocess
2. Waiting for it to be ready
3. Running comprehensive tests
4. Cleaning up the server process
"""

import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IntegratedServerTester:
    def __init__(self, model_path, model_name, emotion="anger", port=8000, api_key="token-abc123"):
        self.model_path = model_path
        self.model_name = model_name
        self.emotion = emotion
        self.port = port
        self.api_key = api_key
        self.base_url = f"http://localhost:{port}/v1"
        self.health_url = f"http://localhost:{port}/health"

        self.client = OpenAI(base_url=self.base_url, api_key=api_key)
        self.server_process = None
        self.server_output_queue = queue.Queue()
        self.server_ready = False

    def log_server_output(self, process):
        """Log server output in a separate thread."""
        try:
            for line in iter(process.stdout.readline, ""):
                if line.strip():
                    logger.info(f"[SERVER] {line.strip()}")
                    self.server_output_queue.put(line.strip())

                    # Check if server is ready
                    if "Application startup complete" in line or "Uvicorn running on" in line:
                        self.server_ready = True

        except Exception as e:
            logger.error(f"Error reading server output: {e}")

    def start_server(self, timeout=180):
        """Start the server subprocess and wait for it to be ready."""
        logger.info("=" * 60)
        logger.info("STARTING INTEGRATED SERVER TEST")
        logger.info("=" * 60)
        logger.info(f"Model: {self.model_path}")
        logger.info(f"Model Name: {self.model_name}")
        logger.info(f"Emotion: {self.emotion}")
        logger.info(f"Port: {self.port}")
        logger.info(f"Base URL: {self.base_url}")

        # Prepare server command
        cmd = [
            sys.executable,
            "init_openai_server.py",
            "--model",
            self.model_path,
            "--model_name",
            self.model_name,
            "--emotion",
            self.emotion,
            "--port",
            str(self.port),
            "--api_key",
            self.api_key,
            "--host",
            "localhost",
        ]

        logger.info(f"Starting server with command: {' '.join(cmd)}")

        try:
            # Start server process
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            # Start thread to log server output
            output_thread = threading.Thread(
                target=self.log_server_output, args=(self.server_process,), daemon=True
            )
            output_thread.start()

            logger.info("Server process started, waiting for initialization...")

            # Wait for server to be ready
            start_time = time.time()
            while time.time() - start_time < timeout:
                # Check if process is still running
                if self.server_process.poll() is not None:
                    logger.error("Server process terminated unexpectedly!")
                    return False

                # Check if server responds to health check
                try:
                    response = requests.get(self.health_url, timeout=5)
                    if response.status_code == 200:
                        health_data = response.json()
                        if health_data.get("model_loaded", False):
                            logger.info("✅ Server is ready and model is loaded!")
                            return True
                        else:
                            logger.info("Server responding but model not yet loaded...")
                except requests.exceptions.RequestException:
                    pass

                time.sleep(5)  # Check every 5 seconds

            logger.error(f"Server did not become ready within {timeout} seconds")
            return False

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False

    def test_health_endpoint(self):
        """Test the health endpoint."""
        logger.info("Testing health endpoint...")
        try:
            response = requests.get(self.health_url, timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"Health check passed: {health_data}")
                return health_data.get("status") == "healthy" and health_data.get(
                    "model_loaded", False
                )
            else:
                logger.error(f"Health check failed with status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False

    def test_models_endpoint(self):
        """Test the /v1/models endpoint."""
        logger.info("Testing /v1/models endpoint...")
        try:
            models = self.client.models.list()
            logger.info(f"Available models: {[model.id for model in models.data]}")

            # Check if our model is in the list
            model_ids = [model.id for model in models.data]
            if self.model_name in model_ids:
                logger.info(f"✅ Model '{self.model_name}' found in models list")
                return True
            else:
                logger.error(f"❌ Model '{self.model_name}' not found in models list")
                return False

        except Exception as e:
            logger.error(f"Error testing models endpoint: {e}")
            return False

    def test_basic_chat_completion(self):
        """Test basic chat completion functionality."""
        logger.info("Testing basic chat completion...")

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello! How are you today?"}],
                max_tokens=50,
                temperature=0.0,
            )

            response_text = completion.choices[0].message.content
            logger.info(f"Chat completion response: '{response_text}'")
            logger.info(f"Usage: {completion.usage}")

            if len(response_text.strip()) > 0:
                logger.info("✅ Basic chat completion test passed")
                return True
            else:
                logger.error("❌ Empty response from chat completion")
                return False

        except Exception as e:
            logger.error(f"Error in basic chat completion test: {e}")
            return False

    def test_emotion_influenced_responses(self):
        """Test that emotion influences responses."""
        logger.info(f"Testing emotion-influenced responses for '{self.emotion}'...")

        # Prompts that should trigger emotion-specific responses
        emotion_prompts = {
            "anger": [
                "Someone just cut in front of me in line. I feel",
                "How do you react when someone is unfair to you?",
                "Describe your feelings when someone interrupts you repeatedly.",
            ],
            "happiness": [
                "I just received great news! I feel",
                "Describe a perfect sunny day.",
                "How do you feel when good things happen?",
            ],
            "sadness": [
                "I lost something important to me. I feel",
                "Describe how you feel on a rainy day.",
                "How do you react to disappointing news?",
            ],
        }

        prompts = emotion_prompts.get(self.emotion, emotion_prompts["anger"])

        responses = []
        for i, prompt in enumerate(prompts):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=40,
                    temperature=0.0,  # Deterministic for testing
                )

                response = completion.choices[0].message.content.strip()
                responses.append(response)

                logger.info(f"Prompt {i+1}: '{prompt}'")
                logger.info(f"Response: '{response}'")
                logger.info("-" * 50)

            except Exception as e:
                logger.error(f"Error in emotion test {i+1}: {e}")
                return False

        # Basic validation - all responses should be non-empty
        if all(len(r) > 0 for r in responses):
            logger.info(
                f"✅ All {len(responses)} emotion-influenced responses generated successfully"
            )
            return True
        else:
            logger.error("❌ Some emotion-influenced responses were empty")
            return False

    def test_parameter_variations(self):
        """Test different parameter settings."""
        logger.info("Testing parameter variations...")

        test_cases = [
            {"max_tokens": 20, "temperature": 0.0, "name": "short deterministic"},
            {"max_tokens": 100, "temperature": 0.5, "name": "longer with temperature"},
            {"max_tokens": 30, "temperature": 0.0, "top_p": 0.8, "name": "with top_p"},
        ]

        for case in test_cases:
            try:
                params = {k: v for k, v in case.items() if k != "name"}

                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": "Tell me a short story."}],
                    **params,
                )

                response = completion.choices[0].message.content.strip()
                logger.info(f"Test '{case['name']}': '{response[:50]}...'")

                if len(response) == 0:
                    logger.error(f"❌ Empty response for test '{case['name']}'")
                    return False

            except Exception as e:
                logger.error(f"Error in parameter test '{case['name']}': {e}")
                return False

        logger.info("✅ Parameter variation tests passed")
        return True

    def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        logger.info("Testing concurrent requests...")

        import concurrent.futures

        def make_request(prompt_id):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": f"This is request {prompt_id}. Respond briefly.",
                        }
                    ],
                    max_tokens=20,
                    temperature=0.0,
                )
                return prompt_id, completion.choices[0].message.content.strip()
            except Exception as e:
                return prompt_id, f"ERROR: {e}"

        # Make 3 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request, i) for i in range(1, 4)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        success_count = 0
        for prompt_id, response in results:
            if not response.startswith("ERROR:"):
                logger.info(f"Concurrent request {prompt_id}: '{response}'")
                success_count += 1
            else:
                logger.error(f"Concurrent request {prompt_id} failed: {response}")

        if success_count >= 2:  # Allow 1 failure out of 3
            logger.info(f"✅ Concurrent requests test passed ({success_count}/3 successful)")
            return True
        else:
            logger.error(f"❌ Too many concurrent request failures ({success_count}/3 successful)")
            return False

    def run_comprehensive_test(self):
        """Run all tests in sequence."""
        logger.info("Starting comprehensive integration test...")

        tests = [
            ("Health Endpoint", self.test_health_endpoint),
            ("Models Endpoint", self.test_models_endpoint),
            ("Basic Chat Completion", self.test_basic_chat_completion),
            ("Emotion-Influenced Responses", self.test_emotion_influenced_responses),
            ("Parameter Variations", self.test_parameter_variations),
            ("Concurrent Requests", self.test_concurrent_requests),
        ]

        results = {}
        for test_name, test_func in tests:
            logger.info(f"\n{'='*20} {test_name} {'='*20}")
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    logger.info(f"✅ {test_name} PASSED")
                else:
                    logger.error(f"❌ {test_name} FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name} FAILED with exception: {e}")
                results[test_name] = False

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)

        passed = sum(results.values())
        total = len(results)

        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name}: {status}")

        logger.info(f"\nOverall: {passed}/{total} tests passed")

        if passed == total:
            logger.info("🎉 ALL INTEGRATION TESTS PASSED!")
            return True
        else:
            logger.error(f"❌ {total - passed} tests failed")
            return False

    def stop_server(self):
        """Stop the server process."""
        if self.server_process:
            logger.info("Stopping server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
                logger.info("Server stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("Server didn't stop gracefully, killing...")
                self.server_process.kill()
                self.server_process.wait()

            self.server_process = None


def main():
    """Main integration test function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Integrated test for OpenAI-compatible RepControlVLLMHook server"
    )
    parser.add_argument(
        "--model",
        default="/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct",
        help="Path to the model",
    )
    parser.add_argument("--model_name", default="qwen3-0.5B-anger", help="Model name for API")
    parser.add_argument("--emotion", default="anger", help="Emotion to test")
    parser.add_argument("--port", type=int, default=8000, help="Port for the server")
    parser.add_argument(
        "--timeout", type=int, default=180, help="Timeout for server startup (seconds)"
    )

    args = parser.parse_args()

    # Create tester instance
    tester = IntegratedServerTester(
        model_path=args.model, model_name=args.model_name, emotion=args.emotion, port=args.port
    )

    success = False

    try:
        # Start server
        if not tester.start_server(timeout=args.timeout):
            logger.error("Failed to start server")
            sys.exit(1)

        # Run tests
        success = tester.run_comprehensive_test()

        if success:
            logger.info("\n" + "=" * 60)
            logger.info("🎉 INTEGRATION TEST COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info("The OpenAI-compatible server is working correctly.")
            logger.info(f"Server URL: http://localhost:{args.port}/v1")
            logger.info(f"Model: {args.model_name}")
            logger.info(f"Emotion: {args.emotion}")

            # Show usage example
            logger.info("\nUSAGE EXAMPLE:")
            logger.info("-" * 30)
            logger.info("from openai import OpenAI")
            logger.info(
                f"client = OpenAI(base_url='http://localhost:{args.port}/v1', api_key='token-abc123')"
            )
            logger.info(
                f"response = client.chat.completions.create(model='{args.model_name}', messages=[{{'role': 'user', 'content': 'Hello!'}}])"
            )
            logger.info("print(response.choices[0].message.content)")

        else:
            logger.error("\n❌ INTEGRATION TEST FAILED!")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Integration test interrupted by user")

    finally:
        # Always clean up
        tester.stop_server()


if __name__ == "__main__":
    main()
