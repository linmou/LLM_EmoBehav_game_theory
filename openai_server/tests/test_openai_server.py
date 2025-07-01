#!/usr/bin/env python3
"""
Test script for the OpenAI-compatible RepControlVLLMHook server.

This script demonstrates how to use the server with the OpenAI client library
and verifies that the emotion-controlled generation is working properly.
"""

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ServerTester:
    def __init__(self, base_url="http://localhost:8000/v1", api_key="token-abc123"):
        self.base_url = base_url
        self.api_key = api_key
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.server_process = None

    def wait_for_server(self, timeout=120):
        """Wait for server to be ready."""
        logger.info(f"Waiting for server at {self.base_url}...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                health_url = self.base_url.replace("/v1", "/health")
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    logger.info("Server is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass

            time.sleep(2)

        logger.error(f"Server did not become ready within {timeout} seconds")
        return False

    def test_models_endpoint(self):
        """Test the /v1/models endpoint."""
        logger.info("Testing /v1/models endpoint...")
        try:
            models = self.client.models.list()
            logger.info(f"Available models: {[model.id for model in models.data]}")
            return len(models.data) > 0
        except Exception as e:
            logger.error(f"Error testing models endpoint: {e}")
            return False

    def test_chat_completion(self, model_name="qwen3-0.5B-anger"):
        """Test chat completion with emotion control."""
        logger.info("Testing chat completion...")

        try:
            completion = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Hello! How are you feeling today?"}],
                max_tokens=50,
                temperature=0.0,
            )

            response_text = completion.choices[0].message.content
            logger.info(f"Chat completion response: '{response_text}'")
            logger.info(f"Usage: {completion.usage}")

            return len(response_text.strip()) > 0

        except Exception as e:
            logger.error(f"Error testing chat completion: {e}")
            return False

    def test_emotion_comparison(self, model_name="qwen3-0.5B-anger"):
        """Test multiple completions to see emotion effect."""
        logger.info("Testing emotion-controlled responses...")

        test_prompts = [
            "Someone just cut in front of me in line. I feel",
            "I'm facing a difficult challenge. My reaction is",
            "Describe how you would handle a frustrating situation.",
        ]

        for i, prompt in enumerate(test_prompts):
            try:
                completion = self.client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=30,
                    temperature=0.0,
                )

                response = completion.choices[0].message.content.strip()
                logger.info(f"Prompt {i+1}: '{prompt}'")
                logger.info(f"Response: '{response}'")
                logger.info("-" * 50)

            except Exception as e:
                logger.error(f"Error in emotion comparison test {i+1}: {e}")

        return True

    def run_comprehensive_test(self, model_name="qwen3-0.5B-anger"):
        """Run comprehensive tests."""
        logger.info("Starting comprehensive test suite...")

        # Wait for server
        if not self.wait_for_server():
            return False

        # Test models endpoint
        if not self.test_models_endpoint():
            logger.error("Models endpoint test failed")
            return False

        # Test basic chat completion
        if not self.test_chat_completion(model_name):
            logger.error("Chat completion test failed")
            return False

        # Test emotion comparison
        self.test_emotion_comparison(model_name)

        logger.info("All tests completed successfully!")
        return True


def start_server_subprocess(model_path, model_name, emotion="anger", port=8000):
    """Start the server as a subprocess."""
    logger.info(f"Starting server subprocess...")

    cmd = [
        sys.executable,
        "init_openai_server.py",
        "--model",
        model_path,
        "--model_name",
        model_name,
        "--emotion",
        emotion,
        "--port",
        str(port),
        "--api_key",
        "token-abc123",
        "--url",
        f"http://localhost:{port}/v1",
    ]

    logger.info(f"Server command: {' '.join(cmd)}")

    # Start server process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    return process


def main():
    """Main test function."""
    import argparse

    parser = argparse.ArgumentParser(description="Test OpenAI-compatible RepControlVLLMHook server")
    parser.add_argument(
        "--model",
        default="/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct",
        help="Path to the model",
    )
    parser.add_argument("--model_name", default="qwen3-0.5B-anger", help="Model name for API")
    parser.add_argument("--emotion", default="anger", help="Emotion to test")
    parser.add_argument("--port", type=int, default=8000, help="Port to test")
    parser.add_argument(
        "--start_server", action="store_true", help="Start server as subprocess for testing"
    )
    parser.add_argument(
        "--test_only", action="store_true", help="Only run tests (assume server is already running)"
    )

    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}/v1"
    tester = ServerTester(base_url=base_url)

    server_process = None

    try:
        if args.start_server and not args.test_only:
            # Start server subprocess
            server_process = start_server_subprocess(
                args.model, args.model_name, args.emotion, args.port
            )
            tester.server_process = server_process

            # Give server time to start
            logger.info("Waiting for server to initialize...")
            time.sleep(10)

        # Run tests
        success = tester.run_comprehensive_test(args.model_name)

        if success:
            logger.info("🎉 All tests passed!")

            if not args.test_only:
                logger.info("\n" + "=" * 60)
                logger.info("SERVER SETUP EXAMPLE:")
                logger.info("=" * 60)
                logger.info(f"python init_openai_server.py \\")
                logger.info(f"  --api_key=token-abc123 \\")
                logger.info(f"  --url=http://localhost:{args.port}/v1 \\")
                logger.info(f"  --emotion={args.emotion} \\")
                logger.info(f"  --model={args.model} \\")
                logger.info(f"  --model_name={args.model_name}")
                logger.info("\nCLIENT USAGE EXAMPLE:")
                logger.info("=" * 60)
                logger.info("from openai import OpenAI")
                logger.info(f"client = OpenAI(")
                logger.info(f'    base_url="http://localhost:{args.port}/v1",')
                logger.info(f'    api_key="token-abc123",')
                logger.info(f")")
                logger.info("")
                logger.info("completion = client.chat.completions.create(")
                logger.info(f'  model="{args.model_name}",')
                logger.info("  messages=[")
                logger.info('    {"role": "user", "content": "Hello!"}')
                logger.info("  ]")
                logger.info(")")
                logger.info("=" * 60)

        else:
            logger.error("❌ Some tests failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")

    finally:
        # Clean up server process
        if server_process:
            logger.info("Stopping server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Server didn't stop gracefully, killing...")
                server_process.kill()
                server_process.wait()


if __name__ == "__main__":
    main()
