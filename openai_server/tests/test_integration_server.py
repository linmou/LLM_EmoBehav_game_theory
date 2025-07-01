#!/usr/bin/env python3
"""
Integration Tests for OpenAI Server

This module contains integration tests that start the actual server
and test end-to-end functionality.
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests
from openai import OpenAI

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))


class TestServerIntegration:
    """Integration tests for the OpenAI server"""

    @classmethod
    def setup_class(cls):
        """Set up test fixtures"""
        cls.test_port = 8765  # Use non-standard port for testing
        cls.base_url = f"http://localhost:{cls.test_port}/v1"
        cls.model_path = os.getenv(
            "TEST_MODEL_PATH", "/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct"
        )
        cls.server_process = None

    @classmethod
    def teardown_class(cls):
        """Clean up after tests"""
        if cls.server_process:
            cls.stop_server()

    @classmethod
    def start_server(cls, emotion="happiness"):
        """Start the server for testing"""
        cmd = [
            sys.executable,
            "-m",
            "openai_server",
            "--model",
            cls.model_path,
            "--model_name",
            f"test-model-{emotion}",
            "--emotion",
            emotion,
            "--port",
            str(cls.test_port),
            "--host",
            "localhost",
        ]

        # Set environment variable for GPU memory
        env = os.environ.copy()
        env["VLLM_GPU_MEMORY_UTILIZATION"] = "0.5"  # Use less memory for tests

        cls.server_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )

        # Wait for server to start
        cls.wait_for_server()

    @classmethod
    def stop_server(cls):
        """Stop the test server"""
        if cls.server_process:
            cls.server_process.terminate()
            try:
                cls.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                cls.server_process.kill()
                cls.server_process.wait()
            cls.server_process = None

    @classmethod
    def wait_for_server(cls, timeout=60):
        """Wait for server to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{cls.base_url}/models", timeout=5)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)
        raise TimeoutError(f"Server did not start within {timeout} seconds")

    def test_models_endpoint(self):
        """Test the /v1/models endpoint"""
        response = requests.get(f"{self.base_url}/models")
        assert response.status_code == 200

        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0
        assert "test-model" in data["data"][0]["id"]

    def test_health_endpoint(self):
        """Test the /health endpoint"""
        health_url = self.base_url.replace("/v1", "/health")
        response = requests.get(health_url)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "model" in data
        assert "emotion" in data

    def test_chat_completion_basic(self):
        """Test basic chat completion"""
        client = OpenAI(base_url=self.base_url, api_key="dummy")

        completion = client.chat.completions.create(
            model="test-model-happiness",
            messages=[{"role": "user", "content": "Say hello in 5 words or less"}],
            max_tokens=20,
            temperature=0.0,
        )

        assert completion.id is not None
        assert completion.choices[0].message.content is not None
        assert len(completion.choices[0].message.content.strip()) > 0
        assert completion.usage.total_tokens > 0

    def test_chat_completion_streaming(self):
        """Test streaming chat completion"""
        client = OpenAI(base_url=self.base_url, api_key="dummy")

        stream = client.chat.completions.create(
            model="test-model-happiness",
            messages=[{"role": "user", "content": "Count to 3"}],
            max_tokens=20,
            stream=True,
        )

        chunks = []
        for chunk in stream:
            if chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)

        full_response = "".join(chunks)
        assert len(full_response) > 0

    def test_emotion_switching(self):
        """Test switching between different emotions"""
        # Stop current server
        self.stop_server()

        # Test with anger emotion
        self.start_server(emotion="anger")
        client = OpenAI(base_url=self.base_url, api_key="dummy")

        anger_response = client.chat.completions.create(
            model="test-model-anger",
            messages=[{"role": "user", "content": "How do you feel about delays?"}],
            max_tokens=30,
            temperature=0.0,
        )

        # Stop and restart with happiness
        self.stop_server()
        self.start_server(emotion="happiness")

        happiness_response = client.chat.completions.create(
            model="test-model-happiness",
            messages=[{"role": "user", "content": "How do you feel about delays?"}],
            max_tokens=30,
            temperature=0.0,
        )

        # Responses should be different due to emotion
        assert (
            anger_response.choices[0].message.content
            != happiness_response.choices[0].message.content
        )

    def test_concurrent_requests(self):
        """Test handling multiple concurrent requests"""
        client = OpenAI(base_url=self.base_url, api_key="dummy")

        async def make_request(prompt):
            completion = client.chat.completions.create(
                model="test-model-happiness",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
            )
            return completion.choices[0].message.content

        # Make multiple requests concurrently
        prompts = ["Say yes", "Say no", "Say maybe"]

        # Use threading for concurrent requests
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request, p) for p in prompts]
            results = [f.result() for f in futures]

        assert len(results) == 3
        assert all(r is not None for r in results)

    def test_error_handling(self):
        """Test error handling for invalid requests"""
        client = OpenAI(base_url=self.base_url, api_key="dummy")

        # Test with invalid model name
        with pytest.raises(Exception):
            client.chat.completions.create(
                model="invalid-model", messages=[{"role": "user", "content": "test"}]
            )

        # Test with empty messages
        with pytest.raises(Exception):
            client.chat.completions.create(model="test-model-happiness", messages=[])

    def test_long_context(self):
        """Test handling of long context"""
        client = OpenAI(base_url=self.base_url, api_key="dummy")

        # Create a long conversation
        messages = [
            {"role": "user", "content": "Remember the number 42."},
            {"role": "assistant", "content": "I'll remember 42."},
            {"role": "user", "content": "What number did I ask you to remember?"},
        ]

        completion = client.chat.completions.create(
            model="test-model-happiness", messages=messages, max_tokens=20, temperature=0.0
        )

        response = completion.choices[0].message.content.lower()
        assert "42" in response or "forty-two" in response


class TestServerRobustness:
    """Test server robustness and edge cases"""

    def test_server_restart(self):
        """Test server can be restarted cleanly"""
        # This test would use the integration setup
        # but tests restart behavior
        pass

    def test_memory_usage(self):
        """Test server memory usage stays reasonable"""
        # This would monitor memory during operation
        pass

    def test_timeout_handling(self):
        """Test request timeout handling"""
        # Test with very long generation that might timeout
        pass


if __name__ == "__main__":
    # Run only if GPU is available
    import torch

    if torch.cuda.is_available():
        pytest.main([__file__, "-v", "-s"])
    else:
        print("Skipping integration tests - no GPU available")
