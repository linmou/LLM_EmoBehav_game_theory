#!/usr/bin/env python3
"""
Integration tests for function calling in the OpenAI server.
These tests require a running server instance.
"""

import json
import time
from unittest.mock import Mock, patch

import pytest
import requests

# Test configuration
TEST_SERVERS = [
    {"name": "happiness", "port": 8000, "emotion": "happiness"},
    {"name": "anger", "port": 8001, "emotion": "anger"},
]

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_game_decision",
            "description": "Make a decision in a prisoner's dilemma scenario",
            "parameters": {
                "type": "object",
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": ["cooperate", "defect"],
                        "description": "The decision to make",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Confidence level in the decision",
                    },
                    "reasoning": {"type": "string", "description": "Reasoning behind the decision"},
                },
                "required": ["decision"],
            },
        },
    },
]


class TestFunctionCallingAPI:
    """Test function calling through the API endpoints."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        self.base_request = {
            "model": "Qwen2.5-0.5B-Instruct",
            "max_tokens": 150,
            "temperature": 0.7,
        }

    def is_server_running(self, port):
        """Check if server is running on given port."""
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    @pytest.mark.parametrize("server", TEST_SERVERS)
    def test_server_health_check(self, server):
        """Test that servers are running and healthy."""
        if not self.is_server_running(server["port"]):
            pytest.skip(f"Server on port {server['port']} not running")

        response = requests.get(f"http://localhost:{server['port']}/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["current_emotion"] == server["emotion"]

    @pytest.mark.parametrize("server", TEST_SERVERS)
    def test_basic_function_call_request(self, server):
        """Test basic function calling request format."""
        if not self.is_server_running(server["port"]):
            pytest.skip(f"Server on port {server['port']} not running")

        request_data = {
            **self.base_request,
            "messages": [{"role": "user", "content": "What's the weather like in New York?"}],
            "tools": SAMPLE_TOOLS,
            "tool_choice": "auto",
        }

        response = requests.post(
            f"http://localhost:{server['port']}/v1/chat/completions", json=request_data, timeout=30
        )

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "choices" in data
        assert len(data["choices"]) > 0

        choice = data["choices"][0]
        assert "message" in choice
        assert "finish_reason" in choice

        message = choice["message"]
        assert "role" in message
        assert message["role"] == "assistant"

        # Check for either content or tool_calls
        assert "content" in message or "tool_calls" in message

    @pytest.mark.parametrize("server", TEST_SERVERS)
    def test_function_call_parsing(self, server):
        """Test that function calls are properly parsed and returned."""
        if not self.is_server_running(server["port"]):
            pytest.skip(f"Server on port {server['port']} not running")

        request_data = {
            **self.base_request,
            "messages": [{"role": "user", "content": "Get weather for Paris"}],
            "tools": [SAMPLE_TOOLS[0]],  # Only weather tool
            "tool_choice": "auto",
            "max_tokens": 200,
        }

        response = requests.post(
            f"http://localhost:{server['port']}/v1/chat/completions", json=request_data, timeout=30
        )

        assert response.status_code == 200
        data = response.json()

        choice = data["choices"][0]
        message = choice["message"]

        # If tool calls were made, verify structure
        if "tool_calls" in message and message["tool_calls"]:
            for tool_call in message["tool_calls"]:
                assert "id" in tool_call
                assert "type" in tool_call
                assert tool_call["type"] == "function"
                assert "function" in tool_call

                function = tool_call["function"]
                assert "name" in function
                assert "arguments" in function

                # Verify it's a valid function name
                assert function["name"] in ["get_weather", "make_game_decision"]

                # Verify arguments are valid JSON
                try:
                    json.loads(function["arguments"])
                except json.JSONDecodeError:
                    pytest.fail(f"Invalid JSON in function arguments: {function['arguments']}")

            # Check finish reason
            assert choice["finish_reason"] == "tool_calls"

    def test_emotion_difference_in_function_calls(self):
        """Test that different emotions may lead to different function call patterns."""
        # Skip if servers not running
        if not all(self.is_server_running(server["port"]) for server in TEST_SERVERS):
            pytest.skip("Both servers not running")

        request_data = {
            **self.base_request,
            "messages": [
                {
                    "role": "user",
                    "content": "You're in a prisoner's dilemma with high stakes. Make your decision.",
                }
            ],
            "tools": [SAMPLE_TOOLS[1]],  # Game decision tool
            "tool_choice": "auto",
            "max_tokens": 200,
        }

        responses = {}

        for server in TEST_SERVERS:
            response = requests.post(
                f"http://localhost:{server['port']}/v1/chat/completions",
                json=request_data,
                timeout=30,
            )

            assert response.status_code == 200
            responses[server["emotion"]] = response.json()

        # Compare responses (this is a behavioral test - results may vary)
        for emotion, data in responses.items():
            choice = data["choices"][0]
            message = choice["message"]

            print(f"\n{emotion.capitalize()} response:")
            print(f"Content: {message.get('content', 'None')}")
            if message.get("tool_calls"):
                for call in message["tool_calls"]:
                    print(f"Function: {call['function']['name']}")
                    print(f"Arguments: {call['function']['arguments']}")

    @pytest.mark.parametrize("server", TEST_SERVERS)
    def test_no_tools_provided(self, server):
        """Test behavior when no tools are provided."""
        if not self.is_server_running(server["port"]):
            pytest.skip(f"Server on port {server['port']} not running")

        request_data = {
            **self.base_request,
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            # No tools provided
        }

        response = requests.post(
            f"http://localhost:{server['port']}/v1/chat/completions", json=request_data, timeout=30
        )

        assert response.status_code == 200
        data = response.json()

        choice = data["choices"][0]
        message = choice["message"]

        # Should not have tool calls
        assert message.get("tool_calls") is None or len(message.get("tool_calls", [])) == 0
        assert choice["finish_reason"] == "stop"
        assert "content" in message

    @pytest.mark.parametrize("server", TEST_SERVERS)
    def test_tool_choice_none(self, server):
        """Test tool_choice='none' prevents tool usage."""
        if not self.is_server_running(server["port"]):
            pytest.skip(f"Server on port {server['port']} not running")

        request_data = {
            **self.base_request,
            "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
            "tools": SAMPLE_TOOLS,
            "tool_choice": "none",  # Explicitly disable tools
        }

        response = requests.post(
            f"http://localhost:{server['port']}/v1/chat/completions", json=request_data, timeout=30
        )

        assert response.status_code == 200
        data = response.json()

        choice = data["choices"][0]
        message = choice["message"]

        # Should not use tools even though they're available
        assert message.get("tool_calls") is None or len(message.get("tool_calls", [])) == 0
        assert choice["finish_reason"] == "stop"
        assert "content" in message

    @pytest.mark.parametrize("server", TEST_SERVERS)
    def test_streaming_with_function_calls(self, server):
        """Test streaming responses with function calls."""
        if not self.is_server_running(server["port"]):
            pytest.skip(f"Server on port {server['port']} not running")

        request_data = {
            **self.base_request,
            "messages": [{"role": "user", "content": "Check the weather for London"}],
            "tools": [SAMPLE_TOOLS[0]],
            "tool_choice": "auto",
            "stream": True,
        }

        response = requests.post(
            f"http://localhost:{server['port']}/v1/chat/completions",
            json=request_data,
            timeout=30,
            stream=True,
        )

        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")

        # Read streaming response
        content = ""
        for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
            if chunk:
                content += chunk

        # Should contain SSE formatted data
        assert "data:" in content
        # May contain function call information in streaming format

    def test_multi_turn_conversation_with_tools(self):
        """Test multi-turn conversation with tool usage."""
        if not self.is_server_running(8000):  # Test with happiness server
            pytest.skip("Happiness server not running")

        # First turn - user asks for weather
        request1 = {
            **self.base_request,
            "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
            "tools": SAMPLE_TOOLS,
            "tool_choice": "auto",
        }

        response1 = requests.post(
            "http://localhost:8000/v1/chat/completions", json=request1, timeout=30
        )

        assert response1.status_code == 200
        data1 = response1.json()

        # Build conversation history
        messages = request1["messages"].copy()
        assistant_message = data1["choices"][0]["message"]
        messages.append(
            {
                "role": "assistant",
                "content": assistant_message.get("content"),
                "tool_calls": assistant_message.get("tool_calls"),
            }
        )

        # Add mock tool result if there were tool calls
        if assistant_message.get("tool_calls"):
            for call in assistant_message["tool_calls"]:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps({"temperature": "22°C", "condition": "sunny"}),
                    }
                )

        # Second turn - continue conversation
        request2 = {
            **self.base_request,
            "messages": messages
            + [{"role": "user", "content": "Thanks! Now make a decision in a prisoner's dilemma."}],
            "tools": SAMPLE_TOOLS,
            "tool_choice": "auto",
        }

        response2 = requests.post(
            "http://localhost:8000/v1/chat/completions", json=request2, timeout=30
        )

        assert response2.status_code == 200
        data2 = response2.json()

        # Should handle multi-turn conversation properly
        choice2 = data2["choices"][0]
        assert "message" in choice2
        assert choice2["message"]["role"] == "assistant"


class TestFunctionCallingErrorHandling:
    """Test error handling in function calling scenarios."""

    def test_invalid_tool_definition(self):
        """Test handling of invalid tool definitions."""
        if not requests.get("http://localhost:8000/health", timeout=2).status_code == 200:
            pytest.skip("Server not running")

        request_data = {
            "model": "Qwen2.5-0.5B-Instruct",
            "messages": [{"role": "user", "content": "Test"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        # Missing required fields
                        "description": "Invalid function"
                    },
                }
            ],
            "max_tokens": 50,
        }

        response = requests.post(
            "http://localhost:8000/v1/chat/completions", json=request_data, timeout=30
        )

        # Should handle gracefully (either error or ignore invalid tool)
        assert response.status_code in [200, 400, 422]

    def test_malformed_request(self):
        """Test handling of malformed requests."""
        if not requests.get("http://localhost:8000/health", timeout=2).status_code == 200:
            pytest.skip("Server not running")

        request_data = {
            "model": "Qwen2.5-0.5B-Instruct",
            "messages": [{"role": "user", "content": "Test"}],
            "tools": "invalid_tools_format",  # Should be array
            "max_tokens": 50,
        }

        response = requests.post(
            "http://localhost:8000/v1/chat/completions", json=request_data, timeout=30
        )

        # Should return validation error
        assert response.status_code in [400, 422]


@pytest.mark.performance
class TestFunctionCallingPerformance:
    """Performance tests for function calling."""

    def test_function_call_latency(self):
        """Test latency of function calling requests."""
        if not requests.get("http://localhost:8000/health", timeout=2).status_code == 200:
            pytest.skip("Server not running")

        request_data = {
            "model": "Qwen2.5-0.5B-Instruct",
            "messages": [{"role": "user", "content": "Get weather for New York"}],
            "tools": [SAMPLE_TOOLS[0]],
            "tool_choice": "auto",
            "max_tokens": 100,
        }

        # Measure latency
        start_time = time.time()

        response = requests.post(
            "http://localhost:8000/v1/chat/completions", json=request_data, timeout=30
        )

        end_time = time.time()
        latency = end_time - start_time

        assert response.status_code == 200

        # Performance assertion (adjust based on expected performance)
        assert latency < 10.0, f"Function call took too long: {latency:.2f}s"

        print(f"Function call latency: {latency:.2f}s")

    def test_concurrent_function_calls(self):
        """Test concurrent function calling requests."""
        if not requests.get("http://localhost:8000/health", timeout=2).status_code == 200:
            pytest.skip("Server not running")

        import concurrent.futures

        def make_request():
            request_data = {
                "model": "Qwen2.5-0.5B-Instruct",
                "messages": [{"role": "user", "content": "Make a quick decision"}],
                "tools": [SAMPLE_TOOLS[1]],
                "tool_choice": "auto",
                "max_tokens": 50,
            }

            response = requests.post(
                "http://localhost:8000/v1/chat/completions", json=request_data, timeout=30
            )
            return response.status_code == 200

        # Test with 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # All requests should succeed
        assert all(results), "Some concurrent function call requests failed"


if __name__ == "__main__":
    # Run with: python -m pytest test_function_calling_integration.py -v
    pytest.main([__file__, "-v", "-s"])
