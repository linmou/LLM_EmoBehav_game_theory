#!/usr/bin/env python3
"""
Demo script showing function calling capabilities of the OpenAI-compatible server.
"""

import json
import time

import requests


def test_function_calling():
    """Test function calling with the emotion-controlled server."""

    # Define some example functions
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
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
                "name": "calculate_cooperation_probability",
                "description": "Calculate cooperation probability in a prisoner's dilemma scenario",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "emotion": {
                            "type": "string",
                            "enum": ["happiness", "anger", "neutral"],
                            "description": "Current emotional state",
                        },
                        "payoff_cooperative": {
                            "type": "number",
                            "description": "Payoff for mutual cooperation",
                        },
                        "payoff_defection": {
                            "type": "number",
                            "description": "Payoff for mutual defection",
                        },
                    },
                    "required": ["emotion"],
                },
            },
        },
    ]

    # Test requests
    test_cases = [
        {
            "name": "Simple weather request",
            "messages": [{"role": "user", "content": "What's the weather like in New York?"}],
            "tools": tools,
            "expected_function": "get_weather",
        },
        {
            "name": "Game theory analysis",
            "messages": [
                {
                    "role": "user",
                    "content": "I'm feeling angry. What's my cooperation probability in a prisoner's dilemma with payoffs 3 for cooperation and 1 for defection?",
                }
            ],
            "tools": tools,
            "expected_function": "calculate_cooperation_probability",
        },
        {
            "name": "No function needed",
            "messages": [{"role": "user", "content": "How are you doing today?"}],
            "tools": tools,
            "expected_function": None,
        },
    ]

    print("🔧 Testing Function Calling with Emotion-Controlled Server")
    print("=" * 60)

    # Test both happiness and anger servers
    servers = [
        {"name": "Happiness Server", "port": 8000, "emotion": "happiness"},
        {"name": "Anger Server", "port": 8001, "emotion": "anger"},
    ]

    for server in servers:
        print(f"\n🧠 Testing {server['name']} (Port {server['port']})")
        print("-" * 40)

        # Check server health
        try:
            health_response = requests.get(f"http://localhost:{server['port']}/health", timeout=5)
            if health_response.status_code == 200:
                health_data = health_response.json()
                print(
                    f"✅ Server healthy - Emotion: {health_data.get('current_emotion', 'unknown')}"
                )
            else:
                print(f"❌ Server not healthy: {health_response.status_code}")
                continue
        except Exception as e:
            print(f"❌ Server not accessible: {e}")
            continue

        # Test each case
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 Test {i}: {test_case['name']}")

            request_data = {
                "model": "Qwen2.5-0.5B-Instruct",
                "messages": test_case["messages"],
                "tools": test_case["tools"],
                "tool_choice": "auto",
                "max_tokens": 200,
                "temperature": 0.7,
            }

            try:
                response = requests.post(
                    f"http://localhost:{server['port']}/v1/chat/completions",
                    json=request_data,
                    timeout=30,
                )

                if response.status_code == 200:
                    result = response.json()

                    # Extract response details
                    choice = result["choices"][0]
                    message = choice["message"]
                    finish_reason = choice["finish_reason"]

                    print(f"  📝 Content: {message.get('content', 'None')}")
                    print(f"  🏁 Finish reason: {finish_reason}")

                    # Check for tool calls
                    if message.get("tool_calls"):
                        print(f"  🔧 Tool calls found: {len(message['tool_calls'])}")
                        for call in message["tool_calls"]:
                            func_name = call["function"]["name"]
                            func_args = call["function"]["arguments"]
                            print(f"    - Function: {func_name}")
                            print(f"    - Arguments: {func_args}")

                            # Verify expected function
                            if test_case["expected_function"] == func_name:
                                print(f"    ✅ Expected function called!")
                            elif test_case["expected_function"]:
                                print(
                                    f"    ⚠️  Expected {test_case['expected_function']}, got {func_name}"
                                )
                    else:
                        if test_case["expected_function"]:
                            print(
                                f"  ⚠️  Expected function {test_case['expected_function']}, but no tool calls made"
                            )
                        else:
                            print(f"  ✅ No tool calls (as expected)")

                else:
                    print(f"  ❌ Request failed: {response.status_code} - {response.text}")

            except Exception as e:
                print(f"  ❌ Request error: {e}")

            time.sleep(1)  # Rate limiting

    print(f"\n🎯 Function calling demo completed!")
    print("\n💡 Key Features Demonstrated:")
    print("  - Tool/function definitions in requests")
    print("  - Automatic function call parsing from model responses")
    print("  - Proper OpenAI-compatible response format")
    print("  - Emotion-controlled function calling behavior")


if __name__ == "__main__":
    test_function_calling()
