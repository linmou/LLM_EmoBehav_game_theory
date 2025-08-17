#!/usr/bin/env python3
"""
Simple example showing how to use function calling with the OpenAI client.
"""

from openai import OpenAI

# Initialize client for happiness server (port 8000)
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# Define a simple function
tools = [
    {
        "type": "function",
        "function": {
            "name": "make_decision",
            "description": "Make a decision in a game theory scenario",
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
                "required": ["decision", "confidence"],
            },
        },
    }
]


def test_simple_function_call():
    """Test a simple function call."""
    print("🧠 Testing Simple Function Call with Happiness Emotion")
    print("=" * 50)

    try:
        response = client.chat.completions.create(
            model="Qwen2.5-0.5B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": "You're in a prisoner's dilemma. Both players get 3 points for mutual cooperation, 1 point for mutual defection. If one defects while other cooperates, defector gets 5 points. Make your decision using the function.",
                }
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=150,
            temperature=0.7,
        )

        # Print response
        choice = response.choices[0]
        message = choice.message

        print(f"📝 Response Content: {message.content}")
        print(f"🏁 Finish Reason: {choice.finish_reason}")

        if message.tool_calls:
            print(f"\n🔧 Function Calls:")
            for call in message.tool_calls:
                print(f"  ID: {call.id}")
                print(f"  Function: {call.function.name}")
                print(f"  Arguments: {call.function.arguments}")

                # Try to parse arguments
                try:
                    import json

                    args = json.loads(call.function.arguments)
                    print(f"  Parsed Arguments:")
                    for key, value in args.items():
                        print(f"    {key}: {value}")
                except:
                    print(f"  Raw Arguments: {call.function.arguments}")
        else:
            print("❌ No function calls made")

    except Exception as e:
        print(f"❌ Error: {e}")


def test_with_openai_client():
    """Test using standard OpenAI client patterns."""
    print("\n🔄 Testing with OpenAI Client Patterns")
    print("=" * 50)

    messages = [
        {
            "role": "user",
            "content": "Help me analyze this prisoner's dilemma scenario and make a decision.",
        }
    ]

    try:
        # First call - get function call
        response = client.chat.completions.create(
            model="Qwen2.5-0.5B-Instruct", messages=messages, tools=tools, max_tokens=200
        )

        response_message = response.choices[0].message
        messages.append(
            {
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": call.type,
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in (response_message.tool_calls or [])
                ],
            }
        )

        print(f"🤖 Assistant: {response_message.content}")

        if response_message.tool_calls:
            print(f"🔧 Function calls: {len(response_message.tool_calls)}")

            # Simulate function execution and add results
            for call in response_message.tool_calls:
                print(f"  Executing: {call.function.name}")

                # Mock function result
                function_result = {
                    "status": "success",
                    "executed_decision": call.function.arguments,
                    "outcome": "Decision recorded in game theory experiment",
                }

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(function_result),
                    }
                )

            # Continue conversation
            response2 = client.chat.completions.create(
                model="Qwen2.5-0.5B-Instruct", messages=messages, max_tokens=100
            )

            print(f"🤖 Follow-up: {response2.choices[0].message.content}")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_simple_function_call()
    test_with_openai_client()
