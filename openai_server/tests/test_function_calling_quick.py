#!/usr/bin/env python3
"""
Quick validation tests for function calling implementation.
These tests don't require a running server.
"""

import json
import sys
from pathlib import Path

import pytest

# Add server module and project root to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from server import (
    ChatCompletionRequest,
    ChatMessage,
    FunctionDefinition,
    ToolCall,
    ToolDefinition,
    apply_emotion_to_prompt,
    format_tools_for_prompt,
    parse_tool_calls_from_response,
)


def test_basic_models():
    """Test that all function calling models can be instantiated."""
    # Test FunctionDefinition
    func_def = FunctionDefinition(
        name="test_function",
        description="A test function",
        parameters={"type": "object", "properties": {"param1": {"type": "string"}}},
    )
    assert func_def.name == "test_function"

    # Test ToolDefinition
    tool_def = ToolDefinition(function=func_def)
    assert tool_def.type == "function"
    assert tool_def.function.name == "test_function"

    # Test ToolCall
    tool_call = ToolCall(
        id="call_123", function={"name": "test_function", "arguments": '{"param1": "value"}'}
    )
    assert tool_call.id == "call_123"
    assert tool_call.function["name"] == "test_function"

    # Test ChatMessage with tool_calls
    message = ChatMessage(role="assistant", content="I'll call a function", tool_calls=[tool_call])
    assert message.role == "assistant"
    assert len(message.tool_calls) == 1

    # Test ChatCompletionRequest with tools
    request = ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(content="Hello")],
        tools=[tool_def],
        tool_choice="auto",
    )
    assert len(request.tools) == 1
    assert request.tool_choice == "auto"


def test_tool_formatting():
    """Test tool formatting for prompts."""
    func_def = FunctionDefinition(
        name="get_weather",
        description="Get weather information",
        parameters={
            "type": "object",
            "properties": {"location": {"type": "string", "description": "City name"}},
            "required": ["location"],
        },
    )
    tool_def = ToolDefinition(function=func_def)

    result = format_tools_for_prompt([tool_def])

    assert "get_weather" in result
    assert "Get weather information" in result
    assert "location" in result
    assert "tool_calls" in result
    assert "function" in result


def test_tool_call_parsing():
    """Test parsing tool calls from responses."""
    # Test JSON format
    response_text = """{"tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": "{\\"location\\": \\"Paris\\"}"}}]}"""

    tool_calls = parse_tool_calls_from_response(response_text)

    assert len(tool_calls) == 1
    assert tool_calls[0].id == "call_123"
    assert tool_calls[0].function["name"] == "get_weather"
    assert "Paris" in tool_calls[0].function["arguments"]

    # Test function syntax format
    response_text2 = "I'll call get_weather(Paris) to get the information."
    tool_calls2 = parse_tool_calls_from_response(response_text2)

    assert len(tool_calls2) == 1
    assert tool_calls2[0].function["name"] == "get_weather"

    # Test no tool calls
    response_text3 = "I can help you with that."
    tool_calls3 = parse_tool_calls_from_response(response_text3)

    assert len(tool_calls3) == 0


def test_prompt_generation():
    """Test prompt generation with tools."""
    func_def = FunctionDefinition(name="test_func", description="Test function", parameters={})
    tool_def = ToolDefinition(function=func_def)

    messages = [ChatMessage(role="user", content="Help me with something")]

    # Test with tools
    prompt = apply_emotion_to_prompt(messages, "happiness", [tool_def])

    assert "test_func" in prompt
    assert "Test function" in prompt
    assert "tool_calls" in prompt
    assert "User: Help me with something" in prompt
    assert "Assistant:" in prompt

    # Test without tools
    prompt2 = apply_emotion_to_prompt(messages, "happiness", None)

    assert "test_func" not in prompt2
    assert "User: Help me with something" in prompt2
    assert "Assistant:" in prompt2


def test_conversation_with_tool_calls():
    """Test multi-turn conversation with tool calls."""
    tool_call = ToolCall(id="call_123", function={"name": "test_func", "arguments": "{}"})

    messages = [
        ChatMessage(role="user", content="Call a function"),
        ChatMessage(role="assistant", content="I'll call the function", tool_calls=[tool_call]),
        ChatMessage(role="tool", content="Function result", tool_call_id="call_123"),
        ChatMessage(role="user", content="Thanks!"),
    ]

    prompt = apply_emotion_to_prompt(messages, "happiness", None)

    assert "User: Call a function" in prompt
    assert "Assistant: I'll call the function" in prompt
    assert "Tool calls:" in prompt
    assert "Tool Result (ID: call_123): Function result" in prompt
    assert "User: Thanks!" in prompt


def test_game_theory_scenario():
    """Test a complete game theory function calling scenario."""
    # Define game decision function
    func_def = FunctionDefinition(
        name="make_decision",
        description="Make a decision in a prisoner's dilemma",
        parameters={
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
                    "description": "Confidence in the decision",
                },
            },
            "required": ["decision"],
        },
    )
    tool_def = ToolDefinition(function=func_def)

    # Create request
    request = ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(content="You're in a prisoner's dilemma. Make your decision.")],
        tools=[tool_def],
        tool_choice="auto",
    )

    # Generate prompt
    prompt = apply_emotion_to_prompt(request.messages, "happiness", request.tools)

    assert "make_decision" in prompt
    assert "prisoner's dilemma" in prompt
    assert "cooperate" in prompt
    assert "defect" in prompt

    # Test parsing a mock response
    mock_response = """{"tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "make_decision", "arguments": "{\\"decision\\": \\"cooperate\\", \\"confidence\\": 0.8}"}}]}"""

    tool_calls = parse_tool_calls_from_response(mock_response)

    assert len(tool_calls) == 1
    assert tool_calls[0].function["name"] == "make_decision"

    # Parse arguments
    args = json.loads(tool_calls[0].function["arguments"])
    assert args["decision"] == "cooperate"
    assert args["confidence"] == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
