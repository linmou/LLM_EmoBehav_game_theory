#!/usr/bin/env python3
"""
Unit tests for function calling functionality in the OpenAI server.
"""

import json

# Import the function calling components
import sys
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.append(str(Path(__file__).parent.parent))

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


class TestFunctionCallingModels:
    """Test the Pydantic models for function calling."""

    def test_function_definition_creation(self):
        """Test creating a FunctionDefinition."""
        func_def = FunctionDefinition(
            name="test_function",
            description="A test function",
            parameters={"type": "object", "properties": {"param1": {"type": "string"}}},
        )

        assert func_def.name == "test_function"
        assert func_def.description == "A test function"
        assert "param1" in func_def.parameters["properties"]

    def test_tool_definition_creation(self):
        """Test creating a ToolDefinition."""
        func_def = FunctionDefinition(name="test_func", parameters={})
        tool_def = ToolDefinition(function=func_def)

        assert tool_def.type == "function"
        assert tool_def.function.name == "test_func"

    def test_tool_call_creation(self):
        """Test creating a ToolCall."""
        tool_call = ToolCall(
            id="call_123", function={"name": "test_func", "arguments": '{"param": "value"}'}
        )

        assert tool_call.id == "call_123"
        assert tool_call.type == "function"
        assert tool_call.function["name"] == "test_func"

    def test_chat_message_with_tool_calls(self):
        """Test ChatMessage with tool_calls."""
        tool_call = ToolCall(id="call_123", function={"name": "test_func", "arguments": "{}"})

        message = ChatMessage(
            role="assistant", content="I'll call a function", tool_calls=[tool_call]
        )

        assert message.role == "assistant"
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0].id == "call_123"

    def test_chat_completion_request_with_tools(self):
        """Test ChatCompletionRequest with tools."""
        func_def = FunctionDefinition(name="test_func", parameters={})
        tool_def = ToolDefinition(function=func_def)

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(content="Hello")],
            tools=[tool_def],
            tool_choice="auto",
        )

        assert len(request.tools) == 1
        assert request.tool_choice == "auto"
        assert request.tools[0].function.name == "test_func"


class TestToolPromptFormatting:
    """Test formatting tools for model prompts."""

    def test_format_tools_for_prompt_empty(self):
        """Test formatting with no tools."""
        result = format_tools_for_prompt(None)
        assert result == ""

        result = format_tools_for_prompt([])
        assert result == ""

    def test_format_tools_for_prompt_single_tool(self):
        """Test formatting a single tool."""
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

    def test_format_tools_for_prompt_multiple_tools(self):
        """Test formatting multiple tools."""
        func1 = FunctionDefinition(name="func1", description="Function 1", parameters={})
        func2 = FunctionDefinition(name="func2", description="Function 2", parameters={})
        tools = [ToolDefinition(function=func1), ToolDefinition(function=func2)]

        result = format_tools_for_prompt(tools)

        assert "func1" in result
        assert "func2" in result
        assert "Function 1" in result
        assert "Function 2" in result


class TestToolCallParsing:
    """Test parsing tool calls from model responses."""

    def test_parse_tool_calls_from_response_json_format(self):
        """Test parsing tool calls in JSON format."""
        response_text = """I'll help you with that.

{"tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": "{\\"location\\": \\"Paris\\"}"}}]}

Let me get that information for you."""

        tool_calls = parse_tool_calls_from_response(response_text)

        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_123"
        assert tool_calls[0].function["name"] == "get_weather"
        assert "Paris" in tool_calls[0].function["arguments"]

    def test_parse_tool_calls_from_response_function_syntax(self):
        """Test parsing tool calls in function syntax format."""
        response_text = "I'll call get_weather(Paris) to get the information."

        tool_calls = parse_tool_calls_from_response(response_text)

        assert len(tool_calls) == 1
        assert tool_calls[0].function["name"] == "get_weather"
        assert "Paris" in tool_calls[0].function["arguments"]

    def test_parse_tool_calls_from_response_no_calls(self):
        """Test parsing when no tool calls are present."""
        response_text = "I can help you with that, but I don't need to call any functions."

        tool_calls = parse_tool_calls_from_response(response_text)

        assert len(tool_calls) == 0

    def test_parse_tool_calls_from_response_multiple_calls(self):
        """Test parsing multiple tool calls."""
        response_text = """{"tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "func1", "arguments": "{}"}},
            {"id": "call_2", "type": "function", "function": {"name": "func2", "arguments": "{}"}}
        ]}"""

        tool_calls = parse_tool_calls_from_response(response_text)

        assert len(tool_calls) == 2
        assert tool_calls[0].function["name"] == "func1"
        assert tool_calls[1].function["name"] == "func2"

    def test_parse_tool_calls_from_response_malformed_json(self):
        """Test parsing with malformed JSON."""
        response_text = '{"tool_calls": [{"id": "call_1", "function": {"name": "func1"'  # Missing closing braces

        tool_calls = parse_tool_calls_from_response(response_text)

        # Should handle gracefully and return empty list
        assert len(tool_calls) == 0


class TestPromptGeneration:
    """Test prompt generation with tools and emotion context."""

    def test_apply_emotion_to_prompt_without_tools(self):
        """Test prompt generation without tools."""
        messages = [ChatMessage(role="user", content="Hello, how are you?")]

        prompt = apply_emotion_to_prompt(messages, "happiness", None)

        assert "User: Hello, how are you?" in prompt
        assert "Assistant:" in prompt
        assert "tools" not in prompt.lower()

    def test_apply_emotion_to_prompt_with_tools(self):
        """Test prompt generation with tools."""
        func_def = FunctionDefinition(name="test_func", description="Test function", parameters={})
        tool_def = ToolDefinition(function=func_def)

        messages = [ChatMessage(role="user", content="Help me with something")]

        prompt = apply_emotion_to_prompt(messages, "happiness", [tool_def])

        assert "test_func" in prompt
        assert "Test function" in prompt
        assert "tool_calls" in prompt
        assert "User: Help me with something" in prompt
        assert "Assistant:" in prompt

    def test_apply_emotion_to_prompt_with_tool_calls_message(self):
        """Test prompt generation with assistant message containing tool calls."""
        tool_call = ToolCall(id="call_123", function={"name": "test_func", "arguments": "{}"})

        messages = [
            ChatMessage(role="user", content="Call a function"),
            ChatMessage(role="assistant", content="I'll call the function", tool_calls=[tool_call]),
            ChatMessage(role="tool", content="Function result", tool_call_id="call_123"),
        ]

        prompt = apply_emotion_to_prompt(messages, "happiness", None)

        assert "User: Call a function" in prompt
        assert "Assistant: I'll call the function" in prompt
        assert "Tool calls:" in prompt
        assert "Tool Result (ID: call_123): Function result" in prompt

    def test_apply_emotion_to_prompt_various_roles(self):
        """Test prompt generation with various message roles."""
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant"),
            ChatMessage(role="user", content="User message"),
            ChatMessage(role="assistant", content="Assistant response"),
            ChatMessage(role="tool", content="Tool response", tool_call_id="call_1"),
        ]

        prompt = apply_emotion_to_prompt(messages, "anger", None)

        assert "System: You are a helpful assistant" in prompt
        assert "User: User message" in prompt
        assert "Assistant: Assistant response" in prompt
        assert "Tool Result (ID: call_1): Tool response" in prompt


class TestIntegrationScenarios:
    """Test integration scenarios for function calling."""

    def test_game_theory_function_call_scenario(self):
        """Test a game theory scenario with function calls."""
        # Define a game theory decision function
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

        # Test request
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

    def test_emotion_context_preservation(self):
        """Test that emotion context is preserved with function calls."""
        func_def = FunctionDefinition(name="analyze_emotion", parameters={})
        tool_def = ToolDefinition(function=func_def)

        messages = [ChatMessage(content="How do you feel about cooperation?")]

        # Test different emotions
        for emotion in ["happiness", "anger", "sadness"]:
            prompt = apply_emotion_to_prompt(messages, emotion, [tool_def])

            assert "analyze_emotion" in prompt
            assert "How do you feel about cooperation?" in prompt
            assert "tool_calls" in prompt

    def test_multi_turn_conversation_with_tools(self):
        """Test multi-turn conversation with tool calls."""
        func_def = FunctionDefinition(name="calculate", parameters={})
        tool_def = ToolDefinition(function=func_def)

        # Simulate conversation flow
        messages = [
            ChatMessage(role="user", content="Calculate 2+2"),
            ChatMessage(
                role="assistant",
                content="I'll calculate that for you",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function={"name": "calculate", "arguments": '{"expression": "2+2"}'},
                    )
                ],
            ),
            ChatMessage(role="tool", content="4", tool_call_id="call_1"),
            ChatMessage(role="user", content="Now calculate 3+3"),
        ]

        prompt = apply_emotion_to_prompt(messages, "happiness", [tool_def])

        assert "User: Calculate 2+2" in prompt
        assert "Tool Result (ID: call_1): 4" in prompt
        assert "User: Now calculate 3+3" in prompt


@pytest.fixture
def sample_tools():
    """Fixture providing sample tools for testing."""
    return [
        ToolDefinition(
            function=FunctionDefinition(
                name="get_weather",
                description="Get weather information",
                parameters={
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            )
        ),
        ToolDefinition(
            function=FunctionDefinition(
                name="make_decision",
                description="Make a game theory decision",
                parameters={
                    "type": "object",
                    "properties": {"choice": {"type": "string", "enum": ["cooperate", "defect"]}},
                    "required": ["choice"],
                },
            )
        ),
    ]


class TestErrorHandling:
    """Test error handling in function calling."""

    def test_parse_tool_calls_with_invalid_json(self):
        """Test parsing with completely invalid JSON."""
        response_text = "This is not JSON at all! Just regular text."

        tool_calls = parse_tool_calls_from_response(response_text)
        assert len(tool_calls) == 0

    def test_parse_tool_calls_with_missing_fields(self):
        """Test parsing with missing required fields."""
        response_text = '{"tool_calls": [{"id": "call_1"}]}'  # Missing function field

        tool_calls = parse_tool_calls_from_response(response_text)
        # Should handle gracefully
        assert len(tool_calls) == 0

    def test_format_tools_with_empty_function(self):
        """Test formatting tools with empty function definition."""
        func_def = FunctionDefinition(name="", parameters={})
        tool_def = ToolDefinition(function=func_def)

        result = format_tools_for_prompt([tool_def])

        # Should still generate valid prompt
        assert "tool_calls" in result
        assert isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
