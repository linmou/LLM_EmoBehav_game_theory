# Function Calling Support in OpenAI Server

## Overview

The OpenAI-compatible emotion-controlled server now supports **function calling** (tool calls), allowing the model to intelligently call predefined functions based on user requests while maintaining emotion-based neural manipulation.

## Key Features

### ✅ **Implemented**
- **OpenAI-compatible function calling API**
- **Tool definitions** with JSON schema validation
- **Automatic function call parsing** from model responses
- **Proper response formatting** with `tool_calls` and `finish_reason`
- **Multi-function support** in single response
- **Tool choice control** (`auto`, `none`, specific function name)
- **Integration with emotion control** - functions called under emotional influence

### 🔧 **API Components**

#### Request Models
```python
# Tool definition
{
    "type": "function",
    "function": {
        "name": "function_name",
        "description": "What the function does",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "Parameter description"}
            },
            "required": ["param1"]
        }
    }
}

# Chat completion request with tools
{
    "model": "Qwen2.5-0.5B-Instruct",
    "messages": [...],
    "tools": [tool_definitions],
    "tool_choice": "auto",  # or "none" or specific function name
    "max_tokens": 150
}
```

#### Response Models
```python
# Response with function call
{
    "choices": [{
        "message": {
            "role": "assistant",
            "content": "I'll help you with that...",
            "tool_calls": [{
                "id": "call_abc123",
                "type": "function", 
                "function": {
                    "name": "function_name",
                    "arguments": "{\"param1\": \"value\"}"
                }
            }]
        },
        "finish_reason": "tool_calls"  # or "stop"
    }]
}
```

## Usage Examples

### 1. Basic Function Call

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        }
    }
}]

response = client.chat.completions.create(
    model="Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    tool_choice="auto"
)

# Check for function calls
if response.choices[0].message.tool_calls:
    for call in response.choices[0].message.tool_calls:
        print(f"Function: {call.function.name}")
        print(f"Arguments: {call.function.arguments}")
```

### 2. Game Theory Function

```python
tools = [{
    "type": "function",
    "function": {
        "name": "make_game_decision",
        "description": "Make decision in game theory scenario",
        "parameters": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["cooperate", "defect"],
                    "description": "Decision to make"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Confidence in decision"
                }
            },
            "required": ["decision"]
        }
    }
}]

# Happiness server (port 8000) vs Anger server (port 8001)
happiness_client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
anger_client = OpenAI(base_url="http://localhost:8001/v1", api_key="dummy")

# Compare emotional responses
for name, client in [("Happiness", happiness_client), ("Anger", anger_client)]:
    response = client.chat.completions.create(
        model="Qwen2.5-0.5B-Instruct",
        messages=[{"role": "user", "content": "You're in a prisoner's dilemma. Make your decision."}],
        tools=tools
    )
    print(f"{name} emotion decision: {response.choices[0].message.tool_calls}")
```

## Function Parsing Logic

The server uses multiple parsing strategies:

1. **Structured JSON** - Looks for `{"tool_calls": [...]}` format
2. **Function syntax** - Parses `function_name(arguments)` patterns  
3. **Fallback parsing** - Handles various response formats

## Integration with Emotion Control

Function calling works seamlessly with the emotion manipulation system:

- **Happiness server** (port 8000) may show more cooperative function calls
- **Anger server** (port 8001) may show more aggressive/competitive function calls
- **Tool choice influenced by emotion** - same tools, different usage patterns

## Testing

Run the demo scripts:

```bash
# Test function calling capabilities
python openai_server/examples/function_calling_demo.py

# Simple usage example
python openai_server/examples/simple_function_example.py
```

## Research Applications

This enables new research possibilities:

1. **Emotion-influenced tool usage patterns**
2. **Strategic function calling in games**
3. **Dynamic scenario generation with functions**
4. **Real-time analysis during experiments**

## Limitations

- **Model dependent** - Qwen2.5-0.5B may need prompt engineering for reliable function calling
- **No execution framework** - Functions are called but not executed (returns tool_calls only)
- **Streaming limitations** - Complex with multi-turn function calls
- **Parsing robustness** - May miss complex or malformed function calls

## Future Enhancements

- **Function execution engine** - Actually execute registered functions
- **Better model fine-tuning** for function calling
- **Streaming support** for function calls
- **Function call validation** and error handling
- **Dynamic function registration** at runtime

---

**The emotion-controlled OpenAI server now supports function calling while maintaining all existing neural manipulation capabilities!**