# OpenAI-Compatible Server for RepControlVLLMHook (LEGACY)

> **⚠️ DEPRECATED**: This documentation is for the legacy `init_openai_server.py` structure. 
> 
> **New documentation**: Please see [`/docs/code_readme/openai_server/README.md`](openai_server/README.md) and [`/openai_server/README.md`](../../openai_server/README.md) for the current modular structure.

This documentation describes how to set up and use the OpenAI-compatible server that integrates RepControlVLLMHook for emotion-controlled language model generation.

## Overview

The `init_openai_server.py` script creates a FastAPI-based server that exposes OpenAI-compatible API endpoints while applying emotion control through RepControlVLLMHook. This allows you to use standard OpenAI client libraries while getting emotion-influenced responses from your language model.

## Features

- **OpenAI-Compatible API**: Standard `/v1/chat/completions` and `/v1/models` endpoints
- **Emotion Control**: Pre-configured emotion activation vectors applied to all generations
- **vLLM Backend**: High-performance inference with tensor parallel support
- **Easy Integration**: Works with existing OpenAI client code with minimal changes

## Installation

Ensure you have the required dependencies installed:

```bash
pip install fastapi uvicorn vllm transformers torch
```

## Quick Start

### 1. Start the Server

```bash
python init_openai_server.py \
  --model /path/to/your/model \
  --model_name qwen3-0.5B-anger \
  --emotion anger \
  --port 8000 \
  --api_key token-abc123
```

### 2. Use with OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123"
)

completion = client.chat.completions.create(
    model="qwen3-0.5B-anger",
    messages=[
        {"role": "user", "content": "Hello! How are you feeling today?"}
    ]
)

print(completion.choices[0].message.content)
```

## Command Line Arguments

### Required Arguments

- `--model`: Path to the model directory or Hugging Face model name
- `--model_name`: Model name to be used in API responses

### Optional Arguments

- `--emotion`: Emotion to activate (default: "anger")
  - Valid emotions: anger, happiness, sadness, disgust, fear, surprise
- `--host`: Host to bind to (default: "localhost")
- `--port`: Port to bind to (default: 8000)
- `--api_key`: API key for authentication (optional, for display purposes)
- `--url`: Base URL for the server (for display purposes)
- `--tensor_parallel_size`: Tensor parallel size for vLLM (default: 1)

## API Endpoints

### GET /v1/models

Lists available models.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3-0.5B-anger",
      "object": "model",
      "created": 1699000000,
      "owned_by": "local"
    }
  ]
}
```

### POST /v1/chat/completions

Creates a chat completion with emotion control.

**Request:**
```json
{
  "model": "qwen3-0.5B-anger",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 100,
  "temperature": 0.0
}
```

**Response:**
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1699000000,
  "model": "qwen3-0.5B-anger",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I'm feeling quite intense today..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 15,
    "total_tokens": 25
  }
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "current_emotion": "anger",
  "server_time": "2024-01-01T12:00:00"
}
```

## Configuration

### Emotion Configuration

The server loads emotion activation vectors based on the specified emotion. These vectors are pre-computed from emotion-related training data and applied to the middle layers of the model during generation.

Supported emotions:
- `anger`: Activates anger-related neural patterns
- `happiness`: Activates happiness-related neural patterns
- `sadness`: Activates sadness-related neural patterns
- `disgust`: Activates disgust-related neural patterns
- `fear`: Activates fear-related neural patterns
- `surprise`: Activates surprise-related neural patterns

### Model Configuration

The server automatically detects model layers and applies emotion control to the middle third of the layers, which has been found to be most effective for representation control.

## Testing

Use the provided test script to verify the server functionality:

```bash
# Test with automatic server startup
python test_openai_server.py --start_server

# Test against running server
python test_openai_server.py --test_only
```

The test script will:
1. Check server health
2. Test the models endpoint
3. Test chat completions
4. Compare emotion-controlled responses

## Examples

### Basic Usage

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123"
)

# Single completion
response = client.chat.completions.create(
    model="qwen3-0.5B-anger",
    messages=[{"role": "user", "content": "Tell me about your day."}],
    max_tokens=50
)

print(response.choices[0].message.content)
```

### Batch Processing

```python
prompts = [
    "How do you feel about waiting in line?",
    "Describe your reaction to unfair treatment.",
    "What's your response to criticism?"
]

for prompt in prompts:
    response = client.chat.completions.create(
        model="qwen3-0.5B-anger",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=30,
        temperature=0.0  # For consistent responses
    )
    
    print(f"Prompt: {prompt}")
    print(f"Response: {response.choices[0].message.content}")
    print("-" * 50)
```

### Async Usage

```python
import asyncio
from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123"
    )
    
    response = await client.chat.completions.create(
        model="qwen3-0.5B-anger",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    print(response.choices[0].message.content)

asyncio.run(main())
```


## Integration with Existing Code

The server is designed to be a drop-in replacement for OpenAI API endpoints. Simply change the `base_url` in your existing OpenAI client code:

```python
# Before
client = OpenAI(api_key="your-openai-key")

# After
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123"
)
```

All other code remains the same, but now responses will be influenced by the configured emotion.

## Advanced Usage

### Multiple Emotion Servers

You can run multiple servers with different emotions on different ports:

```bash
# Anger server on port 8000
python init_openai_server.py --emotion anger --port 8000 --model_name qwen-anger

# Happiness server on port 8001  
python init_openai_server.py --emotion happiness --port 8001 --model_name qwen-happiness
```

### Production Deployment

For production use, consider:

1. Using a proper WSGI server like Gunicorn
2. Adding authentication middleware
3. Implementing rate limiting
4. Adding request/response logging
5. Using environment variables for configuration

Example with Gunicorn:

```bash
gunicorn -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 init_openai_server:app
```

## Limitations

- Single emotion per server instance
- No streaming support (yet)
- Limited to chat completions (no text completions)
- Emotion intensity is fixed at initialization

## Framework Integration

Our OpenAI-compatible server can be integrated with popular AI frameworks like LangGraph and AutoGen (AG2). This allows you to leverage emotion-controlled models within complex agentic workflows.

### Integrating with LangGraph

**Key Concept**: LangGraph uses its own message objects (`HumanMessage`, `AIMessage`). You must convert these into the standard OpenAI dictionary format (`{"role": "user", "content": "..."}`) before sending them to the server.

**Example Node Implementation**:
```python
from langchain_core.messages import HumanMessage, AIMessage

def chatbot_node(state):
    # 1. Convert messages to OpenAI format
    openai_messages = []
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            openai_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            openai_messages.append({"role": "assistant", "content": msg.content})

    # 2. Call the server with the converted messages
    response = client.chat.completions.create(
        model="YOUR_MODEL_NAME",
        messages=openai_messages
    )
    
    # 3. Convert the response back to a LangChain message
    ai_response = AIMessage(content=response.choices[0].message.content)
    return {"messages": [ai_response]}
```

For a complete, runnable example of LangGraph integration, please see the `test_langgraph_basic_fixed` function in `run_complete_compatibility_test.py`.

### Integrating with AutoGen (AG2)

**Key Concept**: To use the server with AutoGen, you need to provide a custom configuration list that points to your local server's endpoint.

**Example Configuration**:
```python
import autogen

# Point the config_list to your local server
config_list = [{
    "model": "qwen3-0.5B-anger", # Your model name
    "api_key": "token-abc123",    # Your API key
    "base_url": "http://localhost:8000/v1",
    "api_type": "openai"
}]

# Use this config list when creating your agents
assistant = autogen.AssistantAgent(
    "assistant",
    llm_config={"config_list": config_list}
)
```

For a complete, runnable example of AutoGen integration, please see the `test_ag2_basic` function in `run_complete_compatibility_test.py`.

## Contributing

To extend the server functionality:

1. Add new endpoints to the FastAPI app
2. Implement additional emotion control features
3. Add support for dynamic emotion switching
4. Implement streaming responses 