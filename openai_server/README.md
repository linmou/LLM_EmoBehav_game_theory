# OpenAI-Compatible Server Module

This module provides an OpenAI-compatible FastAPI server that integrates `RepControlVLLMHook` for emotion-controlled language model generation through standard OpenAI client interfaces.

## Overview

The server enables researchers to study emotional behavior in Large Language Models by:
- Providing full OpenAI API compatibility for easy integration
- Incorporating neural manipulation through `RepControlVLLMHook`
- Supporting real-time emotion activation during inference
- Offering comprehensive testing and monitoring capabilities

## Quick Start

### Basic Usage

```bash
# Run the server with emotion control
python -m openai_server --model /path/to/model --model_name "MyModel" --emotion anger

# Example with specific parameters
python -m openai_server \
    --model /data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct \
    --model_name "Qwen2.5-0.5B-Instruct" \
    --emotion anger \
    --host localhost \
    --port 8000 \
    --tensor_parallel_size 1
```

### API Usage

Once the server is running, you can use any OpenAI-compatible client:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Not required for local server
)

response = client.chat.completions.create(
    model="Qwen2.5-0.5B-Instruct",
    messages=[
        {"role": "user", "content": "Hello! How are you feeling today?"}
    ],
    max_tokens=100,
    temperature=0.7
)

print(response.choices[0].message.content)
```

## Architecture

### Core Components

- **`server.py`**: Main FastAPI server implementation with OpenAI API endpoints
- **`__main__.py`**: Module entry point enabling `python -m openai_server` usage
- **`tests/`**: Comprehensive test suite for server functionality

### Key Features

1. **OpenAI API Compatibility**
   - `/v1/chat/completions` - Chat completion endpoint with emotion control
   - `/v1/models` - List available models
   - `/health` - Server health monitoring
   - Streaming and non-streaming responses
   - CORS support for web applications

2. **Emotion Control Integration**
   - Automatic emotion activation vector loading
   - Real-time neural intervention during inference
   - Support for all emotion types: anger, happiness, sadness, disgust, fear, surprise
   - Configurable activation intensity

3. **Advanced Configuration**
   - Tensor parallel processing support
   - GPU memory optimization
   - Flexible model and tokenizer loading
   - LangGraph and AutoGen compatibility

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Path to the model directory | Required |
| `--model_name` | Model name for API identification | Required |
| `--emotion` | Emotion to activate (anger, happiness, etc.) | anger |
| `--host` | Server host address | localhost |
| `--port` | Server port number | 8000 |
| `--tensor_parallel_size` | Number of GPUs for tensor parallelism | 1 |
| `--api_key` | API key for authentication (optional) | None |
| `--url` | Base URL for display purposes | None |

## Supported Emotions

The server supports the following emotions from the project's emotion classification system:

- **anger**: Aggressive, confrontational responses
- **happiness**: Positive, optimistic responses  
- **sadness**: Melancholic, pessimistic responses
- **disgust**: Disapproving, rejecting responses
- **fear**: Cautious, anxious responses
- **surprise**: Reactive, unexpected responses

## Testing

### Run All Tests

```bash
# New modular approach
python -m openai_server.tests.test_openai_server
python -m openai_server.tests.test_integrated_openai_server

# Backward compatibility (deprecated)
python test_openai_server.py
python test_integrated_openai_server.py
```

### Test Categories

1. **Basic Functionality Tests** (`test_openai_server.py`)
   - Server startup and initialization
   - Model loading and configuration
   - API endpoint responses
   - Emotion activation verification

2. **Integration Tests** (`test_integrated_openai_server.py`)
   - Full subprocess lifecycle management
   - Concurrent request handling
   - Parameter validation
   - Error condition handling
   - Health monitoring

## Integration Examples

### LangGraph Integration

```python
from langgraph import Agent
import openai

# Configure LangGraph to use the emotion-controlled server
client = openai.OpenAI(base_url="http://localhost:8000/v1")
agent = Agent(client=client)

# LangGraph will automatically use emotion-controlled responses
result = agent.run("Analyze this situation and provide recommendations")
```

### AutoGen Integration

```python
import autogen

config_list = [{
    "model": "Qwen2.5-0.5B-Instruct",
    "base_url": "http://localhost:8000/v1",
    "api_key": "dummy"
}]

assistant = autogen.AssistantAgent(
    name="emotional_assistant",
    llm_config={"config_list": config_list}
)

# AutoGen agents will exhibit the configured emotional behavior
user_proxy = autogen.UserProxyAgent(name="user")
user_proxy.initiate_chat(assistant, message="Help me solve this problem")
```

## API Reference

### Chat Completions Endpoint

**POST** `/v1/chat/completions`

Creates a chat completion with emotion control applied.

**Request Body:**
```json
{
    "model": "string",
    "messages": [
        {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 100,
    "temperature": 0.7,
    "top_p": 1.0,
    "stream": false,
    "stop": null
}
```

**Response:**
```json
{
    "id": "chatcmpl-...",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "Qwen2.5-0.5B-Instruct",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "Hello! I'm feeling quite energetic today..."
        },
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30
    }
}
```

### Models Endpoint

**GET** `/v1/models`

Lists available models.

**Response:**
```json
{
    "object": "list",
    "data": [{
        "id": "Qwen2.5-0.5B-Instruct",
        "object": "model",
        "created": 1234567890,
        "owned_by": "local"
    }]
}
```

### Health Endpoint

**GET** `/health`

Server health check with emotion status.

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "current_emotion": "anger",
    "server_time": "2025-06-25T12:34:56"
}
```

## Production Deployment

### Docker Configuration

```dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Install dependencies
RUN pip install -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Run server
CMD ["python", "-m", "openai_server", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

```bash
export CUDA_VISIBLE_DEVICES=0,1  # GPU selection
export MODEL_PATH=/path/to/model
export EMOTION=anger
export TENSOR_PARALLEL_SIZE=2
```

### Load Balancing

For high-throughput scenarios, run multiple server instances:

```bash
# Server 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 python -m openai_server --port 8000 --emotion anger

# Server 2 (GPU 1)  
CUDA_VISIBLE_DEVICES=1 python -m openai_server --port 8001 --emotion happiness
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```
   Error: Failed to load model
   Solution: Ensure model path exists and has proper permissions
   ```

2. **CUDA Memory Issues**
   ```
   Error: CUDA out of memory
   Solution: Reduce tensor_parallel_size or gpu_memory_utilization
   ```

3. **Emotion Vector Loading Failed**
   ```
   Error: Emotion 'xyz' not found
   Solution: Use valid emotions: anger, happiness, sadness, disgust, fear, surprise
   ```

### Debug Mode

Enable detailed logging:

```bash
python -m opened_server --model /path/to/model --model_name MyModel --emotion anger --log-level DEBUG
```

## Migration Guide

### From Legacy Structure

If you were using the old file structure:

**Old:**
```bash
python init_openai_server.py --model /path --emotion anger
python test_openai_server.py
```

**New:**
```bash
python -m openai_server --model /path --emotion anger
python -m openai_server.tests.test_openai_server
```

The old commands still work but will show deprecation warnings.

## Contributing

When contributing to this module:

1. **Add tests** for new functionality in `tests/`
2. **Update documentation** for API changes
3. **Follow code style** consistent with existing patterns
4. **Test compatibility** with both modular and legacy usage

## Related Documentation

- [RepControlVLLMHook Implementation](../neuro_manipulation/repe/README_rep_control_vllm_hook.md)
- [Experiment Configuration](../neuro_manipulation/configs/experiment_config.py)
- [Game Theory Integration](../games/README.md)
- [Model Layer Detection](../neuro_manipulation/model_layer_detector.py)