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

## Server Management

This section covers comprehensive server management using the project's dedicated tools for optimal VRAM usage and server lifecycle management.

### Recommended Management Tools

**Primary Tool: `manage_servers.py`**
- Interactive VRAM-aware server management
- Lists all running OpenAI servers with detailed information
- Safe process termination with orphaned process cleanup
- GPU memory monitoring integration

**Secondary Tool: `start_emotion_servers.sh`**
- Production-ready server startup with configuration management
- Automatic health monitoring and startup verification
- Multi-emotion server coordination (happiness + anger)
- Comprehensive logging and error handling

### Server Management Workflow

#### 1. Starting Servers

**Option A: Multiple Emotion Servers (Recommended for Production)**
```bash
# Start both happiness and anger servers with tensor parallelism
./start_emotion_servers.sh start

# Check server status
./start_emotion_servers.sh status

# View recent logs
./start_emotion_servers.sh logs
```

**Option B: Single Server (Development/Testing)**
```bash
# Activate conda environment first
conda activate llm_fresh

# Start single server
python -m openai_server \
    --model /data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct \
    --model_name "Qwen2.5-0.5B-Instruct" \
    --emotion anger \
    --port 8000 \
    --tensor_parallel_size 2
```

#### 2. Managing Running Servers

**Interactive Server Management**
```bash
# Launch interactive server manager
python manage_servers.py
```

The interactive manager provides:
- **Real-time server listing** with PID, port, model, emotion, and name
- **Selective termination** (kill specific servers by number: `1,3,5` or `1-3`)
- **Bulk operations** (`a` for all servers, `o` for orphaned processes)
- **VRAM cleanup** (`c` for complete cleanup of servers + orphaned processes)
- **GPU memory monitoring** with nvidia-smi integration

**Example Session:**
```
=================================================================
RUNNING OPENAI SERVERS
=================================================================
#  |    PID | Port | Model                | Emotion  | Name           
-----------------------------------------------------------------
1  |  12345 | 8000 | Qwen2.5-0.5B-Instruct | anger    | Qwen2.5-0.5B-Instruct
2  |  12346 | 8001 | Qwen2.5-0.5B-Instruct | happiness| Qwen2.5-0.5B-Instruct
=================================================================

⚠️  Found 3 orphaned multiprocessing processes (may be using VRAM):
   PID: 12340
   PID: 12341  
   PID: 12342

Options:
  [1-N]: Kill specific server(s) (e.g., '1', '1,3', '1-3')
  [a/all]: Kill ALL servers
  [o/orphaned]: Kill orphaned multiprocessing processes
  [c/cleanup]: Kill ALL servers AND orphaned processes
  [r/refresh]: Refresh server list
  [q/quit]: Quit

Enter your choice: c
Kill ALL 2 servers and 3 orphaned processes? (y/N): y
✓ Killed server PID 12345 (Qwen2.5-0.5B-Instruct - anger)
✓ Killed server PID 12346 (Qwen2.5-0.5B-Instruct - happiness)
✓ Killed orphaned process PID 12340
✓ Killed orphaned process PID 12341
✓ Killed orphaned process PID 12342

Killed 2/2 servers and 3/3 orphaned processes.
```

#### 3. Server Status Monitoring

**Quick Health Check**
```bash
# Check if servers are responding
curl http://localhost:8000/health
curl http://localhost:8001/health
```

**Comprehensive Status (via start script)**
```bash
./start_emotion_servers.sh status
```

Output includes:
- Server health status per port
- Current emotion configuration  
- GPU memory utilization per device
- Process information and resource usage

#### 4. GPU Memory Management

**Why Use `manage_servers.py` Instead of `pkill`:**
- **VRAM Safety**: Properly releases GPU memory allocations
- **Process Hierarchy**: Handles parent-child process relationships correctly
- **Orphan Detection**: Identifies and cleans multiprocessing remnants
- **Selective Targeting**: Avoids killing unrelated Python processes
- **Verification**: Confirms successful termination before cleanup

**Manual VRAM Monitoring:**
```bash
# Check GPU memory usage
nvidia-smi

# Monitor in real-time  
watch -n 1 nvidia-smi
```

### Advanced Configuration

#### Tensor Parallelism Setup

For multi-GPU setups, configure tensor parallelism in `start_emotion_servers.sh`:

```bash
# Edit configuration section
TENSOR_PARALLEL=2          # Number of GPUs
GPU_MEMORY_UTIL=0.7       # Memory utilization per GPU
BATCH_SIZE=32             # Inference batch size
MAX_NUM_SEQS=32          # Maximum concurrent sequences
```

#### Production Deployment

**Environment Setup:**
```bash
# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1
export MODEL_PATH=/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct
export EMOTION=anger
export TENSOR_PARALLEL_SIZE=2

# Activate conda environment
conda activate llm_fresh

# Start with production settings
./start_emotion_servers.sh start
```

**Load Balancing Multiple Instances:**
```bash
# Terminal 1: Anger server on GPU 0
CUDA_VISIBLE_DEVICES=0 python -m openai_server --port 8000 --emotion anger

# Terminal 2: Happiness server on GPU 1  
CUDA_VISIBLE_DEVICES=1 python -m openai_server --port 8001 --emotion happiness

# Terminal 3: Additional emotions
CUDA_VISIBLE_DEVICES=0 python -m openai_server --port 8002 --emotion sadness
```

### Troubleshooting Server Management

#### Common Issues

**1. VRAM Not Released After Server Shutdown**
```bash
# Use proper cleanup instead of pkill
python manage_servers.py
# Select 'c' for complete cleanup

# Verify VRAM release
nvidia-smi
```

**2. Port Already in Use**
```bash
# Find process using port
lsof -i :8000

# Use manager to kill specific servers
python manage_servers.py
```

**3. Orphaned Multiprocessing Processes**
```bash
# Automatic detection and cleanup
python manage_servers.py
# Select 'o' to kill orphaned processes only
```

**4. Server Startup Failures**
```bash
# Check startup logs
./start_emotion_servers.sh logs

# Verify prerequisites
./start_emotion_servers.sh help

# Check model path and permissions
ls -la /data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct
```

#### Debug Commands

```bash
# List all Python processes
ps aux | grep python

# Check OpenAI server processes specifically  
ps aux | grep openai_server

# Monitor GPU processes
nvidia-smi pmon

# Check port usage
netstat -tulpn | grep :800
```

### Server Management Best Practices

1. **Always use `manage_servers.py`** for process termination instead of `pkill`
2. **Monitor VRAM usage** regularly during development with `nvidia-smi`
3. **Clean orphaned processes** periodically to prevent VRAM leaks
4. **Use tensor parallelism** efficiently based on available GPU memory
5. **Check server health** before running experiments: `curl localhost:8000/health`
6. **Activate conda environment** before starting servers: `conda activate llm_fresh`
7. **Keep logs** for debugging: servers write to `logs/` directory
8. **Graceful shutdowns** prevent corrupted model states and VRAM issues

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