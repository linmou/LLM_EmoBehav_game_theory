# CLAUDE.md - OpenAI Server Module

This file provides guidance to Claude Code when working with the OpenAI-compatible server module.

## Module Overview

The `openai_server` module provides an OpenAI-compatible API server with neural manipulation capabilities for emotion-based LLM research. It integrates vLLM for efficient inference with custom hooks for real-time neural interventions.

## Quick Start

### Starting the Server

```bash
# Ensure conda environment is active
conda activate llm_fresh

# Start with emotion control (BOTH --model and --model_name are REQUIRED)
python -m openai_server \
    --model /data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct \
    --model_name Qwen2.5-0.5B-anger \
    --emotion anger \
    --port 8000

# Start in background with logging
nohup python -m openai_server \
    --model /data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct \
    --model_name Qwen2.5-0.5B-happiness \
    --emotion happiness > server.log 2>&1 &
```

### Server Management

```bash
# View running servers
python openai_server/manage_servers.py

# Kill all servers and clean up
echo -e "c\ny" | python openai_server/manage_servers.py

# Check server health
curl http://localhost:8000/health | python -m json.tool

# Test the API
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-0.5B-anger",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100
  }'
```

## Architecture

### Core Components

**server.py**
- FastAPI application with OpenAI-compatible endpoints
- Integration with vLLM AsyncLLMEngine
- Real-time emotion vector application via RepControlVLLMHook
- Health monitoring and metrics collection

**async_vllm_wrapper.py**
- Wrapper around vLLM's AsyncLLMEngine
- Implements graceful degradation for overload scenarios
- Thread pool management with timeout handling
- Request queuing and rejection strategies

**adaptive_processor.py** (Production Version)
- Advanced request processing with automatic recovery
- Abandoned thread detection and cleanup
- Dynamic resource management
- Circuit breaker pattern implementation

**health_monitor.py**
- Real-time server health tracking
- Automatic issue detection and recovery
- Performance metrics collection
- Alert thresholds for critical conditions

**request_queue_manager.py**
- Intelligent request queuing with prioritization
- Backpressure handling
- Queue size management
- Request rejection with appropriate error codes

## Configuration Parameters

### Required Arguments
- `--model`: Path to the model directory (e.g., `/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct`)
- `--model_name`: API endpoint name (e.g., `Qwen2.5-0.5B-anger`)

### Emotion Control
- `--emotion`: Emotion to activate (anger, happiness, sadness, disgust, fear, surprise)
- `--emotion_scale`: Emotion intensity multiplier (default: 1.0)

### Server Configuration
- `--host`: Host to bind to (default: "0.0.0.0")
- `--port`: Port to run on (default: 8000)
- `--cors_origins`: CORS allowed origins (default: ["*"])

### vLLM Configuration
- `--gpu_memory_utilization`: GPU memory usage fraction (default: 0.90)
- `--max_num_seqs`: Max concurrent sequences (default: 64)
- `--max_model_len`: Max model context length (default: 4096)
- `--tensor_parallel_size`: Number of GPUs for tensor parallelism (default: 1)

### Graceful Degradation Configuration
- `--request_timeout`: Request processing timeout in seconds (default: 60)
- `--max_queue_size`: Maximum requests in queue (default: 50)
- `--max_concurrent_requests`: Max concurrent processing (default: 3)
- `--queue_rejection_threshold`: Start rejecting at this queue fullness (default: 0.8)
- `--reset_interval`: Abandoned thread cleanup interval (default: 300)
- `--vllm_rejection_threshold`: Start probabilistic rejection (default: 0.7)

## Testing

### Unit Tests
```bash
# Test OpenAI API compatibility
python -m openai_server.tests.test_openai_server

# Test integrated server with emotion control
python -m openai_server.tests.test_integrated_openai_server

# Test graceful degradation
python -m openai_server.tests.test_graceful_degradation
```

### Stress Testing
```bash
# Run comprehensive stress test suite
python openai_server/run_stress_tests.py

# Run graceful degradation test suite
python openai_server/tests/graceful_test_suite.py

# Test long input handling
python openai_server/tests/test_long_input_handling.py
```

### Load Testing
```bash
# Simple load test
python simple_graceful_test.py

# Intensive load test with monitoring
python intensive_graceful_test.py
```

## Common Issues & Solutions

### 1. Missing --model_name Argument
**Error**: "error: the following arguments are required: --model_name"
**Solution**: Always provide BOTH `--model` and `--model_name` arguments

### 2. Conda Activation in Scripts
**Error**: "CommandNotFoundError: Your shell has not been properly configured"
**Solution**: Use bash wrapper:
```bash
bash -c "source /usr/local/anaconda3/etc/profile.d/conda.sh && \
    conda activate llm_fresh && \
    python -m openai_server --model ... --model_name ..."
```

### 3. Server at Capacity
**Error**: "Server at capacity due to ongoing requests"
**Cause**: Too many abandoned threads from timeouts
**Solutions**:
- Restart the server to clear abandoned threads
- Use `production_async_vllm_wrapper.py` for automatic recovery
- Adjust `--reset_interval` for more frequent cleanup

### 4. Port Already in Use
**Error**: "Address already in use"
**Solution**: 
```bash
# Find and kill existing server
python openai_server/manage_servers.py
# Select 'c' to clean all servers
```

### 5. GPU Memory Issues
**Error**: "CUDA out of memory"
**Solutions**:
- Reduce `--gpu_memory_utilization` (e.g., 0.8)
- Reduce `--max_num_seqs` (e.g., 32)
- Use smaller model or enable quantization

## Development Guidelines

### Adding New Features
1. Maintain OpenAI API compatibility
2. Update health checks if adding new components
3. Add appropriate logging for debugging
4. Include unit tests for new functionality
5. Update stress tests if changing request handling

### Code Organization
```
openai_server/
├── __init__.py          # Package initialization
├── __main__.py          # Module entry point
├── server.py            # Main FastAPI application
├── async_vllm_wrapper.py      # vLLM integration
├── adaptive_processor.py      # Production request processor
├── health_monitor.py          # Health monitoring
├── request_queue_manager.py   # Queue management
├── circuit_breaker.py         # Circuit breaker pattern
├── manage_servers.py          # Server management utility
├── tests/                     # Test suite
│   ├── test_openai_server.py
│   ├── test_integrated_openai_server.py
│   ├── graceful_test_suite.py
│   └── stress_test_suite.py
└── examples/                  # Usage examples
    └── long_input_demo.py
```

### Logging

The server uses Python's logging module with the following loggers:
- `openai_server`: Main server operations
- `openai_server.vllm`: vLLM integration logs
- `openai_server.health`: Health monitoring logs
- `openai_server.queue`: Queue management logs

View logs:
```bash
# Real-time log monitoring
tail -f server.log

# Filter for errors
grep ERROR server.log

# Filter for specific component
grep "openai_server.health" server.log
```

## Integration with Main Project

### Using with Experiments
```python
import openai

# Configure client to use local server
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Not used but required by client
)

# Make requests with emotion-controlled model
response = client.chat.completions.create(
    model="Qwen2.5-0.5B-anger",  # Must match --model_name
    messages=[{"role": "user", "content": "Make a decision"}],
    max_tokens=100
)
```

### Neural Hook Integration
The server automatically applies emotion vectors using:
- `RepControlVLLMHook` for emotion activation
- Vectors loaded from `/emotion_control_vectors/`
- Real-time intervention during generation

## Performance Optimization

### Recommended Production Settings
```bash
python -m openai_server \
    --model /path/to/model \
    --model_name model-emotion \
    --emotion anger \
    --gpu_memory_utilization 0.85 \
    --max_num_seqs 48 \
    --max_concurrent_requests 5 \
    --request_timeout 45 \
    --reset_interval 180
```

### Monitoring Performance
```bash
# Check server metrics
curl http://localhost:8000/metrics

# Monitor GPU usage
nvidia-smi -l 1

# Check request latencies in logs
grep "Request completed" server.log | tail -20
```

## Important Notes

1. **Always use manage_servers.py** for server management - don't use `pkill`
2. **Both --model and --model_name are required** - this is a common error
3. **Graceful degradation is working** when you see capacity errors during heavy load
4. **Use production wrappers** for long-running deployments
5. **Monitor health endpoint** regularly in production environments