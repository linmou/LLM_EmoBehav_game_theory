# OpenAI Server Test Suite

This directory contains comprehensive tests for the OpenAI-compatible server with RepControlVLLMHook integration.

## Test Structure

### Unit Tests (`test_unit_server.py`)
- **Pydantic Models**: Validation of request/response models
- **Streaming Functions**: Async streaming response generation
- **Error Handling**: Invalid request handling
- **Component Testing**: Individual component functionality

### Integration Tests (`test_integration_server.py`)
- **End-to-End Testing**: Full server functionality
- **API Endpoints**: `/v1/models`, `/v1/chat/completions`, `/health`
- **Emotion Switching**: Testing different emotion activations
- **Concurrent Requests**: Multi-client handling
- **Long Context**: Context window handling

### Existing Tests
- `test_openai_server.py`: Basic server functionality
- `test_integrated_openai_server.py`: Integration scenarios
- `test_server_connectivity.py`: Connection testing
- `test_server_management.py`: Server lifecycle management
- `test_manage_servers.py`: Multi-server management

## Running Tests

### Quick Test (Unit tests only, no GPU required)
```bash
make test-quick
```

### All Tests
```bash
make test
```

### Specific Test Types
```bash
# Unit tests only
make test-unit

# Integration tests (requires GPU)
make test-integration

# Coverage report
make test-coverage
```

### Using pytest directly
```bash
cd openai_server/tests

# Run all tests
pytest -v

# Run specific test file
pytest test_unit_server.py -v

# Run with coverage
pytest --cov=openai_server --cov-report=html

# Skip GPU tests
pytest -m "not gpu"
```

## Test Configuration

### Environment Variables
- `TEST_MODEL_PATH`: Override default model path
- `VLLM_GPU_MEMORY_UTILIZATION`: Set GPU memory usage (default: 0.5 for tests)

### Pytest Markers
- `@pytest.mark.gpu`: Tests requiring GPU
- `@pytest.mark.slow`: Slow-running tests
- `@pytest.mark.integration`: Integration tests

## Writing New Tests

### Unit Test Example
```python
def test_new_feature(self):
    """Test description"""
    # Arrange
    request = ChatCompletionRequest(...)
    
    # Act
    result = process_request(request)
    
    # Assert
    assert result.status == "success"
```

### Integration Test Example
```python
@pytest.mark.gpu
def test_emotion_response(self):
    """Test emotion-controlled responses"""
    self.start_server(emotion="anger")
    client = OpenAI(base_url=self.base_url)
    
    response = client.chat.completions.create(...)
    assert "anger" in analyze_sentiment(response)
```

## Continuous Integration

Tests are automatically run on:
- Push to `openai_server/**` paths
- Pull requests affecting server code
- Manual workflow dispatch

### CI Stages
1. **Lint & Format**: Code style checks
2. **Unit Tests**: Fast tests on multiple Python versions
3. **Integration Tests**: GPU tests on main branch
4. **Documentation**: Auto-generated docs

## Troubleshooting

### Common Issues

1. **GPU Memory Error**
   ```bash
   export VLLM_GPU_MEMORY_UTILIZATION=0.3
   make test-integration
   ```

2. **Port Already in Use**
   ```bash
   # Use different port
   TEST_PORT=8766 pytest test_integration_server.py
   ```

3. **Model Not Found**
   ```bash
   export TEST_MODEL_PATH=/path/to/your/model
   make test
   ```

### Debug Mode
```bash
# Run with debug output
pytest -v -s --log-cli-level=DEBUG
```

## Coverage Goals

Target coverage: 80%+ for core functionality

Key areas:
- [ ] Request validation
- [ ] Response generation
- [ ] Emotion vector application
- [ ] Error handling
- [ ] Concurrent request handling
- [ ] Streaming responses

## Contributing

1. Write tests for new features
2. Ensure all tests pass: `make test`
3. Check coverage: `make test-coverage`
4. Run linter: `make lint`
5. Format code: `make format`