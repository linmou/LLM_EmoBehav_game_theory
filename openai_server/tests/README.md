# OpenAI Server Test Suite - Complete Guide

This directory contains comprehensive tests for the OpenAI-compatible server with RepControlVLLMHook integration, including unit tests, integration tests, stress testing, and function calling validation.

## 📋 Test Structure Overview

### **Core Test Categories**

1. **Unit Tests** - Fast, isolated component testing (no GPU/server required)
2. **Integration Tests** - End-to-end API testing (requires running server)
3. **Stress Tests** - Performance and stability under load
4. **Function Calling Tests** - Tool/function calling feature validation
5. **Long Input Tests** - Context window and performance testing

---

## 🧪 Unit Tests (`test_unit_server.py`)

**Purpose**: Test individual components without external dependencies

**Coverage**:
- **Pydantic Models**: Request/response validation 
- **Streaming Functions**: Async streaming response generation
- **Error Handling**: Invalid request scenarios
- **Component Testing**: Individual function validation

**Running**:
```bash
# Quick unit tests (recommended for development)
make test-unit
# or
pytest tests/test_unit_server.py -v
```

---

## 🔗 Integration Tests (`test_integration_server.py`)

**Purpose**: Test complete server functionality with live API endpoints

**Coverage**:
- **End-to-End Testing**: Full server workflows
- **API Endpoints**: `/v1/models`, `/v1/chat/completions`, `/health`
- **Emotion Switching**: Different emotion activation testing
- **Concurrent Requests**: Multi-client handling
- **Long Context**: Context window behavior

**Running**:
```bash
# Integration tests (requires GPU and model)
make test-integration
# or
pytest tests/test_integration_server.py -v
```

---

## 🔥 Stress Tests (`stress_test_suite.py`)

**Purpose**: Identify performance bottlenecks and stability issues under load

### Test Categories

#### 1. **Server Health Baseline**
- Basic connectivity and response time
- Resource utilization baseline
- Single request performance

#### 2. **Concurrent Load Testing**
- 2, 5, 10, 15, 20+ simultaneous requests
- Success rate monitoring
- Response time degradation detection

#### 3. **Progressive Context Length**
- Input lengths from 100 to 50,000+ characters
- Context window limit discovery
- Memory pressure testing

#### 4. **Rapid Fire Requests**
- Burst patterns (20 requests in quick succession)
- Sustained load (1 request/second for 60 seconds)
- Queue overflow scenarios

#### 5. **Mixed Load Patterns**
- Concurrent short and long requests
- Different content types (code, structured data, text)
- Resource conflict detection

#### 6. **Resource Exhaustion**
- GPU memory pressure testing
- Connection limit testing
- CPU and memory monitoring

#### 7. **Hang Detection**
- Extremely long context scenarios
- Malformed request handling
- Streaming response monitoring
- Deadlock scenario testing

#### 8. **Recovery Testing**
- Server behavior after timeouts
- Recovery from error conditions
- Health check responsiveness

### Running Stress Tests

```bash
# Run comprehensive stress tests
python run_stress_tests.py

# Quick test (faster, less intensive)
python run_stress_tests.py --quick

# Test specific server
python run_stress_tests.py --server-url http://localhost:8001/v1 --model-name custom-model
```

### Stress Test Configuration

```bash
# Server Configuration
--server-url http://localhost:8000/v1    # Server endpoint
--model-name Qwen3-14B-anger             # Model to test

# Test Intensity
--max-concurrent 20                      # Max simultaneous requests
--hang-timeout 300                       # Hang detection timeout (seconds)

# Features
--no-gpu-monitoring                      # Disable GPU monitoring
--no-recovery-tests                      # Skip recovery scenarios
--quick                                  # Fast test mode

# Output
--output-dir stress_test_results         # Results directory
--verbose                                # Detailed output
```

### Interpreting Stress Test Results

#### Success Rates
- **90%+**: Excellent performance
- **70-89%**: Acceptable performance
- **<70%**: Performance issues require attention

#### Response Times
- **<5s**: Good performance
- **5-15s**: Acceptable for complex requests
- **>15s**: Potential optimization needed

#### Hang Detection
- **0 hangs**: Server operating normally
- **1-2 hangs**: Minor issues, monitor closely
- **3+ hangs**: Significant stability problems

---

## 🛠️ Function Calling Tests

**Purpose**: Comprehensive testing of function/tool calling capabilities

### Test Files

#### 1. **`test_function_calling.py`** - Unit Tests
- **Coverage**: 
  - Pydantic model validation
  - Tool formatting for prompts
  - Function call parsing logic
  - Multi-turn conversation handling
  - Error handling scenarios

#### 2. **`test_function_calling_integration.py`** - Integration Tests  
- **Coverage**:
  - Live API endpoint testing
  - Emotion-specific behavior validation
  - OpenAI client compatibility
  - Streaming response handling
  - Performance benchmarks

#### 3. **`test_function_calling_quick.py`** - Quick Validation
- **Coverage**:
  - Basic model instantiation
  - Core parsing functionality
  - Game theory scenarios
  - Prompt generation

### Function Calling Test Scenarios

#### Game Theory Function Calling
```python
def test_game_theory_scenario():
    """Test prisoner's dilemma decision making with functions."""
    func_def = FunctionDefinition(
        name="make_decision",
        description="Make a decision in a prisoner's dilemma",
        parameters={
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": ["cooperate", "defect"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": ["decision"]
        }
    )
```

#### Emotion-Specific Behavior Testing
```python
def test_emotion_difference_in_function_calls():
    """Compare function calling patterns between emotions."""
    # Same prompt to happiness vs anger servers
    # Analyze behavioral differences in tool usage
```

### Running Function Calling Tests

```bash
# Quick validation (no server required)
python tests/test_function_calling_quick.py

# Complete test suite (requires running servers)
python tests/run_function_calling_tests.py

# Individual categories
pytest tests/test_function_calling.py -v              # Unit tests
pytest tests/test_function_calling_integration.py -v # Integration tests
pytest tests/test_function_calling_integration.py -m performance -v # Performance
```

**Server Requirements for Function Calling Tests**:
- **Port 8000**: Happiness emotion server  
- **Port 8001**: Anger emotion server

---

## 📏 Long Input Handling Tests

**Purpose**: Test variable input lengths and context window behavior

### Test Coverage
- **Variable Input Lengths**: Tests with inputs from 100 to 25,000+ characters
- **Multi-turn Conversations**: Context accumulation testing
- **Context Window Limits**: Behavior at context boundaries
- **Streaming with Long Input**: Streaming response performance
- **Concurrent Long Requests**: Multiple simultaneous long inputs
- **Performance Metrics**: Response time measurements

### Performance Tests (`test_input_length_performance.py`)
- **Input Length Progression**: Systematic testing of increasing input sizes
- **Content Type Comparison**: Different patterns (realistic, structured, code)
- **Concurrent Load Testing**: Performance under simultaneous requests
- **Streaming Performance**: First-chunk and total response timing
- **Lightweight Testing**: Can run against existing server instance

### Running Long Input Tests

```bash
# Run performance tests against existing server (lightweight)
python ../run_long_input_tests.py --test-type performance --direct

# Run full integration tests (requires GPU, starts own server)
python ../run_long_input_tests.py --test-type integration

# Run all long input tests
python ../run_long_input_tests.py --test-type full
```

---

## 🏃 Quick Start Commands

### **Development Workflow**

```bash
# After making changes - quick validation
make test-quick

# Full test suite
make test

# Before committing - complete CI pipeline
make ci
```

### **Makefile Commands**

#### Testing Commands
- `make test` - Run all tests
- `make test-unit` - Unit tests only (fast)
- `make test-integration` - Integration tests (requires GPU)
- `make test-coverage` - Generate coverage report
- `make test-quick` - Alias for test-unit

#### Code Quality
- `make lint` - Check code style
- `make format` - Auto-format code
- `make type-check` - Run type checker

#### Development
- `make server` - Start test server (happiness)
- `make server-anger` - Start server with anger
- `make clean` - Clean generated files
- `make watch` - Auto-run tests on file changes

---

## ⚙️ Test Configuration

### Environment Variables
- `TEST_MODEL_PATH`: Override default model path
- `VLLM_GPU_MEMORY_UTILIZATION`: Set GPU memory usage (default: 0.5 for tests)

### Pytest Markers
- `@pytest.mark.gpu`: Tests requiring GPU
- `@pytest.mark.slow`: Slow-running tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.performance`: Performance benchmarking tests
- `@pytest.mark.function_calling`: Function calling feature tests

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

# Run only function calling tests
pytest -m function_calling -v
```

---

## 🔍 Writing New Tests

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

### Stress Test Example
```python
def test_custom_stress_scenario():
    """Custom stress test scenario"""
    config = TestConfig(
        server_url="http://custom-server:8000/v1",
        model_name="custom-model",
        max_concurrent_requests=50,
        hang_timeout=600
    )
    
    suite = StressTestSuite(config)
    results = suite.run_all_tests()
    assert results.success_rate > 0.8
```

---

## 🚨 Troubleshooting

### Common Issues

#### 1. **GPU Memory Error**
```bash
export VLLM_GPU_MEMORY_UTILIZATION=0.3
make test-integration
```

#### 2. **Port Already in Use**
```bash
# Use different port
TEST_PORT=8766 pytest test_integration_server.py
```

#### 3. **Model Not Found**
```bash
export TEST_MODEL_PATH=/path/to/your/model
make test
```

#### 4. **Server Not Responding** (Stress Tests)
```bash
# Check if server is running
curl http://localhost:8000/health

# Verify correct URL and port
python run_stress_tests.py --server-url http://localhost:8001/v1
```

#### 5. **Import Errors**
```bash
# Ensure you're in the project root
cd ${USER_HOME}/LLM_EmoBehav_game_theory
make test
```

#### 6. **Function Calling Tests Failing**
- **Server Not Running**: Integration tests skip automatically
- **Parsing Failures**: Model response format unexpected
- **Timeout**: Increase timeout for slow model responses

### Debug Mode
```bash
# Run with debug output
pytest -v -s --log-cli-level=DEBUG

# Test specific parsing issue
python -c "from openai_server.server import parse_tool_calls_from_response; print(parse_tool_calls_from_response('test_input'))"

# Check server health
curl http://localhost:8000/health
curl http://localhost:8001/health

# Verbose stress test output
python run_stress_tests.py --verbose
```

---

## 📊 Coverage Goals

**Target coverage: 80%+ for core functionality**

Key areas:
- ✅ Request validation
- ✅ Response generation  
- ✅ Emotion vector application
- ✅ Error handling
- ✅ Concurrent request handling
- ✅ Streaming responses
- ✅ Function calling integration
- ✅ Graceful degradation under load

---

## 🔄 Continuous Integration

### GitHub Actions Integration
Tests are automatically run on:
- Push to `openai_server/**` paths
- Pull requests affecting server code
- Manual workflow dispatch

#### CI Stages
1. **Lint & Format**: Code style checks
2. **Unit Tests**: Fast tests on multiple Python versions
3. **Integration Tests**: GPU tests on main branch
4. **Documentation**: Auto-generated docs

### Local CI Simulation
```bash
make ci  # Runs the same checks as GitHub Actions
```

### CI/CD Integration Examples

#### GitHub Actions Example
```yaml
- name: Run Function Calling Tests
  run: |
    # Start test servers
    python -m openai_server --model $MODEL_PATH --emotion happiness --port 8000 &
    python -m openai_server --model $MODEL_PATH --emotion anger --port 8001 &
    
    # Wait for servers to start
    sleep 30
    
    # Run tests
    python openai_server/tests/run_function_calling_tests.py
```

#### Stress Test Integration
```bash
# Run tests and check exit code
python run_stress_tests.py --quick
if [ $? -eq 0 ]; then
    echo "Server stress tests passed"
else
    echo "Server stress tests failed"
    exit 1
fi
```

---

## 📈 Performance Benchmarks & Success Criteria

### **Unit Tests**
- **Target**: < 1 second per test
- **Coverage**: 80%+ of core functions

### **Integration Tests**
- **Target**: < 30 seconds total runtime
- **Success Rate**: 100% with running server

### **Stress Tests**
- **Success Rate**: 90%+ under normal load
- **Response Time**: < 5s for simple requests
- **Hang Detection**: 0 hangs under normal conditions
- **Resource Usage**: < 80% CPU/GPU utilization

### **Function Calling Tests**
- **Function Call Latency**: < 10 seconds
- **Concurrent Requests**: 5+ simultaneous without failures
- **Parsing Success Rate**: > 95% for valid JSON responses
- **Memory Usage**: Stable during extended testing

### **Long Input Tests**
- **Context Window**: Support up to model limits
- **Performance Degradation**: < 2x slowdown for 10x input length
- **Memory Efficiency**: Linear scaling with input length

---

## 🤝 Contributing

### Adding New Tests

1. **Write tests for new features**
2. **Ensure all tests pass**: `make test`
3. **Check coverage**: `make test-coverage`
4. **Run linter**: `make lint`
5. **Format code**: `make format`

### Test Categories to Add To

1. **Unit Tests**: Add to `test_unit_server.py` or `test_function_calling.py`
2. **Integration Tests**: Add to `test_integration_server.py` or `test_function_calling_integration.py`
3. **Stress Tests**: Add to `stress_test_suite.py` or create new load patterns in `load_generators.py`
4. **Performance Tests**: Add benchmarks to appropriate test files

### For Stress Testing Contributions

To add new test scenarios:

1. Add test methods to `stress_test_suite.py`
2. Implement load patterns in `load_generators.py`
3. Add hang detection logic to `hang_detector.py`
4. Update report generation in `stress_report_generator.py`

---

## 🎯 Best Practices

### Development Workflow
1. **Start Small**: Begin with quick tests before running full suite
2. **Always run tests before committing**:
   ```bash
   make test-quick  # At minimum
   make ci         # Ideally
   ```
3. **Write tests for new features**:
   - Add unit tests for new functions
   - Add integration tests for new endpoints
   - Add stress tests for performance-critical features

### Testing Strategy
4. **Keep tests fast**: Unit tests should run in < 1 second each
5. **Use mocks for external dependencies**
6. **Monitor Resources**: Watch GPU/CPU usage during tests
7. **Regular Testing**: Run stress tests after configuration changes
8. **Save Results**: Keep test reports for performance trending

### Quality Assurance  
9. **Maintain test coverage**: Target 80%+ coverage
10. **Review Recommendations**: Act on specific suggestions in reports
11. **Test edge cases**: Include error conditions and boundary cases
12. **Validate performance**: Set and monitor performance benchmarks

---

## 📁 Test File Organization

```
openai_server/tests/
├── README.md                           # This comprehensive guide
├── conftest.py                         # Shared fixtures and configuration
├── pytest.ini                         # Pytest configuration
│
├── # Core Test Files
├── test_unit_server.py                 # Unit tests (fast, no dependencies)
├── test_integration_server.py          # Integration tests (requires server)
├── test_openai_server.py              # Legacy basic functionality tests
├── test_integrated_openai_server.py   # Legacy integration scenarios
│
├── # Function Calling Tests
├── test_function_calling.py            # Function calling unit tests
├── test_function_calling_integration.py # Function calling integration tests
├── test_function_calling_quick.py      # Quick function calling validation
├── run_function_calling_tests.py       # Function calling test runner
│
├── # Long Input & Performance Tests
├── test_long_input_handling.py         # Long input integration tests
├── test_input_length_performance.py    # Performance benchmarks
│
├── # Stress Testing Framework
├── stress_test_suite.py                # Main stress test orchestrator
├── graceful_test_suite.py              # Graceful degradation specific tests
├── stress_report_generator.py          # Report generation utilities
├── load_generators.py                  # Load generation utilities
├── hang_detector.py                    # Hang detection utilities
├── server_monitor.py                   # Server monitoring utilities
│
├── # Test Utilities & Management
├── test_server_connectivity.py         # Connection testing
├── test_server_management.py           # Server lifecycle management
├── test_manage_servers.py              # Multi-server management
└── run_all_tests.py                    # Master test runner script
```

---

**The OpenAI server now has comprehensive test coverage ensuring reliability, performance, compatibility, and robust function calling capabilities with the emotion-controlled neural manipulation system!** 🎉