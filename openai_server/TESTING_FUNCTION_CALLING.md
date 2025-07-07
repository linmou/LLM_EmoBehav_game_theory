# Function Calling Testing Suite

## Overview

Comprehensive test suite for the function calling implementation in the emotion-controlled OpenAI server. Tests cover unit functionality, integration scenarios, and performance characteristics.

## Test Files Created

### 1. **`test_function_calling.py`** - Unit Tests
- **Purpose**: Test individual components and functions
- **Coverage**: 
  - Pydantic model validation
  - Tool formatting for prompts
  - Function call parsing logic
  - Multi-turn conversation handling
  - Error handling scenarios

### 2. **`test_function_calling_integration.py`** - Integration Tests  
- **Purpose**: Test end-to-end functionality with running servers
- **Coverage**:
  - Live API endpoint testing
  - Emotion-specific behavior validation
  - OpenAI client compatibility
  - Streaming response handling
  - Performance benchmarks

### 3. **`test_function_calling_quick.py`** - Quick Validation
- **Purpose**: Fast validation without server dependencies
- **Coverage**:
  - Basic model instantiation
  - Core parsing functionality
  - Game theory scenarios
  - Prompt generation

### 4. **`run_function_calling_tests.py`** - Test Runner
- **Purpose**: Orchestrate all test suites
- **Features**:
  - Server health checking
  - Test categorization
  - Performance metrics
  - Comprehensive reporting

## Test Categories

### 🧪 **Unit Tests** (`pytest -m unit`)
- **Models**: Pydantic validation and serialization
- **Parsing**: Tool call extraction from responses
- **Formatting**: Tool definitions to prompt conversion
- **Edge Cases**: Error handling and malformed inputs

### 🔗 **Integration Tests** (`pytest -m integration`)
- **API Endpoints**: Live server testing
- **Emotion Servers**: Both happiness (8000) and anger (8001)
- **Client Compatibility**: OpenAI Python client integration
- **Multi-turn**: Conversation flows with tool usage

### ⚡ **Performance Tests** (`pytest -m performance`)
- **Latency**: Function call response times
- **Concurrency**: Multiple simultaneous requests
- **Throughput**: Requests per second measurements

## Key Test Scenarios

### 1. **Game Theory Function Calling**
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
    # Test complete workflow
```

### 2. **Emotion-Specific Behavior**
```python
def test_emotion_difference_in_function_calls():
    """Compare function calling patterns between emotions."""
    # Same prompt to happiness vs anger servers
    # Analyze behavioral differences in tool usage
```

### 3. **Multi-Turn Conversations**
```python
def test_multi_turn_conversation_with_tools():
    """Test complex conversations with tool execution."""
    # User request → Function call → Tool result → Continuation
```

### 4. **Parsing Robustness**
```python
def test_parse_tool_calls_from_response():
    """Test parsing various response formats."""
    # JSON format: {"tool_calls": [...]}
    # Function syntax: function_name(arguments)
    # Malformed inputs and error handling
```

## Running Tests

### **Quick Validation** (No server required)
```bash
python openai_server/tests/test_function_calling_quick.py
```

### **Complete Test Suite** (Requires running servers)
```bash
python openai_server/tests/run_function_calling_tests.py
```

### **Individual Test Categories**
```bash
# Unit tests only
pytest openai_server/tests/test_function_calling.py -v

# Integration tests (needs servers)  
pytest openai_server/tests/test_function_calling_integration.py -v

# Performance tests
pytest openai_server/tests/test_function_calling_integration.py -m performance -v
```

### **With Coverage**
```bash
pytest openai_server/tests/ --cov=openai_server --cov-report=html
```

## Test Configuration

### **pytest.ini Updates**
```ini
markers =
    performance: marks tests as performance tests
    function_calling: marks tests related to function calling functionality
```

### **Server Requirements**
Tests expect servers running on:
- **Port 8000**: Happiness emotion server
- **Port 8001**: Anger emotion server

Start servers with:
```bash
python -m openai_server --model /path/to/Qwen2.5-0.5B-Instruct --emotion happiness --port 8000
python -m openai_server --model /path/to/Qwen2.5-0.5B-Instruct --emotion anger --port 8001
```

## Test Results Validation

### **Expected Behaviors**
1. **Tool Definition Validation**: Proper Pydantic model validation
2. **Function Call Parsing**: Multiple formats supported reliably  
3. **Emotion Influence**: Different patterns between happiness/anger
4. **OpenAI Compatibility**: Standard client works seamlessly
5. **Performance**: Sub-10s latency for function calls

### **Success Criteria**
- ✅ All unit tests pass (no server dependencies)
- ✅ Integration tests pass with running servers
- ✅ Function calls parsed correctly from model responses
- ✅ Tools properly formatted in prompts
- ✅ Emotion context preserved through function calls
- ✅ OpenAI client compatibility maintained

## Debugging Failed Tests

### **Common Issues**
1. **Server Not Running**: Integration tests skip automatically
2. **Import Errors**: Path configuration in test files
3. **Parsing Failures**: Model response format unexpected
4. **Timeout**: Increase timeout for slow model responses

### **Debug Commands**
```bash
# Test specific parsing issue
python -c "from openai_server.server import parse_tool_calls_from_response; print(parse_tool_calls_from_response('test_input'))"

# Check server health
curl http://localhost:8000/health
curl http://localhost:8001/health

# Verbose test output
pytest openai_server/tests/test_function_calling_quick.py -v -s
```

## Performance Benchmarks

### **Target Metrics**
- **Function Call Latency**: < 10 seconds
- **Concurrent Requests**: 5+ simultaneous without failures
- **Parsing Success Rate**: > 95% for valid JSON responses
- **Memory Usage**: Stable during extended testing

### **Monitoring**
```python
def test_function_call_latency():
    start_time = time.time()
    # Make function call request
    latency = time.time() - start_time
    assert latency < 10.0, f"Too slow: {latency:.2f}s"
```

## Integration with CI/CD

### **GitHub Actions Example**
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

---

**The function calling feature now has comprehensive test coverage ensuring reliability, performance, and compatibility with the emotion-controlled neural manipulation system!**