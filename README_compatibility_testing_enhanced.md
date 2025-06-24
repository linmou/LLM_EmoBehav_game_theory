# Enhanced LangGraph & AG2 Compatibility Testing

This directory contains enhanced compatibility testing scripts for your OpenAI-compatible server implementation. The tests verify compatibility with LangGraph and AG2 (AutoGen) frameworks, with specific fixes for message format issues.

## 🚀 Quick Start

### Option 1: Complete Test (Recommended)
Automatically starts the server and runs all tests:

```bash
./run_complete_test.sh
```

### Option 2: Test with Running Server
If your server is already running in another terminal:

```bash
./test_only.sh
```

### Option 3: Manual Python Execution
```bash
conda activate llm_fresh
python run_complete_compatibility_test.py --model_path /path/to/model --model_name qwen2.5-0.5B-anger --emotion anger
```

## 📁 Files Overview

### Main Scripts
- **`run_complete_compatibility_test.py`** - Main Python testing script with LangGraph fixes
- **`run_complete_test.sh`** - Shell script that starts server and runs tests
- **`test_only.sh`** - Shell script for testing with pre-running server

### Original Files (for reference)
- **`test_langgraph_ag2_compatibility.py`** - Original test script (has known LangGraph issues)
- **`init_openai_server.py`** - Server initialization script
- **`init_openai_server.sh`** - Server startup shell script

## 🔧 Key Improvements

### LangGraph Compatibility Fixes

The enhanced test script (`run_complete_compatibility_test.py`) includes fixes for the message format issues that were causing LangGraph tests to fail:

1. **Proper Message Conversion**: Converts between OpenAI `ChatCompletionMessage` objects and LangChain message formats
2. **Enhanced State Management**: Properly handles LangChain's `StateGraph` message flow
3. **Tool Integration**: Improved tool node handling with correct message types

### Enhanced Features

- **Robust Server Management**: Starts and stops the server automatically with proper process cleanup
- **Process Tree Cleanup**: Uses `psutil` to ensure all server processes and children are properly terminated
- **Signal Handling**: Handles Ctrl+C and other signals to ensure clean shutdown
- **Port-based Cleanup**: Finds and kills processes using the server port as a fallback
- **Better Error Handling**: More detailed error reporting and troubleshooting
- **Comprehensive Testing**: Tests basic compatibility, streaming, concurrency, and advanced features
- **Result Persistence**: Saves detailed results to JSON file

## 🧪 Test Categories

### 1. Basic OpenAI Compatibility
- Server health check
- Models endpoint verification  
- Basic chat completion functionality

### 2. Advanced Features
- Streaming response support
- Parameter variation handling
- Concurrent request processing

### 3. LangGraph Integration (Enhanced)
- Basic LangGraph functionality with message conversion fixes
- Tool integration with proper message handling
- State graph execution

### 4. AG2 (AutoGen) Integration
- Basic AG2 agent functionality
- Group chat support
- Multi-agent conversations

## 📊 Understanding Results

### Test Output Format
```
✅ PASS - Test Name (X.XXs)
  Details: Success details
  
❌ FAIL - Test Name (X.XXs)  
  Details: Error information
```

### Summary Categories
- **Basic Compatibility**: Core OpenAI API compatibility
- **Advanced Features**: Streaming, concurrency, parameter handling
- **LangGraph**: LangGraph framework integration
- **AG2**: AutoGen framework integration

### Success Rates
- **100%**: Full compatibility - framework ready for production use
- **80-99%**: High compatibility - minor issues may exist
- **<80%**: Compatibility issues - review failed tests

## 🛠 Troubleshooting

### Common Issues

#### 1. Server Won't Start
```bash
# Check if port is in use
lsof -i :8000

# The new server management will automatically handle cleanup, but if needed:
# Kill existing process manually
pkill -f "init_openai_server"

# Check model path
ls -la /data/home/jjl7137/huggingface_models/Qwen/
```

#### 2. LangGraph Import Errors
```bash
# Install LangGraph dependencies
conda activate llm_fresh
pip install langgraph langchain langchain-core
```

#### 3. AG2 Import Errors
```bash
# Install AutoGen
conda activate llm_fresh
pip install pyautogen
```

#### 4. Model Loading Issues
- Verify model path in `run_complete_test.sh`
- Check model permissions and accessibility
- Ensure sufficient disk space and memory

### Environment Setup

```bash
# Ensure conda environment exists
conda env list

# Create environment if needed
conda create -n llm_fresh python=3.9

# Activate environment
conda activate llm_fresh

# Install dependencies
pip install openai requests psutil langgraph langchain pyautogen
```

## 📝 Configuration

### Model Configuration

Edit `run_complete_test.sh` to change model settings:

```bash
MODEL_PATH="/path/to/your/model"
MODEL_NAME="your-model-name"
EMOTION="anger"  # or other emotion
```

### Server Configuration

Edit `run_complete_compatibility_test.py` for different server settings:

```python
# Default settings
host = "localhost"
port = 8000
api_key = "token-abc123"
```

## 🔍 Advanced Usage

### Custom Test Parameters

```bash
python run_complete_compatibility_test.py \
    --model_path /custom/path/to/model \
    --model_name custom-model-name \
    --emotion neutral \
    --host 0.0.0.0 \
    --port 8080 \
    --output custom_results.json \
    --keep_server  # Optional: don't stop server after tests
```

### Test-Only Mode

```bash
# Skip server startup (assumes server is running)
python run_complete_compatibility_test.py \
    --model_path dummy \
    --model_name qwen2.5-0.5B-anger \
    --no_server
```

### JSON Results Analysis

The test saves detailed results to `compatibility_test_results.json`:

```python
import json

with open('compatibility_test_results.json', 'r') as f:
    results = json.load(f)

# Access summary
summary = results['summary']
print(f"Success Rate: {summary['success_rate']:.1%}")

# Access detailed results
for test in results['detailed_results']:
    if not test['success']:
        print(f"Failed: {test['test_name']} - {test['details']}")
```

## 🎯 Next Steps

### If All Tests Pass ✅
Your server is fully compatible! You can use it with:
- LangGraph applications using `base_url='http://localhost:8000/v1'`
- AG2 applications using `api_type='openai'` and `base_url='http://localhost:8000/v1'`

### If Some Tests Fail ⚠️
1. Review the specific error messages in the test output
2. Check the troubleshooting section above
3. Verify dependencies are installed correctly
4. Consider the failed test categories for impact assessment

### For Development 🔧
- The enhanced test script provides a solid foundation for testing new features
- Add new test cases by extending the `EnhancedCompatibilityTester` class
- Use the JSON results for automated CI/CD integration
