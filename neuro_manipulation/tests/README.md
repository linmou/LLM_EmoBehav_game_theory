# Neuro Manipulation Tests

This directory contains tests for the neuro_manipulation package. The tests are designed to ensure that the package works correctly with various models and prompt formats.

## Test Structure

- **Unit Tests:** Testing individual components in isolation
- **Integration Tests:** Testing how components work together

## Model Testing Strategy

The tests are designed to work with multiple LLM models to ensure compatibility across different architectures:

1. **Llama 2:** `meta-llama/Llama-2-7b-chat-hf`
2. **Llama 3:** `meta-llama/Llama-3.1-8B-Instruct` 
3. **Mistral:** `mistralai/Mistral-7B-Instruct-v0.3`
4. **ChatGLM:** `THUDM/glm-4-9b`

Each test class instantiates tokenizers for all supported models and runs tests against each model. This ensures that our prompt formatting and other functionality works consistently across different LLM architectures.

### Integration Test Implementation

For integration tests, we:

1. Initialize tokenizers for each model in the `setUp` method
2. Create a PromptFormat instance for each tokenizer
3. Use subtests to run tests for each model separately
4. Extract the model name from the tokenizer using `tokenizer.name_or_path` in the wrapper

This approach ensures that each model is tested independently and any model-specific issues are identified immediately.

### Main Test Files

- `test_prompt_format.py`: Tests for basic prompt formatting functionality
- `test_prompt_format_integration.py`: Integration tests for prompt formatting with wrappers
- `test_model_layer_detector.py`: Tests for automatic layer detection in various model architectures

### Running Tests

```bash
# Run all tests
python -m unittest discover

# Run a specific test file
python -m unittest neuro_manipulation.tests.test_prompt_format

# Run a specific test class
python -m unittest neuro_manipulation.tests.test_prompt_format.TestPromptTemplates

```

### Common Test Issues

- **Missing model name**: Ensure the model name is passed correctly to the `PromptFormat.build` method
- **Format expectations**: Different models use different chat templates, so test assertions should be flexible
- **Tokenizer limitations**: Some models have limitations on how system prompts are handled
- **Layer detection**: Some models may have non-standard architectures requiring special handling:
  - RWKV uses attention-free mechanisms
  - ChatGLM has a unique transformer structure
  - Deeply nested models require careful traversal


## Experiment test

# ModelLayerDetector Tests

This directory contains tests for the `ModelLayerDetector` module, which is responsible for identifying and extracting model layers from various transformer architectures.

## `test_model_layer_detector.py`

This test suite validates that the `ModelLayerDetector` can correctly identify model layers across different model architectures:

### Test Cases:

1. **Small Models** - Tests with lightweight models like GPT-2 and OPT-125M
2. **ChatGLM Models** - Specific test for ChatGLM architecture (skipable)
3. **RWKV Models** - Specific test for RWKV architecture (skipable)
4. **Custom Models** - Tests with custom-built transformer architectures to ensure robustness
5. **vLLM Integration** - Tests with vLLM models, specifically Llama-3.1-8B-Instruct

### Running Tests

```bash
# Run all tests
python -m unittest neuro_manipulation/tests/test_model_layer_detector.py

# Run a specific test
python -m unittest neuro_manipulation/tests/test_model_layer_detector.py TestModelLayerDetector.test_vllm_model_layers
```

### Requirements

- Most tests require GPU with CUDA
- Tests for large models will be skipped if:
  - CUDA is not available
  - Specific packages (like vLLM) are not installed
  - Running in CI environment

### vLLM Model Structure

The vLLM test specifically validates that `ModelLayerDetector` can navigate the nested structure of vLLM models to find the correct layers path:

```
self.llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
```

This test ensures that the `ModelLayerDetector` works correctly with representation engineering techniques that modify model layers in vLLM backends.
