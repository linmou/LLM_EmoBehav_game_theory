# Prompt Format System

## Overview

The prompt format system in this project provides a way to format prompts for various language models in a consistent way. The system supports different model formats including Llama-2, Llama-3, and Mistral.

## Changes

The prompt format system was updated to use the tokenizer's built-in `apply_chat_template` method instead of manually formatting prompts. This change ensures that prompts are formatted according to the official chat template defined by the model providers.

## Classes

### PromptFormat

The `PromptFormat` class is the main entry point for the prompt format system. It uses the tokenizer's `apply_chat_template` method to format prompts.

```python
from transformers import AutoTokenizer
from neuro_manipulation.prompt_formats import PromptFormat

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Create prompt format
prompt_format = PromptFormat(tokenizer)

# Format a prompt
prompt = prompt_format.build(
    "meta-llama/Llama-2-7b-chat-hf",
    "You are a helpful assistant.",
    ["Hello, how are you?"],
    ["I'm doing well, thank you!"]
)
```

### OldPromptFormat (Deprecated)

The `OldPromptFormat` class is the original implementation that manually formatted prompts. It is now deprecated and should not be used in new code. It is kept for compatibility and testing purposes.

## Model-specific Format Classes

The system also includes model-specific format classes that define the format for each model type:

- `Llama2InstFormat`: Format for Llama-2 models
- `Llama3InstFormat`: Format for Llama-3 models
- `MistralInstFormat`: Format for Mistral models

These classes are used as fallbacks in case the tokenizer's `apply_chat_template` method fails.

## Integration with PromptWrapper

The `PromptFormat` class is designed to be used with the `PromptWrapper` class, which provides a higher-level API for generating prompts:

```python
from neuro_manipulation.prompt_formats import PromptFormat
from neuro_manipulation.prompt_wrapper import PromptWrapper

# Initialize tokenizer and prompt format
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
prompt_format = PromptFormat(tokenizer)

# Create prompt wrapper
wrapper = PromptWrapper(prompt_format)

# Generate a prompt
prompt = wrapper(
    "You are at a party",
    ["Go talk to people", "Stay in a corner"],
    "What should I do?"
)
```

## Testing

Tests for the prompt format system are located in the `neuro_manipulation/tests` directory. They validate:

1. Compatibility with different model formats
2. Handling different numbers of messages
3. Integration with the prompt wrapper system

To run the tests:

```bash
python -m unittest neuro_manipulation/tests/test_prompt_format.py
python -m unittest neuro_manipulation/tests/test_prompt_format_integration.py
``` 