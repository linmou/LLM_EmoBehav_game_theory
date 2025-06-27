# Prompt Wrapper

The `PromptWrapper` class is a key component in the prompt formatting system that helps with the construction of prompts for specific use cases like game theory experiments.

## Overview

The `PromptWrapper` class wraps a `PromptFormat` instance and provides a simplified interface for generating prompts with event descriptions and options. It handles the construction of system prompts and user messages in a consistent way.

## Class Hierarchy

- **PromptWrapper**: Base wrapper that generates standard prompts.
- **ReactionPromptWrapper**: Specialization for prompting user reactions.
- **GameReactPromptWrapper**: Specialization for game theory experiments using a ReAct-style format.

## Usage

```python
from transformers import AutoTokenizer
from neuro_manipulation.prompt_formats import PromptFormat
from neuro_manipulation.prompt_wrapper import PromptWrapper

# Initialize tokenizer and prompt format
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
prompt_format = PromptFormat(tokenizer)

# Create prompt wrapper
wrapper = PromptWrapper(prompt_format)

# Generate a prompt
event = "You are at a party"
options = ["Go talk to people", "Stay in a corner"]
user_message = "What should I do?"
prompt = wrapper(event, options, user_message)
```

## Implementation Details

### Model Name Detection

The `PromptWrapper` utilizes a `PromptFormat` instance, which is initialized with a tokenizer corresponding to a specific model. The `PromptFormat` object then handles model-specific formatting internally.

```python
def __call__(self, event, options, user_messages):
    # The self.prompt_format object (initialized with a model-specific tokenizer)
    # handles the prompt building.
    return self.prompt_format.build(self.system_prompt(event, options), self.user_messages(user_messages))
```

### System Prompt Formatting

The system prompt is formatted using the following template:

```python
system_prompt_format = 'You are an average American.'
```

For `GameReactPromptWrapper`, the template is:

```python
system_prompt_format = "You are Alice. You are in the following situation: {event}. "
```

### Options Formatting

Options are formatted with numbering:

```
Option 1. Go talk to people
Option 2. Stay in a corner
```

### GameReactPromptWrapper Response Format

The `GameReactPromptWrapper` includes instructions for the model to respond in a specific JSON format:

```python
def format_instruction(self):
    return f"response in json format, with the following structure: {self.response_format.example()}"
```

## Integration with Multiple Models

The wrapper works with multiple model architectures:

1. **Llama 2**: `meta-llama/Llama-2-7b-chat-hf`
2. **Llama 3**: `meta-llama/Llama-3.1-8B-Instruct` 
3. **Mistral**: `mistralai/Mistral-7B-Instruct-v0.3`

Each model has its own chat template, which is applied by the underlying `PromptFormat` class.

## Testing

The wrapper is tested in the integration tests:

```bash
python -m unittest neuro_manipulation.tests.test_prompt_format_integration
```

These tests verify that the wrapper correctly integrates with the prompt format and generates valid prompts for each supported model. 