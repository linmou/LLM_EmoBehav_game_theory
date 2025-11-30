# DeltaProbesPromptWrapper (`delta_activation_engine/prompts/wrappers.py`)
Last updated: 2024-03-19 (working copy)

## Purpose
Callable wrapper that feeds probe text through `PromptFormat` to apply the model’s chat template. Used by `DeltaProbesDataset` and the chat pipeline.

## Implementation Walkthrough
- Constructor stores `prompt_format`, default user message text, `enable_thinking` flag, and optional `system_prompt`.
- `__call__(context=None, question, answer=None, options=None)`: concatenates context + question when context exists, otherwise uses question alone. Delegates to `prompt_format.build(system_prompt, [user_text], [], images=None, enable_thinking=enable_thinking)` and returns the rendered prompt string.

## Key Logic
- Simple two-turn chat rendering: empty assistant history, optional thinking mode, optional system prompt.
- Context inclusion is just newline concatenation; no additional formatting safeguards.

## Dependencies
- `neuro_manipulation.prompt_formats.PromptFormat` contract (expects `.build`).

## Potential Issues / Gaps
- No guard against empty `question`; will pass an empty string to PromptFormat.
- No support for multi-turn histories beyond a single user message.
- Does not expose choices/options; they are ignored.

## Usage Example
```python
from delta_activation_engine.prompts.wrappers import DeltaProbesPromptWrapper
from neuro_manipulation.prompt_formats import PromptFormat
from neuro_manipulation.utils import load_tokenizer_only

tokenizer, _ = load_tokenizer_only(model_name_or_path="/models/DUMMY", expand_vocab=False, auto_load_multimodal=True)
pf = PromptFormat(tokenizer)
wrapper = DeltaProbesPromptWrapper(pf, enable_thinking=False)
prompt = wrapper(question="Say hello")
print(prompt)
```
