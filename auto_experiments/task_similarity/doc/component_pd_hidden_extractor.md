# Component: `pd_hidden_extractor.py`

## Purpose and Responsibility

`pd_hidden_extractor.py` implements **span-based hidden state pooling** for PD prompts. It focuses on:

- Locating the Assistant answer span within a prompt.  
- Computing mean hidden representations over that span for selected transformer layers.  
- Providing clear runtime assertions when prompts are malformed or truncated.

This module is PD-specific and intentionally does not modify shared representation infrastructure.

## Core Function: `collect_answer_means`

Signature:

```python
def collect_answer_means(
    model,
    tokenizer,
    prompts: Sequence[str],
    layers: Sequence[int],
    max_length: int,
    batch_size: int = 8,
    span: str = "assistant",
) -> Dict[int, np.ndarray]:
    ...
```

Returns a mapping:

- `layer_idx -> np.ndarray` with shape `(num_prompts, hidden_dim)`.

### Span Detection

Internally, the helper `_find_assistant_start(prompt: str) -> int`:

- Uses `prompt.rfind("Assistant:")` to locate the **last** occurrence of `"Assistant:"`.  
- Asserts that this is also the **first** occurrence to catch any unexpected duplications.  
- Returns the character index where `"Assistant:"` begins.

For each prompt:

- If `span == "assistant"`:
  - The prefix is everything **before** `"Assistant:"`; the span starts at that marker.  
- If `span == "option"`:
  - The prefix is everything up to the first character of the option label (`"A)"` or `"B)"`) following `"Assistant:"`.  
  - This effectively excludes the `"Assistant:"` token from the span.

The prefix text is tokenized separately (with `add_special_tokens=False` and truncation to `max_length`) to measure how many tokens belong to the prefix; this becomes `prefix_len`.

### Model Forward and Hidden-State Pooling

For batched prompts:

1. Tokenize the full prompts with padding and truncation to `max_length`.  
2. Move tensors to the model's device.  
3. Run the model with `output_hidden_states=True`.  
4. For each example in the batch:
   - Compute `seq_len = attention_mask[i].sum()`.  
   - Assert `prefix_len < seq_len`:
     - If not, the Assistant answer span has been truncated away, and an assertion is raised with a clear message.  
   - Set `answer_start_tok = prefix_len`.  
   - For each requested layer:
     - Index into `outputs.hidden_states[layer + 1][i, :seq_len, :]`.  
     - Slice from `answer_start_tok:` to get only the answer tokens.  
     - Average over the token dimension to get a `(hidden_dim,)` vector.  
     - Accumulate per layer.

Finally, vectors per layer are stacked into `(num_prompts, hidden_dim)` arrays and returned.

### Edge Cases and Guards

- If `span` is not `"assistant"` or `"option"`, a `ValueError` is raised.  
- If `prompts` is empty, the function returns empty `(0, 0)` arrays per layer to keep callers simple.  
- Several `assert` statements ensure:
  - The prompt contains exactly one `"Assistant:"` marker.  
  - The prefix length is non-zero and less than the effective sequence length.  
  - The answer span yields at least one token after truncation.

`tests/test_pd_hidden_extractor.py` exercises these guarantees using dummy tokenizers/models:

- `test_collect_answer_means_basic` validates that the pooled vectors have the expected shapes and are non-trivial.  
- `test_collect_answer_means_truncation_asserts` constructs a prompt whose answer span is fully truncated and checks that the appropriate assertion is raised.

## Dependencies and Interactions

- **Inputs**:
  - A HuggingFace-compatible causal LM that supports `output_hidden_states=True`.  
  - A tokenizer whose tokenization is consistent between prefix and full prompts.

- **Consumers**:
  - `run_pd_defection_experiment.train_pd_repreader`:
    - Uses `collect_answer_means` with `span="option"` to build representations for defection vs cooperation prompts.  
  - Potential future variants of representation learning (e.g., different span modes) can reuse this helper without touching the global RepReader code.

## Potential Issues and Improvements

- The current implementation assumes that the PD prompts always contain exactly one `"Assistant:"` marker and that the option label immediately follows (possibly after a single space). If the prompt format changes, this module should be updated in sync.

- `max_length` is a crucial hyperparameter: too small and assertions will fire; too large and unnecessary computation is spent. The tests help catch the truncation failure mode.

## Usage Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_experiments.task_similarity.pd_hidden_extractor import collect_answer_means

model_name = "/data/home/.../Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True)

prompts = [
    "User: ...\nChoices:\nA) ...\nB) ...\nAssistant: A) Confess",
    "User: ...\nChoices:\nA) ...\nB) ...\nAssistant: B) Stay silent",
]

layer_ids = [8, 9, 10]
means = collect_answer_means(
    model=model,
    tokenizer=tokenizer,
    prompts=prompts,
    layers=layer_ids,
    max_length=256,
    batch_size=2,
    span="option",
)

vec_layer8 = means[8]  # shape (2, hidden_dim)
```

This is essentially what `train_pd_repreader` does internally for its training and test datasets.
