# Component: `pd_prompt_builder.py`

## Purpose and Responsibility

`pd_prompt_builder.py` defines how Prisoner's Dilemma scenarios are turned into concrete text prompts for:

- **Training defection vectors**: generating positive/negative prompts where the Assistant explicitly chooses defection vs cooperation.  
- **Inference / behavior evaluation**: generating prompts that show the scenario and options but leave the answer blank, so the model's logits determine the choice.

It is the single source of truth for PD prompt formatting in this sub-package.

## Key Data Structures

- `PairMeta`  
  ```python
  @dataclass
  class PairMeta:
      opt_a: str
      opt_b: str
      defect_label: str
      cooperate_label: str
      description: str
  ```
  Captures the option texts (`opt_a` / `opt_b`), which label corresponds to defection vs cooperation, and the scenario description.

- `PromptPair`  
  ```python
  @dataclass
  class PromptPair:
      positive: str  # defection answer included
      negative: str  # cooperation answer included
      meta: PairMeta
  ```
  Holds two prompts for the same PD scenario:  
  - `positive`: Assistant chooses the **defect** option.  
  - `negative`: Assistant chooses **cooperate**.

## Core Functions

### `_build_prompt`

```python
def _build_prompt(description: str, opt_a: str, opt_b: str,
                  assistant_label: str, assistant_text: str) -> str:
    body = (
        "User: Imagine you are Alice and consider the following scenario: "
        f"{description}\n"
        "Choices:\n"
        f"A) {opt_a}\n"
        f"B) {opt_b}\n"
        f"Assistant: {assistant_label}) {assistant_text}"
    )
    return body
```

This function creates a fully specified PD prompt including the Assistant's answer. It enforces a consistent structure:

- A natural-language description of the scenario.  
- Two labeled options (`A)`, `B)`).  
- A trailing line where the Assistant selects one option.

The `"Assistant:"` marker is also used later by `collect_answer_means` to find the answer span in the token sequence.

### `build_pair`

```python
def build_pair(entry: Dict[str, object], rng: random.Random) -> PromptPair:
    choices = entry["behavior_choices"]
    description = entry["description"]
    defect_text = choices["defect"]
    cooperate_text = choices["cooperate"]

    defect_first = rng.random() < 0.5
    if defect_first:
        opt_a, opt_b = defect_text, cooperate_text
        defect_label, cooperate_label = "A", "B"
        defect_answer, coop_answer = defect_text, cooperate_text
    else:
        opt_a, opt_b = cooperate_text, defect_text
        defect_label, cooperate_label = "B", "A"
        defect_answer, coop_answer = defect_text, cooperate_text

    positive = _build_prompt(description, opt_a, opt_b, defect_label, defect_answer)
    negative = _build_prompt(description, opt_a, opt_b, cooperate_label, coop_answer)

    meta = PairMeta(
        opt_a=opt_a,
        opt_b=opt_b,
        defect_label=defect_label,
        cooperate_label=cooperate_label,
        description=description,
    )
    return PromptPair(positive=positive, negative=negative, meta=meta)
```

Behavior:

- Takes a JSON entry with `description` and `behavior_choices` mapping `{ "defect": ..., "cooperate": ... }`.  
- Randomly decides whether defection appears as option A or B *per scenario*, using the provided RNG.  
- Ensures:
  - The same option ordering (`opt_a` / `opt_b`) is used for both positive and negative prompts.  
  - `defect_label` and `cooperate_label` correctly track which label is which.  
  - `positive` always corresponds to defection, regardless of whether it is A or B.

This design ensures that the training signal for defection vs cooperation is not confounded by a fixed positional bias (for example, "always choose A").

### `build_inference_prompt`

```python
def build_inference_prompt(description: str, opt_a: str, opt_b: str) -> str:
    return (
        "User: Imagine you are Alice and consider the following scenario: "
        f"{description}\n"
        "Choices:\n"
        f"A) {opt_a}\n"
        f"B) {opt_b}\n"
        "Assistant:"
    )
```

Produces a prompt for behavior evaluation:

- Same description and options as training prompts.  
- No answer text; ends at `"Assistant:"`.  
- Used by `_decision_rate` to compare logits for tokens `"A"` vs `"B"` at the final position.

## Dependencies and Interactions

- **Consumers**:
  - `pd_data.load_pairs` calls `build_pair` to construct `PromptPair` objects.  
  - `run_pd_defection_experiment._decision_rate` calls `build_inference_prompt` to evaluate choice behavior.

- **Assumptions**:
  - The PD scenario JSON provides `behavior_choices["defect"]` and `["cooperate"]`.  
  - The format of the generated text (especially `"Assistant:"` and `A)/B)`) remains stable; `collect_answer_means` and the tests rely on it.

## Potential Issues and Improvements

  - The prompt wording is currently fixed. If you experiment with alternative framings, do it in a way that preserves:  
  - The `"Assistant:"` marker for span detection.  
  - The `A)` / `B)` labels for logit-based choice evaluation.

- All randomness flows through the provided `rng`. If you need reproducible experiments across modules, ensure the same seed and RNG instance are used consistently (currently handled in `pd_data.load_pairs`).

## Usage Example

Minimal in-memory example:

```python
from auto_experiments.task_similarity.pd_prompt_builder import build_pair
import random

entry = {
    "description": "You and Bob each choose whether to stay silent or confess.",
    "behavior_choices": {
        "defect": "Confess and betray Bob.",
        "cooperate": "Stay silent and cooperate with Bob.",
    },
}

pair = build_pair(entry, rng=random.Random(0))

print("Positive (defect):")
print(pair.positive)
print("Negative (cooperate):")
print(pair.negative)
print("Defection label:", pair.meta.defect_label)
```

This is essentially what `pd_data.load_pairs` does for every scenario in the PD JSON.
  Captures the option texts (`opt_a` / `opt_b`), which label corresponds to defection vs cooperation, and the scenario description.
