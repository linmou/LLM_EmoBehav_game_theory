# Generic Probes (`delta_activation_engine/prompts/probes_texts.py`)
Last updated: 2024-03-19 (working copy)

## Purpose
Provides the canonical set of five neutral instruction templates used across baseline and chat pipelines. Mirrors legacy `delta_activations.py` behavior for comparability.

## Implementation Walkthrough
- `get_generic_probes()`: returns a static list of five instruction strings. Each contains `{task}` and `{input}` placeholders but is used verbatim (no formatting) in current pipelines.

## Key Logic
- Fixed ordering and count (five probes) to keep runs reproducible and comparable across models/backends.
- Probes are generic and non-task-specific; they act as neutral prompts to sample baseline representations.

## Dependencies
- None beyond standard library.

## Potential Issues / Gaps
- Placeholders are not interpolated; the literal `{task}`/`{input}` tokens remain in prompts.
- Set is static; no mechanism to extend or randomize without code change.

## Usage Example
```python
from delta_activation_engine.prompts.probes_texts import get_generic_probes
probes = get_generic_probes()
for p in probes:
    print(p[:40])
```
