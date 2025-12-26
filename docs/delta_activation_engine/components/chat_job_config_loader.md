# Chat Job Config Loader (`delta_activation_engine/config/chat_job.py`)
Last updated: 2024-03-19 (working copy)

## Purpose
Parses chat-template-aware YAML into `DeltaActivationChatJobConfig` and nested `PromptingConfig`, extending the baseline job schema with prompt controls.

## Implementation Walkthrough
- `_ensure_required` reused to assert required keys at both top level and inside `prompt_config` (requires `benchmark_name` and `task_type`).
- `load_chat_job_config_from_yaml(stream_or_path)`: reads YAML, validates mapping, enforces required fields (base job keys + `prompt_config`). Copies strings/lists/dicts, coercing intensities to float. Builds `PromptingConfig` with optional `probes`, `probe_source`, `enable_thinking` (bool or None). Performs the same basic type checks for emotions/intensities/config mappings before returning `DeltaActivationChatJobConfig`.

## Key Logic
- Supports two probe specification modes: explicit list (`probes`) or named source (`probe_source`, typically `generic`).
- Optional `enable_thinking` flag propagates into prompt construction but defaults to `None` if absent.

## Dependencies
- `yaml.safe_load` and dataclasses.
- Shares structure with `job.py` for core fields.

## Potential Issues / Gaps
- Does not validate that `probe_source` matches a known provider; downstream code silently falls back to generic probes.
- No validation that `probes` list is non-empty when provided.
- Nested configs are not deeply validated beyond shallow type checks.

## Usage Example
```python
from delta_activation_engine.config import load_chat_job_config_from_yaml
cfg = load_chat_job_config_from_yaml("config/delta_activations/chat_smoke.yaml")
print(cfg.prompt_config.benchmark_name, cfg.prompt_config.enable_thinking)
```
