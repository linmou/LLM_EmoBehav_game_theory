# Job Config Loader (`delta_activation_engine/config/job.py`)
Last updated: 2024-03-19 (working copy)

## Purpose
Parses baseline delta activation YAML into a strict `DeltaActivationJobConfig` dataclass with no implicit defaults. Ensures required keys exist before pipelines run.

## Implementation Walkthrough
- `_ensure_required(data, keys)`: raises `ValueError` if any required keys are missing or `None`.
- `load_job_config_from_yaml(stream_or_path)`: reads YAML from a path or file-like, enforces top-level mapping, checks required keys (`model_path`, `emotions`, `intensities`, `output_dir`, `loading_config`, `repe_eng_config`). Copies values into Python types, coercing intensities to `float` and dict/list copies for isolation. Validates types (`emotions` list of str, `intensities` list of float, configs as dicts), then returns `DeltaActivationJobConfig`.

## Key Logic
- Early failure on schema mismatches prevents expensive model loads with bad config.
- Coercion to concrete Python containers avoids shared references back into YAML structures.

## Dependencies
- `yaml.safe_load` for parsing.
- Dataclasses for structuring config.

## Potential Issues / Gaps
- No validation of file paths or directory existence.
- Does not deduplicate emotions/intensities; caller can supply duplicates.
- Type validation is shallow; nested `loading_config`/`repe_eng_config` contents are unchecked.

## Usage Example
```python
from delta_activation_engine.config import load_job_config_from_yaml
cfg = load_job_config_from_yaml("config/delta_activations/qwen2_5b_default.yaml")
print(cfg.model_path, cfg.emotions, cfg.intensities)
```
