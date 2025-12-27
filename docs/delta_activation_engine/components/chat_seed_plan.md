# build_seeded_chat_jobs (`delta_activation_engine/pipelines/chat_seed_plan.py`)
Last updated: 2025-12-01 (working copy)

## Purpose
Utility to generate multiple `DeltaActivationChatJobConfig` instances for different model paths and seeds without mutating the base config. Supports batch experiments.

## Implementation Walkthrough
- `_clone_prompt_config` deep-copies `PromptingConfig`, including optional probes list.
- `build_seeded_chat_jobs(base_cfg, model_paths, seeds, output_root)`: for each model path and seed, copies `loading_config`, injects `model_path` and `seed`, deep-copies `repe_eng_config`, sets `emotion_data_seed` to the same seed for RepE shuffling, clones `prompt_config`, and produces a new `DeltaActivationChatJobConfig` with shared emotions/intensities/output root.

## Key Logic
- Ensures immutability by copying nested dicts/lists; avoids side effects on the base config.
- Seeds are cast to `int` to prevent YAML/JSON string surprises and are reused for `emotion_data_seed` to vary RepE splits across jobs.

## Dependencies
- `DeltaActivationChatJobConfig` and `PromptingConfig` dataclasses.
- Python `deepcopy` for RepE config copying.

## Potential Issues / Gaps
- No validation that provided seeds/model paths are unique; caller must manage collisions.
- `output_root` is reused for all jobs; downstream callers must ensure directory uniqueness if running in parallel.

## Usage Example
```python
from delta_activation_engine.config import load_chat_job_config_from_yaml
from delta_activation_engine.pipelines.chat_seed_plan import build_seeded_chat_jobs

base_cfg = load_chat_job_config_from_yaml("config/delta_activations/chat_smoke.yaml")
jobs = build_seeded_chat_jobs(base_cfg, ["/models/a", "/models/b"], [0, 1], output_root="results/delta")
for job in jobs:
    print(job.model_path, job.loading_config["seed"])
```
