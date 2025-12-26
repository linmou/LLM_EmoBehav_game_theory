# run_job (`delta_activation_engine/pipelines/runner.py`)
Last updated: 2024-03-19 (working copy)

## Purpose
Baseline orchestrator that computes delta activations without chat templating. It hashes probes, collects baseline vectors, computes steered vectors for each emotion×intensity, and writes artifacts.

## Implementation Walkthrough
1) Build output directory `<output_dir>/<model>_<timestamp>/`; hash generic probes for metadata.
2) Compute `baseline_vec = backend.get_repr(probes, steered=False)` and save to `baseline.npz`.
3) Persist `metadata.json` with job config snapshot, probe hash, timestamp, and backend metadata.
4) For each emotion and intensity, call `backend.get_repr(..., steered=True, emotion=emo, intensity=it)`, compute `delta = steered_vec - baseline_vec`, and save to `deltas/emotion=<emo>_int=<it>.npz`.

## Key Logic
- Probe set sourced from `prompts/probes_texts.get_generic_probes()`; no formatting applied to `{task}`/`{input}` placeholders.
- Uses SHA-256 hashing to fingerprint probe texts for reproducibility.
- Baseline vector reused for all deltas to ensure consistent subtraction.

## Dependencies
- `BaseBackend` contract (expecting `get_repr` and `get_run_metadata`).
- `DeltaActivationJobConfig` for model path/emotions/intensities/config snapshots.
- IO helpers: `ensure_dir`, `save_npz_vector`, `save_json`.
- Probe provider: `get_generic_probes`.

## Potential Issues / Gaps
- No parallelism; iterates emotions×intensities sequentially.
- No validation that probes are unique or meaningful (placeholders remain).
- Output directory naming based on timestamp; concurrent runs may collide if started in same second with identical model path.

## Usage Example
```python
from delta_activation_engine.config import load_job_config_from_yaml
from delta_activation_engine.backends.hf import HFBackend
from delta_activation_engine.pipelines.runner import run_job

cfg = load_job_config_from_yaml("config/delta_activations/qwen2_5b_default.yaml")
backend = HFBackend(cfg)
out_dir = run_job(cfg, backend)
print("Artifacts in", out_dir)
```
